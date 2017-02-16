#include <atomic>
#include <cassert>
#include <chrono>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#define FIXEDFLOAT_CODE 2
#define RECV_BUFFER_SIZE_F64 5000000
#define SEND_BUFFER_SIZE_F64 500000
#define HEADER_OFFSET_F64 2

using namespace tensorflow;
typedef std::chrono::time_point<std::chrono::high_resolution_clock> hr_clock;

Status LoadGraph(string graph_file_name, Session** session, std::vector<string>* vNames) {
  GraphDef graph_def;
  TF_RETURN_IF_ERROR(
    ReadBinaryProto(Env::Default(), graph_file_name, &graph_def));
  TF_RETURN_IF_ERROR(NewSession(SessionOptions(), session));
  TF_RETURN_IF_ERROR((*session)->Create(graph_def));
  int node_count = graph_def.node_size();
  std::cout << node_count << "\n";
  for (int i = 0; i < node_count; i++) {
    auto n = graph_def.node(i);
    std::cout << n.name() << "\n";
    if (n.name().find("nWeights") != std::string::npos) {
      vNames->push_back(n.name());
    }
  }

  return Status::OK();
}

void compute_statistics(std::string name,
                        std::atomic_ulong& pred_counter,
                        std::atomic_ulong& latency_sum_micros,
                        std::atomic_ulong& latency_sum_micros_squared,
                        const std::vector<tensorflow::uint64>& latencies) {
  std::vector<tensorflow::uint64> my_latencies;
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(15));
    my_latencies = latencies;
    std::sort(my_latencies.begin(), my_latencies.end());
    tensorflow::uint64 processed_reqs = pred_counter.load();
    tensorflow::uint64 total_batch_latency = latency_sum_micros.load();
    tensorflow::uint64 squared_total_batch_latency = latency_sum_micros_squared.load();
    // LOG(INFO) << "Batch latency vals: " << total_batch_latency << " " << squared_total_batch_latency << " " << processed_reqs;
    double mean_latency = (double) total_batch_latency / (double) processed_reqs;
    double std_latency = sqrt(((double) processed_reqs * (double) squared_total_batch_latency)  - pow((double) total_batch_latency, 2)) /
                         (((double) processed_reqs) * ((double) processed_reqs - 1));
    LOG(INFO) << tensorflow::strings::StrCat(name, " Mean Latency: ", mean_latency, " Standard deviation: ", std_latency,
        "  p99 latency: ", my_latencies[(int)(0.99 * my_latencies.size())], "  max latency: ", my_latencies[my_latencies.size() - 1],
        "  min latency: ", my_latencies[0]);
  }
}

struct header {
  uint32_t batch_id;
  uint32_t code;
  uint32_t num_inputs;
  uint32_t input_len;
};

struct BufferedServerS {};

typedef struct BufferedServerS* (*init_server)(char *);
typedef void (*server_free)(void *);
typedef void (*wait_for_con)(struct BufferedServerS*, double*, int, double*, int, double*, int, double*, int);
typedef struct header (*get_n_batch)(struct BufferedServerS*);
typedef void (*finish_batch)(struct BufferedServerS*, int, int);

class Model {
  public:
    Model(string model_path, int batch_size);
    // void predict_floats(double *data, int num_inputs, int input_len, double *output_buffer);
    std::pair<std::pair<hr_clock, uint64_t>,
              std::pair<cudaEvent_t, cudaEvent_t>>
        predict_floats(double *data, int num_inputs, int input_len, double *output_buffer);
    std::vector<double> predictions;
  private:
    Session *session;
    std::vector<std::pair<string, Tensor>> inputs;
    Tensor input;

};

Model::Model(string model_path, int batch_size) {
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return;
  }

  // Get variables to run and load graph
  std::vector<string> vNames;
  status = LoadGraph(model_path, &session, &vNames);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return;
  }

  tensorflow::TensorShape inputShape;
  inputShape.InsertDim(0, batch_size);
  inputShape.InsertDim(1, 299);
  inputShape.InsertDim(2, 299);
  inputShape.InsertDim(3, 3);

  input = Tensor(DT_FLOAT, inputShape);
  inputs.push_back(std::pair<string, Tensor>("Placeholder", input));
  std::vector<tensorflow::Tensor> outputs;

  status = session->Run(inputs, {"inception_v3/logits/predictions"}, {}, &outputs);
  predictions.resize(batch_size);
}

std::pair<std::pair<hr_clock, uint64_t>,
          std::pair<cudaEvent_t, cudaEvent_t>> Model::predict_floats(
// void Model::predict_floats(
    double *data, int num_inputs, int input_len, double *output_buffer) {
  auto dst = input.flat_outer_dims<float>().data();
  std::copy(data, data + num_inputs * input_len, dst);
  // for (int i = 0; i < num_inputs * input_len; i++) {
  //   dst[i] = (float) data[i];
  // }

  std::vector<tensorflow::Tensor> outputs; 
  // metrics
  auto before_predict = std::chrono::high_resolution_clock::now();
  cudaEvent_t start_predict;
  cudaEvent_t stop_predict;
  cudaEventCreate(&start_predict);
  cudaEventCreate(&stop_predict);
  cudaEventRecord(start_predict);
  session->Run(this->inputs, {"inception_v3/logits/predictions"}, {}, &outputs);
  cudaEventRecord(stop_predict);
  auto after_predict = std::chrono::high_resolution_clock::now();
  auto src = outputs[0].flat_outer_dims<float>().data();
  for (int i = 0; i < num_inputs; i++) {
    output_buffer[i] = (double) src[i];
  }
  auto after_copy = std::chrono::high_resolution_clock::now();
  uint64_t second_copy_duration = std::chrono::duration_cast<std::chrono::microseconds>(after_copy - after_predict).count();
  return std::make_pair(std::make_pair(before_predict, second_copy_duration),
                        std::make_pair(start_predict, stop_predict));
}

class BufferedServer {
  public:
    BufferedServer(char *ip_port, Model *model, void *handle);
    char *ip_port;
    Model *model;
    void *handle;
    struct BufferedServerS *obj;
    double *recv_buffer_one;
    double *recv_buffer_two;
    double *send_buffer_one;
    double *send_buffer_two;
    void serve_once(void);
    void handle_connection(void);
};

BufferedServer::BufferedServer(char *ip_port, Model* model, void *handle):
  ip_port(ip_port), model(model), handle(handle) {
  init_server init_fn = (init_server) dlsym(handle, "init_server");
  this->obj = (*init_fn)(ip_port);
  recv_buffer_one = (double *) malloc(RECV_BUFFER_SIZE_F64 * 8);
  recv_buffer_two = (double *) malloc(RECV_BUFFER_SIZE_F64 * 8);
  send_buffer_one = (double *) malloc(SEND_BUFFER_SIZE_F64 * 8);
  send_buffer_two = (double *) malloc(SEND_BUFFER_SIZE_F64 * 8);
}

void BufferedServer::serve_once() {
  wait_for_con wait_fn = (wait_for_con) dlsym(this->handle, "wait_for_connection");
  (*wait_fn)(this->obj,
          this->recv_buffer_one,
          RECV_BUFFER_SIZE_F64,
          this->send_buffer_one,
          SEND_BUFFER_SIZE_F64,
          this->recv_buffer_two,
          RECV_BUFFER_SIZE_F64,
          this->send_buffer_two,
          SEND_BUFFER_SIZE_F64);
  this->handle_connection();
}

void BufferedServer::handle_connection() {
  LOG(INFO) << "new connection (C++)\n";
  std::vector<double> preds;
  std::atomic_ulong pred_counter;
  pred_counter.store(0);
  std::vector<tensorflow::uint64> copy_latencies;
  std::vector<tensorflow::uint64> gpu_latencies;
  copy_latencies.reserve(10);
  gpu_latencies.reserve(10);
  std::atomic_ulong copy_latency_sum_micros;
  std::atomic_ulong gpu_latency_sum_micros;
  copy_latency_sum_micros.store(0);
  gpu_latency_sum_micros.store(0);
  std::atomic_ulong copy_latency_sum_micros_squared;
  std::atomic_ulong gpu_latency_sum_micros_squared;
  copy_latency_sum_micros_squared.store(0);
  gpu_latency_sum_micros_squared.store(0);
  std::thread copy_stats([&]{
    compute_statistics("Data copy time. ", pred_counter, copy_latency_sum_micros,
                       copy_latency_sum_micros_squared, copy_latencies);
  });
  std::thread gpu_stats([&]{
    compute_statistics("GPU kernel time. ", pred_counter, gpu_latency_sum_micros,
                       gpu_latency_sum_micros_squared, gpu_latencies);
  });

  bool shutdown = false;
  uint32_t current_batch = 1;
  double *cur_batch_recv_buffer = this->recv_buffer_one;
  double *cur_batch_send_buffer = this->send_buffer_one;
  get_n_batch batch_fn = (get_n_batch) dlsym(this->handle, "get_next_batch");
  finish_batch finish_fn = (finish_batch) dlsym(this->handle, "finish_batch");
  while (!shutdown) {
    struct header header = (*batch_fn)(this->obj);
    auto start_predict = std::chrono::high_resolution_clock::now();
    assert(header.batch_id == current_batch);
    assert(header.code == FIXEDFLOAT_CODE);
    // Check if sum of inputs is zero

    int num_inputs = header.num_inputs;
    int input_len = header.input_len;

    // this->model->predict_floats(cur_batch_recv_buffer + 2, num_inputs, input_len, cur_batch_send_buffer);
    auto results = this->model->predict_floats(
        cur_batch_recv_buffer + 2, num_inputs, input_len, cur_batch_send_buffer);
    auto before_predict = results.first.first;
    auto second_copy_duration = results.first.second;
    cudaEvent_t start = results.second.first;
    cudaEvent_t stop = results.second.second;
    // assert(preds.size() == num_inputs);

    // (*finish_fn)(this->obj, header->batch_id, preds.size());
    (*finish_fn)(this->obj, header.batch_id, num_inputs);
    // Flip buffers
    if (current_batch == 1) {
      current_batch = 2;
      cur_batch_recv_buffer = this->recv_buffer_two;
      cur_batch_send_buffer = this->send_buffer_two;
    } else if (current_batch == 2) {
      current_batch = 1;
      cur_batch_recv_buffer = this->recv_buffer_one;
      cur_batch_send_buffer = this->send_buffer_one;
    } else {
      LOG(ERROR) << "INVALID BATCH NUMBER: " << current_batch;
      return;
    }
    // Append to latencies
    pred_counter.fetch_add(1, std::memory_order::memory_order_relaxed);
    auto pred_duration = std::chrono::duration_cast<std::chrono::microseconds>(before_predict - start_predict).count();
    pred_duration += second_copy_duration;
    copy_latency_sum_micros.fetch_add(pred_duration, std::memory_order::memory_order_relaxed);
    copy_latency_sum_micros_squared.fetch_add(pred_duration * pred_duration, std::memory_order::memory_order_relaxed);
    copy_latencies.push_back(pred_duration);
    // cuda synchronize and record
    float elapsed = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    elapsed *= 1000.0; // interpret as microseconds
    // std::cout << "Elapsed time in GPU: " << elapsed << " ms\n";
    gpu_latency_sum_micros.fetch_add(elapsed, std::memory_order::memory_order_relaxed);
    gpu_latency_sum_micros_squared.fetch_add(elapsed * elapsed, std::memory_order::memory_order_relaxed);
    gpu_latencies.push_back(elapsed);
  }
}

int main(int argc, char* argv[]) {
  const string model_path(argv[1]);
  int batch_size = 16;
  void *myso = dlopen("/giulio-local/tf_serving/tensorflow/libbufferedrpc.so", RTLD_NOW);
  init_server init_fn = (init_server) dlsym(myso, "init_server");
  printf("%p\n", init_fn);
  server_free free_fn = (server_free) dlsym(myso, "server_free");
  printf("%p\n", free_fn);
  wait_for_con wait_fn = (wait_for_con) dlsym(myso, "wait_for_connection");
  printf("%p\n", wait_fn);
  get_n_batch batch_fn = (get_n_batch) dlsym(myso, "get_next_batch");
  printf("%p\n", batch_fn);
  finish_batch finish_fn = (finish_batch) dlsym(myso, "finish_batch");
  printf("%p\n", finish_fn);

  Model *model = new Model(model_path, batch_size);
  BufferedServer *server = new BufferedServer("0.0.0.0:6001", model, myso);
  server->serve_once();
  return 0;
}
