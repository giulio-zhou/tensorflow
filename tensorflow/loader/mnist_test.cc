#include <atomic>
#include <cassert>
#include <vector>
#include <dlfcn.h>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#define FIXEDFLOAT_CODE 2
#define RECV_BUFFER_SIZE_F64 1000000
#define SEND_BUFFER_SIZE_F64 100000
#define HEADER_OFFSET_F64 2

using namespace tensorflow;

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
    void predict_floats(double *data, int num_inputs, int input_len, double *output_buffer);
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
  inputShape.InsertDim(1, 28);
  inputShape.InsertDim(2, 28);

  input = Tensor(DT_FLOAT, inputShape);
  inputs.push_back(std::pair<string, Tensor>("Placeholder", input));
  std::vector<tensorflow::Tensor> outputs;

  status = session->Run(inputs, {"Softmax"}, {}, &outputs);
  predictions.resize(batch_size);
}

void Model::predict_floats(
    double *data, int num_inputs, int input_len, double *output_buffer) {
  auto dst = input.flat_outer_dims<float>().data();
  for (int i = 0; i < num_inputs * input_len; i++) {
    dst[i] = (float) data[i];
  }

  std::vector<tensorflow::Tensor> outputs; 
  session->Run(this->inputs, {"Softmax"}, {}, &outputs);
  auto src = outputs[0].flat_outer_dims<float>().data();
  for (int i = 0; i < num_inputs; i++) {
    output_buffer[i] = (double) src[i];
  }
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

  bool shutdown = false;
  uint32_t current_batch = 1;
  double *cur_batch_recv_buffer = this->recv_buffer_one;
  double *cur_batch_send_buffer = this->send_buffer_one;
  get_n_batch batch_fn = (get_n_batch) dlsym(this->handle, "get_next_batch");
  finish_batch finish_fn = (finish_batch) dlsym(this->handle, "finish_batch");
  while (!shutdown) {
    struct header header = (*batch_fn)(this->obj);
    assert(header.batch_id == current_batch);
    assert(header.code == FIXEDFLOAT_CODE);
    // Check if sum of inputs is zero

    int num_inputs = header.num_inputs;
    int input_len = header.input_len;

    this->model->predict_floats(cur_batch_recv_buffer + 2, num_inputs, input_len, cur_batch_send_buffer);
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
  }
}

int main(int argc, char* argv[]) {
  std::atomic_ulong latency_sum_micros;
  latency_sum_micros.store(0);
  std::atomic_ulong latency_sum_micros_squared;
  latency_sum_micros_squared.store(0);
  const string model_path(argv[1]);
  int batch_size = 512;
  int num_batches = 5000;
  void* a = (void*) &batch_size;
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
