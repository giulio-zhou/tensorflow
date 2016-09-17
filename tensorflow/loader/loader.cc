#include <atomic>
#include <vector>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

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

bool ParseInt32Flag(tensorflow::StringPiece arg, tensorflow::StringPiece flag,
                    int32* dst) {
  if (arg.Consume(flag) && arg.Consume("=")) {
    char extra;
    return (sscanf(arg.data(), "%d%c", dst, &extra) == 1);
  }

  return false;
}

bool ParseBoolFlag(tensorflow::StringPiece arg, tensorflow::StringPiece flag,
                   bool* dst) {
  if (arg.Consume(flag)) {
    if (arg.empty()) {
      *dst = true;
      return true;
    }

    if (arg == "=true") {
      *dst = true;
      return true;
    } else if (arg == "=false") {
      *dst = false;
      return true;
    }
  }

  return false;
}

int main(int argc, char* argv[]) {
  std::atomic_ulong latency_sum_micros;
  latency_sum_micros.store(0);
  std::atomic_ulong latency_sum_micros_squared;
  latency_sum_micros_squared.store(0);
  const string model_path(argv[1]);
  int batch_size = 512;
  int num_batches = 5000;

  // Initialize a tensorflow session
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Get variables to run and load graph
  std::vector<string> vNames;
  status = LoadGraph(model_path, &session, &vNames);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  std::cout << "did something\n";

  // Set up inputs and outputs
  tensorflow::TensorShape inputShape;
  inputShape.InsertDim(0, batch_size);
  inputShape.InsertDim(1, 28);
  inputShape.InsertDim(2, 28);

  Tensor sample_input(DT_FLOAT, inputShape);
  std::vector<std::pair<string, Tensor>> inputs;
  inputs.push_back(std::pair<string, Tensor>("Placeholder", sample_input));
  std::vector<tensorflow::Tensor> outputs; 

  status = session->Run(inputs, {"Softmax"}, {}, &outputs);
  tensorflow::uint64 start = tensorflow::Env::Default()->NowMicros();
  for (int i = 0; i < num_batches; i++) {
    tensorflow::uint64 start_time = tensorflow::Env::Default()->NowMicros();
    status = session->Run(inputs, {"Softmax"}, {}, &outputs);
    tensorflow::uint64 end_time = tensorflow::Env::Default()->NowMicros();
    // std::cout << i << " " << (end_time - start_time) << "\n";
    tensorflow::uint64 latency = end_time - start_time;
    latency_sum_micros.fetch_add(latency, std::memory_order::memory_order_relaxed);
    latency_sum_micros_squared.fetch_add(latency * latency, std::memory_order::memory_order_relaxed);
  }
  tensorflow::uint64 end = tensorflow::Env::Default()->NowMicros();
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }
  std::cout << outputs.size() << "\n";
  tensorflow::uint64 processed_reqs = num_batches * batch_size;
  tensorflow::uint64 total_batch_latency = latency_sum_micros.load() * batch_size;
  tensorflow::uint64 squared_total_batch_latency = latency_sum_micros_squared.load() * batch_size;
  LOG(INFO) << "Batch latency vals: " << total_batch_latency << " " << squared_total_batch_latency << " " << processed_reqs;
  double mean_latency = (double) total_batch_latency / (double) processed_reqs;
  double std_latency = sqrt(((double) processed_reqs * (double) squared_total_batch_latency)  - pow((double) total_batch_latency, 2)) /
	                   (((double) processed_reqs) * ((double) processed_reqs - 1));

  tensorflow::uint64 total_micros = (end - start);
  double total_seconds = total_micros / (1000.0 * 1000.0);
  double throughput = ((double) processed_reqs) / total_seconds;
  LOG(INFO) << tensorflow::strings::StrCat("Processed ", processed_reqs,
    " predictions in ", total_micros, " micros. Throughput: ",
    throughput, " Mean Latency: ", mean_latency, " Standard deviation: ", std_latency);
}
