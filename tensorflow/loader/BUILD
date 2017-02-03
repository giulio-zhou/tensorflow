cc_binary(
  name = "loader",
  srcs = ["loader.cc"],
  deps = [
      "//tensorflow/core:tensorflow",
  ]
)

cc_binary(
  name = "loader_cifar10",
  srcs = ["loader_cifar10.cc"],
  deps = [
      "//tensorflow/core:tensorflow",
  ]
)

cc_binary(
  name = "loader_inception",
  srcs = ["loader_inception.cc"],
  deps = [
      "//tensorflow/core:tensorflow",
  ]
)

cc_binary(
  name = "mnist_test",
  srcs = ["mnist_test.cc"],
  deps = [
      "//tensorflow/core:tensorflow",
      "libbufferedrpc.so",
  ]
)

cc_binary(
  name = "cifar10_test",
  srcs = ["cifar10_test.cc"],
  deps = [
      "//tensorflow/core:tensorflow",
      "libbufferedrpc.so",
  ]
)

cc_binary(
  name = "inception_test",
  srcs = ["inception_test.cc"],
  deps = [
      "//tensorflow/core:tensorflow",
      "libbufferedrpc.so",
  ]
)

# load("@io_bazel_rules_rust//rust:rust.bzl", "rust_library")
# 
# rust_library(
#   name = "buffered_rpc_lib",
#   srcs = ["buffered_rpc_rs/src/lib.rs"],
# )
cc_library(
  name = "libbufferedrpc.so",
  srcs = ["buffered_rpc_rs/target/release/libbufferedrpc.so"],
)
