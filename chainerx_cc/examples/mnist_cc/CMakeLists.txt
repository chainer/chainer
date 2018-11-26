add_executable(train_mnist
  train_mnist.cc
)
target_link_libraries(train_mnist
  chainerx
)
