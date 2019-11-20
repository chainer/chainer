include_directories("${PROJECT_SOURCE_DIR}")

add_executable(train_mnist
  mnist.cc
  train_mnist.cc
)
target_link_libraries(train_mnist
  chainerx
)
