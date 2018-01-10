add_library(xchainer_testing STATIC
  numeric.cc
  )

if(${ENABLE_TEST})
    add_executable(xchainer_testing_test
        numeric_test.cc
        )
    target_link_libraries(xchainer_testing_test
        xchainer
        xchainer_testing
        gtest_main)
    add_test(NAME xchainer_testing_test COMMAND xchainer_testing_test)
endif()
