add_library(xchainer_routines STATIC
    creation.cc
    indexing.cc
    manipulation.cc
    math.cc
)

if(${ENABLE_TEST})
  add_executable(xchainer_routines_test
      creation_test.cc
      indexing_test.cc
      manipulation_test.cc
      math_test.cc
  )
  target_link_libraries(xchainer_routines_test
      xchainer
      xchainer_routines
      xchainer_testing
      gtest)
  add_test(NAME xchainer_routines_test COMMAND xchainer_routines_test)
endif()
