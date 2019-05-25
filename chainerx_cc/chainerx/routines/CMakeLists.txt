add_library(chainerx_routines STATIC
    binary.cc
    connection.cc
    creation.cc
    hyperbolic.cc
    indexing.cc
    linalg.cc
    logic.cc
    manipulation.cc
    math.cc
    normalization.cc
    pooling.cc
    sorting.cc
    statistics.cc
<<<<<<< HEAD
    n_step_rnn.cc
=======
    trigonometric.cc
>>>>>>> 393b89783f89684a214ea1cca79fae7f98de0bcd
    type_util.cc
)

install(FILES
    binary.h
    connection.h
    creation.h
    hyperbolic.h
    indexing.h
    linalg.h
    logic.h
    manipulation.h
    math.h
    normalization.h
    pooling.h
    routines_util.h
    sorting.h
    statistics.h
<<<<<<< HEAD
    n_step_rnn.h
=======
    trigonometric.h
>>>>>>> 393b89783f89684a214ea1cca79fae7f98de0bcd
    type_util.h
    DESTINATION include/chainerx/routines
    )

if(${CHAINERX_BUILD_TEST})
  add_executable(chainerx_routines_test
      creation_test.cc
      statistics_test.cc
      type_util_test.cc
  )
  target_link_libraries(chainerx_routines_test
      chainerx
      chainerx_routines
      chainerx_testing
      gtest)
  add_test(NAME chainerx_routines_test COMMAND chainerx_routines_test)
endif()
