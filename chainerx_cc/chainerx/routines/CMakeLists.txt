add_library(chainerx_routines STATIC
    activation.cc
    arithmetic.cc
    binary.cc
    connection.cc
    creation.cc
    evaluation.cc
    explog.cc
    hyperbolic.cc
    indexing.cc
    linalg.cc
    logic.cc
    loss.cc
    manipulation.cc
    misc.cc
    normalization.cc
    pooling.cc
    reduction.cc
    rounding.cc
    sorting.cc
    statistics.cc
    n_step_rnn.cc
    trigonometric.cc
    type_util.cc
)

install(FILES
    activation.h
    arithmetic.h
    binary.h
    connection.h
    creation.h
    evaluation.h
    explog.h
    hyperbolic.h
    indexing.h
    linalg.h
    logic.h
    loss.h
    manipulation.h
    misc.h
    normalization.h
    pooling.h
    reduction.h
    rounding.h
    routines_util.h
    sorting.h
    statistics.h
    n_step_rnn.h
    trigonometric.h
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
      chainerx_test_main
      # gtest is linked via chainerx_testing.
      )
  add_test(NAME chainerx_routines_test COMMAND chainerx_routines_test)
endif()
