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
    reduction.cc
    sorting.cc
    statistics.cc
    trigonometric.cc
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
    reduction.h
    routines_util.h
    sorting.h
    statistics.h
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
      chainerx_routines
      chainerx_testing
      gtest)
  add_test(NAME chainerx_routines_test COMMAND chainerx_routines_test)
endif()
