add_library(xchainer_routines STATIC
    connection.cc
    creation.cc
    indexing.cc
    linalg.cc
    logic.cc
    manipulation.cc
    math.cc
    normalization.cc
    pooling.cc
    sorting.cc
    statistics.cc
)

install(FILES
    connection.h
    creation.h
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
    DESTINATION include/xchainer/routines
    )

if(${ENABLE_TEST})
  add_executable(xchainer_routines_test
      connection_test.cc
      creation_test.cc
      indexing_test.cc
      linalg_test.cc
      logic_test.cc
      manipulation_test.cc
      math_test.cc
      normalization_test.cc
      pooling_test.cc
      sorting_test.cc
      statistics_test.cc
  )
  target_link_libraries(xchainer_routines_test
      xchainer
      xchainer_routines
      xchainer_testing
      gtest)
  add_test(NAME xchainer_routines_test COMMAND xchainer_routines_test)
endif()
