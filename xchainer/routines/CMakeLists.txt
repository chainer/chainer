add_library(xchainer_routines STATIC
    creation.cc
    indexing.cc
    logic.cc
    manipulation.cc
    math.cc
    util.cc
)

install(FILES
    creation.h
    indexing.h
    logic.h
    manipulation.h
    math.h
    util.h
    DESTINATION include/xchainer/routines
    )

if(${ENABLE_TEST})
  add_executable(xchainer_routines_test
      creation_test.cc
      indexing_test.cc
      logic_test.cc
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
