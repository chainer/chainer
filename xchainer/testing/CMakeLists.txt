install(FILES
    array.h
    array_check.h
    context_session.h
    device_session.h
    threading.h
    util.h
    DESTINATION include/xchainer/testing
    )

if(${ENABLE_TEST})
    add_library(xchainer_testing SHARED
        util.cc
        gtest_main.cc)

    target_link_libraries(xchainer_testing
      xchainer)

    add_executable(xchainer_testing_test
        array_test.cc)

    target_link_libraries(xchainer_testing_test
      xchainer
      xchainer_testing
      gtest)
    add_test(NAME xchainer_testing_test COMMAND xchainer_testing_test)
endif()
