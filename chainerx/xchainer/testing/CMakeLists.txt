install(FILES
    array.h
    array_check.h
    context_session.h
    device_session.h
    routines.h
    threading.h
    util.h
    DESTINATION include/xchainer/testing
    )

add_library(xchainer_testing STATIC
    array_check.cc
    util.cc
    routines.cc)

# TODO(niboshi): Remove gtest dependency from testing
target_link_libraries(xchainer_testing
    gtest)

if(${XCHAINER_BUILD_TEST})
    target_sources(xchainer_testing PRIVATE
        gtest_main.cc)

    add_executable(xchainer_testing_test
        array_test.cc
        routines_test.cc)

    target_link_libraries(xchainer_testing_test
      xchainer
      xchainer_testing)
    add_test(NAME xchainer_testing_test COMMAND xchainer_testing_test)
endif()
