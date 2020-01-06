install(FILES
    array.h
    array_check.h
    context_session.h
    device_session.h
    routines.h
    threading.h
    util.h
    DESTINATION include/chainerx/testing
    )

add_library(chainerx_testing STATIC
    array_check.cc
    util.cc
    routines.cc)

# TODO(niboshi): Remove gtest dependency from testing. When fixed, add gtest
# dependency to *_test libraries.
target_link_libraries(chainerx_testing
    gtest)

add_library(chainerx_test_main STATIC gtest_main.cc)

if(${CHAINERX_BUILD_TEST})
    add_executable(chainerx_testing_test
        array_test.cc
        routines_test.cc)

    target_link_libraries(chainerx_testing_test
      chainerx
      chainerx_test_main)
    add_test(NAME chainerx_testing_test COMMAND chainerx_testing_test)
endif()
