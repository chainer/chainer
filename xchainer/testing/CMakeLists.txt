install(FILES
    array.h
    context_session.h
    device_session.h
    DESTINATION include/xchainer/testing
    )

if(${ENABLE_TEST})
    add_library(xchainer_testing SHARED
        gtest_main.cc)
endif()
