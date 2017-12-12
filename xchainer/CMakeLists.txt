if(${BUILD_PYTHON})
    add_subdirectory(python)
endif()

install(FILES
    xchainer.h
    DESTINATION include/xchainer
    )

add_library(xchainer SHARED
    xchainer.cc
    )

install(TARGETS xchainer DESTINATION lib)

if(ENABLE_TEST)
    add_executable(xchainer_test
        xchainer_test.cc)
    target_link_libraries(xchainer_test
        xchainer
        gtest_main)
    add_test(NAME xchainer_test COMMAND xchainer_test)
endif()
