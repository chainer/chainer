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
