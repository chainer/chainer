install(FILES
    xchainer.h
    DESTINATION include/xchainer
    )

add_library(xchainer SHARED
    xchainer.cc
    )

install(TARGETS xchainer DESTINATION lib)
