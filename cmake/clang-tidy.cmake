# clang-tidy
option(ENABLE_CLANG_TIDY "Enable clang-tidy rules" ON)
if(ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY "clang-tidy")
    if(NOT CLANG_TIDY)
        message(SEND_ERROR "clang-tidy not found.")
    else()
        message(STATUS "Configuring clang-tidy")

        # Let cmake generate compile_commands.json
        set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

        file(GLOB_RECURSE SOURCE_FILES RELATIVE ${CMAKE_BINARY_DIR} *.cc)

        add_custom_target(
            clang-tidy
            COMMAND ${CLANG_TIDY} ${SOURCE_FILES}
            )
        add_custom_target(
            clang-tidy-fix
            COMMAND ${CLANG_TIDY} -fix ${SOURCE_FILES}
            )
    endif()
endif()
