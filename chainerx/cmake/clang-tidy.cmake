# clang-tidy
option(ENABLE_CLANG_TIDY "Enable clang-tidy rules" ON)
if(ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY "clang-tidy")
    if(NOT CLANG_TIDY)
        message(AUTHOR_WARNING "clang-tidy not found.")
    else()
        message(STATUS "Configuring clang-tidy")

        # Let cmake generate compile_commands.json
        set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

        add_custom_target(
            clang-tidy
            COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/run-clang-tidy.sh normal
            COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/run-clang-tidy.sh test
            )
    endif()
endif()
