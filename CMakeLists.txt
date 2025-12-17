cmake_minimum_required(VERSION 3.20)
project(pacman)
set(CMAKE_C_STANDARD 11)

# Generate compile_commands.json for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(PACMAN_ENABLE_API "Enable HTTP API for AI training" OFF)

# POSIX thread handling (covers Linux, GNU/Hurd, BSD)
if (UNIX AND NOT APPLE)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
endif()

#=== LIBRARY: sokol
# add headers to the the file list because they are useful to have in IDEs
set(SOKOL_HEADERS
    external/sokol/sokol_gfx.h
    external/sokol/sokol_app.h
    external/sokol/sokol_audio.h
    external/sokol/sokol_glue.h)
add_library(sokol STATIC external/sokol/sokol.c ${SOKOL_HEADERS})
if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    # compile sokol.c as Objective-C
    target_compile_options(sokol PRIVATE -x objective-c)
    target_link_libraries(sokol
        "-framework QuartzCore"
        "-framework Cocoa"
        "-framework MetalKit"
        "-framework Metal"
        "-framework OpenGL"
        "-framework AudioToolbox")
else()
    if (UNIX AND NOT APPLE)
        target_link_libraries(sokol INTERFACE X11 Xi Xcursor GL asound dl m)
        target_link_libraries(sokol PUBLIC Threads::Threads)
    endif()
endif()
target_include_directories(sokol INTERFACE external/sokol)

#=== LIBRARY: mongoose (optional, for HTTP API)
if(PACMAN_ENABLE_API)
    # Create mongoose library from submodule
    add_library(mongoose_lib STATIC external/mongoose/mongoose.c)
    target_include_directories(mongoose_lib PUBLIC external/mongoose)
    target_compile_definitions(mongoose_lib PUBLIC MG_ENABLE_LINES=1)
endif()

#=== EXECUTABLE: pacman
if(CMAKE_SYSTEM_NAME STREQUAL Windows)
    add_executable(pacman WIN32 src/pacman.c)
else()
    add_executable(pacman src/pacman.c)
endif()
target_link_libraries(pacman sokol)

# Link mongoose if API is enabled
if(PACMAN_ENABLE_API)
    target_link_libraries(pacman mongoose_lib)
    target_compile_definitions(pacman PRIVATE PACMAN_API_ENABLED=1)
endif()
if (CMAKE_SYSTEM_NAME STREQUAL Emscripten)
    set(CMAKE_EXECUTABLE_SUFFIX ".html")
    target_link_options(pacman PUBLIC --shell-file ../sokol/shell.html)
    target_link_options(pacman PUBLIC -sUSE_WEBGL2=1 -sNO_FILESYSTEM=1 -sASSERTIONS=0 -sMALLOC=emmalloc --closure=1)
endif()
if (MSVC)
    target_compile_options(pacman PUBLIC /W3)
else()
    target_compile_options(pacman PUBLIC -Wall -Wextra -Wsign-compare)
endif()

# explicitly strip dead code (Darwin only)
if (CMAKE_SYSTEM_NAME STREQUAL "Darwin"
    AND CMAKE_C_COMPILER_ID MATCHES "Clang"
    AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    target_link_options(pacman PRIVATE LINKER:-dead_strip)
endif()
