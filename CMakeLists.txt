cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(khronos_vulkan_tutorial)

set(CMAKE_CXX_STANDARD 17)           # 将 C++ 标准设置为 C++ 17
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # C++ 17 是强制要求，不会衰退至低版本
set(CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

add_subdirectory(vk_api)

add_executable(
    main
        main.cpp
        stb_image_impl.cpp
        tiny_obj_loader_impl.cpp
)

target_compile_definitions(
    main PRIVATE
    PROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(
    main
    PRIVATE
        ${Stb_INCLUDE_DIR}
)

target_link_libraries(
    main
    PRIVATE
        glfw
        glm::glm # glm::glm-header-only
        tinyobjloader::tinyobjloader
        fmt::fmt
        vk_api
)