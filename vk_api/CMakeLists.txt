project(vulkan_api VERSION 0.1.0 LANGUAGES CXX)

find_package(VulkanHeaders CONFIG)
find_package(volk CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)

add_library(
    vk_api
    STATIC
        vk_api.cpp
)

target_include_directories(
    vk_api
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vk_api
    PUBLIC
        Vulkan::Headers
        volk::volk # volk::volk_headers
        GPUOpen::VulkanMemoryAllocator
)