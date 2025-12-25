#ifndef VK_API_H
#define VK_API_H

// Disable Vulkan function prototypes to use volk for loading
#define VK_NO_PROTOTYPES
#define VK_ENABLE_BETA_EXTENSIONS // for macro VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
#include <vulkan/vulkan.h>

// Include volk for Vulkan function loading
#define VOLK_NO_DEVICE_PROTOTYPES // Disable global device-level prototypes, use volkLoadDeviceTable instead
#include <volk.h>

// Configure and include Vulkan Memory Allocator (VMA)
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vk_mem_alloc.h>

#endif // VK_API_H