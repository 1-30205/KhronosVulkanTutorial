#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <limits>
#include <array>
#include <optional>
#include <set>
#include <map>
#include <unordered_map>

#include <vk_api.h>
#include <GLFW/glfw3.h>
#include <fmt/format.h>
#include <vulkan/vulkan_core.h>
#include "glm_api.h" // IWYU pragma: keep

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const std::string COMPUTE_SHADER_PATH = PROJECT_ROOT_DIR "/shaders/compute_shader_comp.spv";
const std::string VERTEX_SHADER_PATH = PROJECT_ROOT_DIR "/shaders/compute_shader_vert.spv";
const std::string FRAGMENT_SHADER_PATH = PROJECT_ROOT_DIR "/shaders/compute_shader_frag.spv";

const std::uint32_t MAX_FRAMES_IN_FLIGHT = 2; // 并行帧数量
const std::uint32_t EXPECTED_SWAPCHAIN_IMAGE_COUNT = 3; // 期望的交换链图像数量

const std::vector<const char*> g_validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> g_deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    // VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME // vk1.1, enabled by default
    // VK_KHR_MAINTENANCE_1_EXTENSION_NAME,                   // vk1.1, enabled by default
    // VK_KHR_MAINTENANCE_2_EXTENSION_NAME,                   // vk1.1, enabled by default
    // VK_KHR_MAINTENANCE_3_EXTENSION_NAME,                   // vk1.1, enabled by default
    // VK_KHR_BIND_MEMORY2_EXTENSION_NAME,                    // vk1.1, for vma
    // VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,           // vk1.2, for vma
    // VK_KHR_MAINTENANCE_4_EXTENSION_NAME,                   // vk1.3, for vma, must be explicitly enabled
    // VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,               // vk1.3
    // VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,               // vk1.3
    // VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME,          // partially promoted to vk1.3
};

#ifdef NDEBUG
const bool g_enableValidationLayers = false;
#else
const bool g_enableValidationLayers = true;
#endif

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pDebugMessenger) {
    if (vkCreateDebugUtilsMessengerEXT != nullptr) {
        return vkCreateDebugUtilsMessengerEXT(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(
    VkInstance instance,
    VkDebugUtilsMessengerEXT debugMessenger,
    const VkAllocationCallbacks* pAllocator) {
    if (vkDestroyDebugUtilsMessengerEXT != nullptr) {
        vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, pAllocator);
    }
}

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct UniformBufferObject
{
    float deltaTime = 1.0f;
};

struct Particle
{
    glm::vec2 position;
    glm::vec2 velocity;
    glm::vec4 color;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0; // 绑定点索引，vkCmdBindVertexBuffers调用时，会指定绑定的顶点缓冲区到哪个绑定点
        bindingDescription.stride = sizeof(Particle);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].binding = 0; // 绑定索引，与getBindingDescription中的binding一致
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Particle, position);

        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Particle, color);

        return attributeDescriptions;
    }
};

class ComputeShaderApplication
{
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        m_window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(m_window, this);
        glfwSetFramebufferSizeCallback(m_window, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<ComputeShaderApplication *>(glfwGetWindowUserPointer(window));
        app->m_framebufferResized = true;
    }

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createVMA();
        createSwapChain();
        createImageViews();
        createComputeDescriptorSetLayout();
        createGraphicsPipeline();
        createComputePipeline();
        createCommandPool();
        createShaderStorageBuffers();
        // createUniformBuffers();
        // createDescriptorPool();
        // createComputeDescriptorSets();
        // createCommandBuffers();
        // createComputeCommandBuffers();
        // createSyncObjects();
    }

    void mainLoop() {

    }

    void cleanup() {

    }

    void createInstance() {
        VkResult result = volkInitialize();
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to initialize volk, error code: " + std::to_string(result));
        }

        m_apiVersion = volkGetInstanceVersion();
        if (m_apiVersion < VK_API_VERSION_1_3) {
            throw std::runtime_error("Vulkan API version 1.3 or higher is required!");
        }
        fmt::println("Vulkan API version supported by this system: {}.{}.{}",
            VK_VERSION_MAJOR(m_apiVersion), VK_VERSION_MINOR(m_apiVersion), VK_VERSION_PATCH(m_apiVersion));

        if (g_enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_3;

        uint32_t glfwInstanceExtensionCount = 0;
        const char** glfwInstanceExtensions = glfwGetRequiredInstanceExtensions(&glfwInstanceExtensionCount);
        fmt::println("{} glfw required instance extensions;", glfwInstanceExtensionCount);
        for (uint32_t i = 0; i < glfwInstanceExtensionCount; ++i) {
            fmt::println("\t{}", glfwInstanceExtensions[i]);
        }

        std::vector<const char*> instanceExtensions(glfwInstanceExtensions, glfwInstanceExtensions + glfwInstanceExtensionCount);
        if (g_enableValidationLayers) {
            instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        uint32_t propertiesCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &propertiesCount, nullptr);
        std::vector<VkExtensionProperties> properties(propertiesCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &propertiesCount, properties.data());

        VkInstanceCreateFlags instanceCreateFlags = 0;
#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
        if (IsExtensionAvailable(properties, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
            instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            instanceCreateFlags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }
#endif

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
        createInfo.ppEnabledExtensionNames = instanceExtensions.data();
        createInfo.flags = instanceCreateFlags;

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (g_enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(g_validationLayers.size());
            createInfo.ppEnabledLayerNames = g_validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = &debugCreateInfo;
        } else {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }

        if (vkCreateInstance(&createInfo, nullptr, &m_instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }

        volkLoadInstanceOnly(m_instance);
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity =
            /* VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | */
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
        createInfo.pUserData = nullptr; // Optional
    }

    void setupDebugMessenger() {
        if (!g_enableValidationLayers) {
            return;
        }

        VkDebugUtilsMessengerCreateInfoEXT createInfo{};
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(m_instance, &createInfo, nullptr, &m_debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    void createSurface() {
        // 手动创建surface
        /*VkWin32SurfaceCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        createInfo.hwnd = glfwGetWin32Window(m_window);
        createInfo.hinstance = GetModuleHandle(nullptr);

        if (vkCreateWin32SurfaceKHR(m_instance, &createInfo, nullptr, &m_surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }*/

        // 调用GLFW接口创建surface
        if (glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                m_physicalDevice = device;
                break;
            }
        }
        if (m_physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    void createLogicalDevice() {
        auto queueFamilyIndex = findQueueFamilies(m_physicalDevice);
        if (queueFamilyIndex.has_value()) {
            fmt::println("queueFamilyIndex: {}", queueFamilyIndex.value());
        } else {
            throw std::runtime_error("failed to find a suitable queue family!");
        }
        m_queueFamilyIdx = queueFamilyIndex.value();

        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = m_queueFamilyIdx;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        VkPhysicalDeviceFeatures2 deviceFeatures2{};
        deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        deviceFeatures2.features.samplerAnisotropy = VK_TRUE;
        deviceFeatures2.features.sampleRateShading = VK_TRUE;

        // 启用VK_KHR_buffer_device_address扩展
        VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{};
        bufferDeviceAddressFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
        bufferDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE;

        // 启用VK_KHR_synchronization2/VK_KHR_maintenance4/VK_KHR_dynamic_rendering扩展
        VkPhysicalDeviceVulkan13Features vk13Features{};
        vk13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        vk13Features.synchronization2 = VK_TRUE;
        vk13Features.dynamicRendering = VK_TRUE;
        vk13Features.maintenance4 = VK_TRUE;

        // 启用VK_EXT_extended_dynamic_state扩展
        VkPhysicalDeviceExtendedDynamicStateFeaturesEXT extendedDynamicStateFeatures{};
        extendedDynamicStateFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT;
        extendedDynamicStateFeatures.extendedDynamicState = VK_TRUE;

        deviceFeatures2.pNext = &bufferDeviceAddressFeatures;
        bufferDeviceAddressFeatures.pNext = &vk13Features;
        vk13Features.pNext = &extendedDynamicStateFeatures;

        std::vector<const char*> deviceExtensions = g_deviceExtensions;
        uint32_t propertiesCount = 0;
        vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &propertiesCount, nullptr);
        std::vector<VkExtensionProperties> properties(propertiesCount);
        m_availableDeviceExtensions.resize(propertiesCount);
        vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &propertiesCount, m_availableDeviceExtensions.data());

#ifdef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
        if (IsExtensionAvailable(m_availableDeviceExtensions, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
            deviceExtensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
        }
#endif

#ifdef VK_EXT_MEMORY_BUDGET_EXTENSION_NAME
        if (IsExtensionAvailable(m_availableDeviceExtensions, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
            deviceExtensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        }
#endif

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pNext = &deviceFeatures2;
        createInfo.queueCreateInfoCount = 1;
        createInfo.pQueueCreateInfos = &queueCreateInfo;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();
        createInfo.pEnabledFeatures = nullptr; // 使用pNext链来启用功能，所以这里设为nullptr

        if (vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        volkLoadDeviceTable(&m_deviceTable, m_device);

        m_deviceTable.vkGetDeviceQueue(m_device, m_queueFamilyIdx, 0, &m_queue);
    }

    void createVMA() {
        VmaAllocatorCreateInfo allocatorCreateInfo{};
        allocatorCreateInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT |
                                    VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT;
#ifdef VK_EXT_MEMORY_BUDGET_EXTENSION_NAME
        if (IsExtensionAvailable(m_availableDeviceExtensions, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
            allocatorCreateInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
        }
#endif
        // VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT
        //   vulkan ext: VK_KHR_bind_memory2(vk1.1 core).
        //   works only if VmaAllocatorCreateInfo::vulkanApiVersion `== VK_API_VERSION_1_0` because the extension has
        //   been promoted to Vulkan 1.1.
        // VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT
        //   vulkan ext: VK_KHR_buffer_device_address(vk1.2 core).
        //   Found as available and enabled device feature `VkPhysicalDeviceBufferDeviceAddressFeatures::bufferDeviceAddress`
        // VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT
        //   vulkan ext: VK_KHR_maintenance4(vk1.3 core).
        //   Found as available and enabled device feature `VkPhysicalDeviceMaintenance4Features::maintenance4`
        allocatorCreateInfo.physicalDevice = m_physicalDevice;
        allocatorCreateInfo.device = m_device;
        allocatorCreateInfo.instance = m_instance;
        allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;

        VmaVulkanFunctions vulkanFunctions{};
        vmaImportVulkanFunctionsFromVolk(&allocatorCreateInfo, &vulkanFunctions);
        allocatorCreateInfo.pVulkanFunctions = &vulkanFunctions;

        if (vmaCreateAllocator(&allocatorCreateInfo, &m_allocator) != VK_SUCCESS) {
            throw std::runtime_error("failed to create vulkan memory allocator!");
        }
    }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(m_physicalDevice);
        printSwapChainSupportDetails(swapChainSupport);
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        fmt::println("surfaceFormat.format: {}", static_cast<int>(surfaceFormat.format));
        fmt::println("surfaceFormat.colorSpace: {}", static_cast<int>(surfaceFormat.colorSpace));
        fmt::println("presentMode: {}", static_cast<int>(presentMode));
        fmt::println("extent.width: {}", extent.width);
        fmt::println("extent.height: {}", extent.height);
        fmt::println("VkSurfaceCapabilitiesKHR.minImageCount: {}", swapChainSupport.capabilities.minImageCount);
        fmt::println("VkSurfaceCapabilitiesKHR.maxImageCount: {}", swapChainSupport.capabilities.maxImageCount);
        if (swapChainSupport.capabilities.minImageCount == 0) { // 0表示最大图片数量没有限制
            m_swapChainImageCount = std::max(swapChainSupport.capabilities.minImageCount, EXPECTED_SWAPCHAIN_IMAGE_COUNT);
        } else {
            m_swapChainImageCount = std::clamp(EXPECTED_SWAPCHAIN_IMAGE_COUNT,
                swapChainSupport.capabilities.minImageCount, swapChainSupport.capabilities.maxImageCount);
        }
        if (swapChainSupport.capabilities.maxImageCount > 0
            && m_swapChainImageCount > swapChainSupport.capabilities.maxImageCount) {
            m_swapChainImageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = m_surface;
        createInfo.minImageCount = m_swapChainImageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0; // Optional
        createInfo.pQueueFamilyIndices = nullptr; // Optional
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (m_deviceTable.vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        m_deviceTable.vkGetSwapchainImagesKHR(m_device, m_swapChain, &m_swapChainImageCount, nullptr);
        m_swapChainImages.resize(m_swapChainImageCount);
        m_deviceTable.vkGetSwapchainImagesKHR(m_device, m_swapChain, &m_swapChainImageCount, m_swapChainImages.data());
        fmt::println("swapChainImageCount: {}", m_swapChainImageCount);

        m_swapChainImageFormat = surfaceFormat.format;
        m_swapChainExtent = extent;
    }

    void createImageViews() {
        m_swapChainImageViews.resize(m_swapChainImages.size());

        for (size_t i = 0; i < m_swapChainImages.size(); ++i) {
            m_swapChainImageViews[i] = createImageView(m_swapChainImages[i], m_swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
        }
    }

    void createComputeDescriptorSetLayout() {
        std::array<VkDescriptorSetLayoutBinding, 3> bindings{};

        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[0].pImmutableSamplers = nullptr;

        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].pImmutableSamplers = nullptr;

        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[2].pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (m_deviceTable.vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_computeDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute descriptor set layout!");
        }
    }

    void createGraphicsPipeline() {
        auto vertShaderCode = readFile(VERTEX_SHADER_PATH);
        auto fragShaderCode = readFile(FRAGMENT_SHADER_PATH);
        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        auto bindingDescription = Particle::getBindingDescription();
        auto attributeDescriptions = Particle::getAttributeDescriptions();

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = nullptr; // 使用动态视口
        viewportState.scissorCount = 1;
        viewportState.pScissors = nullptr; // 使用动态剪刀矩形

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.sampleShadingEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_computeDescriptorSetLayout;

        if (m_deviceTable.vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkPipelineRenderingCreateInfo pipelineRenderingInfo{};
        pipelineRenderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        pipelineRenderingInfo.colorAttachmentCount = 1;
        pipelineRenderingInfo.pColorAttachmentFormats = &m_swapChainImageFormat;
        pipelineRenderingInfo.depthAttachmentFormat = VK_FORMAT_UNDEFINED;
        pipelineRenderingInfo.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.pNext = &pipelineRenderingInfo;
        pipelineInfo.stageCount = sizeof(shaderStages) / sizeof(shaderStages[0]);
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = nullptr; // 不使用深度/模板测试
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = m_pipelineLayout;
        pipelineInfo.renderPass = nullptr; // 使用动态渲染，所以这里设为nullptr
        pipelineInfo.subpass = 0; // pipelineInfo.renderPass中subpass的索引

        if (m_deviceTable.vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        m_deviceTable.vkDestroyShaderModule(m_device, fragShaderModule, nullptr);
        m_deviceTable.vkDestroyShaderModule(m_device, vertShaderModule, nullptr);
    }

    void createComputePipeline() {
        auto computeShaderCode = readFile(COMPUTE_SHADER_PATH);
        VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

        VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
        computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computeShaderStageInfo.module = computeShaderModule;
        computeShaderStageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = computeShaderStageInfo;
        pipelineInfo.layout = m_pipelineLayout;

        if (m_deviceTable.vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_computePipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline!");
        }
    }

    void createCommandPool() {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = m_queueFamilyIdx;

        if (m_deviceTable.vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    void createShaderStorageBuffers() {

    }

    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = mipLevels;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        VkImageView imageView;
        if (m_deviceTable.vkCreateImageView(m_device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture image view!");
        }

        return imageView;
    }

    static bool IsExtensionAvailable(const std::vector<VkExtensionProperties>& properties, const char* extensionName) {
        for (const auto& extension : properties) {
            if (strcmp(extension.extensionName, extensionName) == 0) {
                return true;
            }
        }
        return false;
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (m_deviceTable.vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB
                && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (VkPresentModeKHR presentMode : availablePresentModes) {
            fmt::println("presentMode: {}", static_cast<int>(presentMode));
        }
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        } else {
            int width, height;
            glfwGetFramebufferSize(m_window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice phyDevice) {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phyDevice, m_surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(phyDevice, m_surface, &formatCount, nullptr);
        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(phyDevice, m_surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(phyDevice, m_surface, &presentModeCount, nullptr);
        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(phyDevice, m_surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    void printSwapChainSupportDetails(const SwapChainSupportDetails& details) {
        fmt::println("SwapChainSupportDetails:");
        fmt::println("  capabilities:");
        fmt::println("    minImageCount: {}", details.capabilities.minImageCount);
        fmt::println("    maxImageCount: {}", details.capabilities.maxImageCount);
        fmt::println("    currentExtent: {}x{}", details.capabilities.currentExtent.width, details.capabilities.currentExtent.height);
        fmt::println("    minImageExtent: {}x{}", details.capabilities.minImageExtent.width, details.capabilities.minImageExtent.height);
        fmt::println("    maxImageExtent: {}x{}", details.capabilities.maxImageExtent.width, details.capabilities.maxImageExtent.height);
        fmt::println("    maxImageArrayLayers: {}", details.capabilities.maxImageArrayLayers);
        fmt::println("    supportedTransforms: 0x{:08x}", static_cast<int>(details.capabilities.supportedTransforms));
        fmt::println("    currentTransform: 0x{:08x}", static_cast<int>(details.capabilities.currentTransform));
        fmt::println("    supportedCompositeAlpha: 0x{:08x}", static_cast<int>(details.capabilities.supportedCompositeAlpha));
        fmt::println("    supportedUsageFlags: 0x{:08x}", static_cast<int>(details.capabilities.supportedUsageFlags));
        fmt::println("  formats:");
        for (const auto& format : details.formats) {
            fmt::println("    format: {}, colorSpace: {}", static_cast<int>(format.format), static_cast<int>(format.colorSpace));
        }
        fmt::println("  presentModes:");
        for (const auto& presentMode : details.presentModes) {
            fmt::println("    presentMode: {}", static_cast<int>(presentMode));
        }
    }

    bool isDeviceSuitable(VkPhysicalDevice device) {
        VkPhysicalDeviceProperties physicalDeviceProperties{};
        vkGetPhysicalDeviceProperties(device, &physicalDeviceProperties);

        std::optional<uint32_t> queueFamilyIdx = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        VkPhysicalDeviceFeatures supportedFeatures;
        vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

        return physicalDeviceProperties.apiVersion >= VK_API_VERSION_1_3 && queueFamilyIdx.has_value() && extensionsSupported
            && swapChainAdequate && (supportedFeatures.samplerAnisotropy == VK_TRUE);
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(g_deviceExtensions.begin(), g_deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    std::optional<uint32_t> findQueueFamilies(VkPhysicalDevice device) {
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        uint32_t i = 0;
        for (const auto& queueFamily : queueFamilies) {
            VkBool32 presentSupport = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, m_surface, &presentSupport);
            if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (presentSupport == VK_TRUE)) {
                return i;
            }
            ++i;
        }

        return std::nullopt;
    }

    bool checkValidationLayerSupport() const {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : g_validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    fmt::println("layer {} found", layerName);
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                fmt::println("layer {} not found", layerName);
                return false;
            }
        }

        return true;
    }

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = static_cast<size_t>(file.tellg());
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();
        return buffer;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData) {
        const char* level = "UNKNOWN";
        static const std::map<VkDebugUtilsMessageTypeFlagsEXT, const char*> levelMap = {
            { VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT, "VERBOSE"},
            { VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT,    "INFO"},
            { VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT, "WARNING"},
            { VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,   "ERROR"},
        };
        if (auto it = levelMap.find(messageSeverity);
            it != levelMap.end()) {
            level = it->second;
        }

        fmt::println("Validation Layer[{}]: {}", level, pCallbackData->pMessage);
        return VK_FALSE;
    }

    VkPhysicalDevice chooseSuitableDevice(const std::vector<VkPhysicalDevice>& devices) {
        if (devices.empty()) {
            return VK_NULL_HANDLE;
        }

        for (const auto& device : devices) {
            VkPhysicalDeviceProperties deviceProperties;
            vkGetPhysicalDeviceProperties(device, &deviceProperties);
            VkPhysicalDeviceFeatures deviceFeatures;
            vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

            fmt::println("device type: {}", static_cast<int>(deviceProperties.deviceType));
        }

        return devices.front();
    }

    void createBufferWithVMA(VkDeviceSize size,
                             VkBufferUsageFlags usage,
                             VmaAllocationCreateFlags allocFlags,
                             VkMemoryPropertyFlags requiredFlags,
                             VkMemoryPropertyFlags preferredFlags,
                             VkBuffer& buffer,
                             VmaAllocation& vmaAlloc,
                             VmaAllocationInfo* vmaAllocInfo = nullptr) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.flags = allocFlags;
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocInfo.requiredFlags = requiredFlags;
        allocInfo.preferredFlags = preferredFlags;

        if (vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo, &buffer, &vmaAlloc, vmaAllocInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer!");
        }
    }

    VkMemoryPropertyFlags getVmaAllocationMemoryProperties(VmaAllocation allocation) {
        VmaAllocationInfo allocInfo;
        vmaGetAllocationInfo(m_allocator, allocation, &allocInfo);
        VkMemoryPropertyFlags flags = 0;
        vmaGetAllocationMemoryProperties(m_allocator, allocation, &flags);
        return flags;
    }
private:
    GLFWwindow* m_window{ nullptr };

    uint32_t                     m_apiVersion = 0;

    VkInstance                   m_instance;
    VkDebugUtilsMessengerEXT     m_debugMessenger;
    VkSurfaceKHR                 m_surface;

    VkPhysicalDevice             m_physicalDevice { VK_NULL_HANDLE };
    VkDevice                     m_device;
    std::vector<VkExtensionProperties> m_availableDeviceExtensions;

    VolkDeviceTable              m_deviceTable;
    VmaAllocator                 m_allocator;

    uint32_t                     m_queueFamilyIdx;
    VkQueue                      m_queue;

    VkSwapchainKHR               m_swapChain;
    uint32_t                     m_swapChainImageCount { 0 };
    std::vector<VkImage>         m_swapChainImages;
    VkFormat                     m_swapChainImageFormat;
    VkExtent2D                   m_swapChainExtent;
    std::vector<VkImageView>     m_swapChainImageViews;

    VkPipelineLayout             m_pipelineLayout;
    VkPipeline                   m_graphicsPipeline;

    VkDescriptorSetLayout        m_computeDescriptorSetLayout;
    VkPipelineLayout             m_computePipelineLayout;
    VkPipeline                   m_computePipeline;

    std::vector<VkBuffer>        m_shaderStorageBuffers;
    std::vector<VmaAllocation>   m_shaderStorageBufferAllocations;

    std::vector<VkBuffer>        m_uniformBuffers;
    std::vector<VmaAllocation>   m_uniformBufferAllocations;
    std::vector<void *>          m_uniformBufferMapped;

    VkDescriptorPool             m_descriptorPool;
    std::vector<VkDescriptorSet> m_computeDescriptorSets;

    VkCommandPool                m_commandPool;
    std::vector<VkCommandBuffer> m_commandBuffers;
    std::vector<VkCommandBuffer> m_computeCommandBuffers;

    VkSemaphore                  m_semaphore;
    uint64_t                     timelineValue { 0 };
    std::vector<VkFence>         m_inflightFences;
    uint32_t                     m_frameIndex { 0 };

    double                       m_lastFrameTime { 0.0 };

    bool                         m_framebufferResized { false };

    double                       m_lastTime { 0.0 };

};

int main(int argc, const char* argv[]) {
    fmt::println("hello vulkan compute shader");

    try {
        ComputeShaderApplication app;
        app.run();
    } catch (const std::exception& e) {
        fmt::println("error: {}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
