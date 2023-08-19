cmake_minimum_required(VERSION 3.26)

project(MPSModels LANGUAGES CXX Swift)

# some checks
if("${CMAKE_Swift_COMPILER_VERSION}" VERSION_LESS 5.8)
    message(FATAL_ERROR "Requires Swift 5.8 or greater. Have ${CMAKE_Swift_COMPILER_VERSION}")
endif()
if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND
    NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    message(FATAL_ERROR "Project requires building with Clang. Have ${CMAKE_CXX_COMPILER_ID}")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OSX_DEPLOYMENT_TARGET 13.0)

add_library(MPSModels SHARED
    Sources/VGG16.swift
    Sources/utils.swift
)
set_target_properties(MPSModels PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}
)

# compile MPSModels as static library for swift/cxx interop
add_library(MPSModelsCxx STATIC
    Sources/VGG16_cxx.swift
)
target_link_libraries(MPSModelsCxx PRIVATE MPSModels)
target_compile_options(MPSModelsCxx PRIVATE
  "SHELL:-Xcc -std=c++17"
  "SHELL:-emit-objc-header"
  "SHELL:-module-name MPSModelsCxx"
  "SHELL:-enable-experimental-cxx-interop"
)
set_target_properties(MPSModelsCxx PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}
)

# swift entry point
add_executable(main main.swift)
target_link_libraries(main PRIVATE MPSModels)
target_include_directories(main PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# cxx entry point
add_executable(main_cxx main.cpp)
# compile MPSModels first
add_dependencies(main_cxx MPSModelsCxx)
set_property(SOURCE main PROPERTY OBJECT_DEPENDS MPSModelsCxx)
target_link_libraries(main_cxx PRIVATE MPSModelsCxx)

# install settings
install(TARGETS main RUNTIME DESTINATION bin)
install(TARGETS main_cxx RUNTIME DESTINATION bin)
install(TARGETS MPSModels RUNTIME DESTINATION lib)
