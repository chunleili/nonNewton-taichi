cmake_minimum_required(VERSION 3.22)
project(cpp_map_bridge LANGUAGES CXX)
set(CPP_REPO "" CACHE PATH "Built nonNewtonCode checkout")
find_package(OpenMP REQUIRED)
add_library(cpp_map_bridge SHARED map_bridge.cpp)
target_compile_features(cpp_map_bridge PRIVATE cxx_std_17)
target_include_directories(cpp_map_bridge PRIVATE
    "${CPP_REPO}/extern/eigen"
    "${CPP_REPO}/build/extern/install/Discregrid/include")
target_link_libraries(cpp_map_bridge PRIVATE OpenMP::OpenMP_CXX
    "${CPP_REPO}/build/extern/install/Discregrid/lib/Discregrid.lib")
