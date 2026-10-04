// Read-only access to the same cached Bender2019 map used by SPlisHSPlasH.
#include <Discregrid/cubic_lagrange_discrete_grid.hpp>
#include <limits>
#include <string>

#ifdef _WIN32
#define MAP_EXPORT __declspec(dllexport)
#else
#define MAP_EXPORT
#endif

static std::string last_error;
extern "C" {
MAP_EXPORT const char* map_error() { return last_error.c_str(); }
MAP_EXPORT void* map_open(const char* path) {
    try { return new Discregrid::CubicLagrangeDiscreteGrid(path); }
    catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}
MAP_EXPORT void map_close(void* handle) {
    delete static_cast<Discregrid::CubicLagrangeDiscreteGrid*>(handle);
}
MAP_EXPORT void map_query(void* handle, const double* positions, int n, double* result) {
    auto* map = static_cast<Discregrid::CubicLagrangeDiscreteGrid*>(handle);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        Eigen::Vector3d x(positions[3*i], positions[3*i+1], positions[3*i+2]);
        Eigen::Vector3d c0, gradient = Eigen::Vector3d::Zero();
        std::array<unsigned int, 32> cell;
        Eigen::Matrix<double, 32, 1> N;
        Eigen::Matrix<double, 32, 3> dN;
        double dist = std::numeric_limits<double>::max(), volume = 0;
        if (map->determineShapeFunctions(0, x, cell, c0, N, &dN)) {
            dist = map->interpolate(0, x, cell, c0, N, &gradient, &dN);
            if (dist > 0 && dist < 0.1)
                volume = map->interpolate(1, x, cell, c0, N);
            if (volume == std::numeric_limits<double>::max() || volume < 0)
                volume = 0;
        }
        result[5*i] = dist;
        result[5*i+1] = volume;
        for (int a = 0; a < 3; ++a) result[5*i+2+a] = gradient[a];
    }
}
}
