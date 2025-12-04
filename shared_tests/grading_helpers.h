#pragma once
#include "your_code_here.h"
// Suppress warnings in third-party code.
//#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_transform_2d.hpp>
#include <glm/mat3x3.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
DISABLE_WARNINGS_POP()
#include <random>
#include <tuple>

#include <framework/image.h>

/// <summary>
/// Structure to hold all test variables returned by initTestVariables
/// </summary>
struct TestVariables {
    ImageFloat disparity_filtered;
    ImageFloat linear_depth;
    Mesh src_grid;
    Mesh dst_grid;
    ImageFloat target_disparity;

    void saveBinary(const std::filesystem::path& basePath) {
        disparity_filtered.saveBinary(basePath.string() + "_disparity_filtered.bin");
        linear_depth.saveBinary(basePath.string() + "_linear_depth.bin");
        src_grid.saveBinary(basePath.string() + "_src_grid.bin");
        dst_grid.saveBinary(basePath.string() + "_dst_grid.bin");
        target_disparity.saveBinary(basePath.string() + "_target_disparity.bin");
    }

    void readBinary(const std::filesystem::path& basePath) {
        disparity_filtered.readBinary(basePath.string() + "_disparity_filtered.bin");
        linear_depth.readBinary(basePath.string() + "_linear_depth.bin");
        src_grid.readBinary(basePath.string() + "_src_grid.bin");
        dst_grid.readBinary(basePath.string() + "_dst_grid.bin");
        target_disparity.readBinary(basePath.string() + "_target_disparity.bin");
    }
};

using ImageFloat = Image<float>;
using ImageRGB = Image<glm::vec3>;

#define APPROX_FLOAT(x) Catch::Approx(x).margin(1e-2f)
#define APPROX_FLOAT_5e(x) Catch::Approx(x).margin(5e-2f)
#define APPROX_FLOAT_EPS(x,eps) Catch::Approx(x).margin(eps)

// ApproxZero by default uses an epsilon of 0.0f and for some reason it makes one of the tests fail ( 0.0f == Approx(0.0) fails... ).
// https://github.com/catchorg/Catch2/issues/1444
//
// https://stackoverflow.com/questions/56466022/what-is-the-canonical-way-to-check-for-approximate-zeros-in-catch2
#define ApproxZero Catch::Approx(0.0f).margin(1e-2f)

static constexpr size_t testsPerSection = 25;


#define CHECK_GLM(x, y) CHECK(glm::length((x) - (y)) == APPROX_FLOAT(0))
#define REQUIRE_GLM(x, y) REQUIRE(glm::length((x) - (y)) == APPROX_FLOAT(0))

/// <summary>
/// https://stackoverflow.com/questions/48955718/c-how-to-calculate-rmse-between-2-vectors
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="a"></param>
/// <param name="b"></param>
/// <returns></returns>
template <typename T>
auto calcVectorRMSE(const std::vector<T>& a, const std::vector<T>& b)
{
    auto squareError = [](T a, T b) {
        auto e = a - b;
        return glm::dot(e, e);
    };

    auto sum = std::transform_reduce(a.begin(), a.end(), b.begin(), 0.0f, std::plus<>(), squareError);
    auto rmse = std::sqrt(sum / a.size());
    return rmse;
}

template <typename T>
auto calcImageRMSE(const T& a, const T& b)
{
    return calcVectorRMSE(a.data, b.data);
}

template <typename T>
auto calcImageRMSEWithMask(const T& ref, const T& b, const ImageFloat& mask)
{
    auto sum = 0.0f;

    for (auto y = 1; y < ref.height - 1; y++) {
        for (auto x = 1; x < ref.width - 1; x++) {
            auto center_offset = y * ref.width + x;

            auto test_signal = b.data[center_offset];
            auto ref_signal = ref.data[center_offset];

            if (mask.data[center_offset] > 0.5f) {
                auto error = ref_signal - test_signal;
                sum += glm::dot(error, error);
            } else {
                // We allow the students to use bilateral filtering to fill the holes this year(2025).
                // In this case, we don't check the error as long as the hole is filled(test_signal is not zero).
                // An extreme case is when a large hole can not be filled (ref_signal is zero).
                if (glm::dot(test_signal, test_signal) < 0.01f) {
                    auto error = ref_signal - test_signal;
                    sum += glm::dot(error, error);
                }
            }
        }
    }
    auto rmse = std::sqrt(sum / (ref.width * ref.height));
    return rmse;
}

void saveVec2ToFile(const glm::vec2& vec, const std::filesystem::path& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(&vec), sizeof(vec));
        file.close();
    }
}

void saveVec3ToFile(const glm::vec3& vec, const std::filesystem::path& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(&vec), sizeof(vec));
        file.close();
    }
}

template<typename T>
void saveVecToFile(const T& vec, const std::filesystem::path& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(&vec), sizeof(vec));
        file.close();
    }
}

glm::vec2 loadVec2FromFile(const std::filesystem::path& filepath) {
    glm::vec2 vec;
    std::ifstream file(filepath, std::ios::binary);
    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(&vec), sizeof(vec));
        file.close();
    }
    return vec;
}

glm::vec3 loadVec3FromFile(const std::filesystem::path& filepath) {
    glm::vec3 vec;
    std::ifstream file(filepath, std::ios::binary);
    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(&vec), sizeof(vec));
        file.close();
    }
    return vec;
}

template<typename T>
T loadVecFromFile(const std::filesystem::path& filepath) {
    T vec;
    std::ifstream file(filepath, std::ios::binary);
    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(&vec), sizeof(vec));
        file.close();
    }
    return vec;
}

float saveFloatToFile(const float value, const std::filesystem::path& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        file.close();
    }
    return value;
}

float loadFloatFromFile(const std::filesystem::path& filepath) {
    float value;
    std::ifstream file(filepath, std::ios::binary);
    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(&value), sizeof(value));
        file.close();
    }   
    return value;
}
