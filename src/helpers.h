#pragma once
/*
This file contains useful function and definitions.
Do not ever edit this file - it will not be uploaded for evaluation.
If you want to modify any of the functions here (e.g. extend triangle test to quads),
copy the function "your_code_here.h" and give it a new name.
*/

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>
#include <glm/gtx/matrix_transform_2d.hpp>
DISABLE_WARNINGS_POP()

#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <array>

#ifdef _OPENMP
// Only if OpenMP is enabled.
#include <omp.h>
#endif

#include <framework/image.h>

/// <summary>
/// A value used to mark invalid pixels.
/// </summary>
static constexpr float INVALID_VALUE = -1e10f;


/// <summary>
/// Structure of an image with 3 planes.
/// </summary>
template<typename T>
struct ImagePlane3 {
    T X;
    T Y;
    T Z;

    T& operator[](size_t index)
    {
        switch (index) {
        case 0:
            return X;
        case 1:
            return Y;
        case 2:
            return Z;
        default:
            throw std::out_of_range("Index out of range");
        }
    }

    const T& operator[](size_t index) const
    {
        switch (index) {
        case 0:
            return X;
        case 1:
            return Y;
        case 2:
            return Z;
        default:
            throw std::out_of_range("Index out of range");
        }
    }
};



/// <summary> Scalar image </summary>
using ImageFloat = Image<float>;
/// <summary> 3-channel image stored in attrtibute order </summary>
using ImageVec3 = Image<glm::vec3>;
/// <summary> 3-channel image stored in plane order </summary>
using ImageFloatPlane3 = ImagePlane3<ImageFloat>;

/// <summary> Image in RGB colorspace </summary>
using ImageRGB = ImageVec3;
/// <summary> Image in XYZ colorspace </summary>
using ImageXYZ = ImageFloatPlane3;


/// <summary>
/// Structure for the warping grid mesh.
/// </summary>
struct Mesh {
    std::vector<glm::vec2> vertices;
    std::vector<glm::ivec3> triangles;

    void saveBinary(const std::filesystem::path& filepath) const {
        std::ofstream file(filepath, std::ios::binary);
        if (file.is_open()) {
            size_t vertices_size = vertices.size();
            file.write(reinterpret_cast<const char*>(&vertices_size), sizeof(vertices_size));
            file.write(reinterpret_cast<const char*>(vertices.data()), vertices_size * sizeof(glm::vec2));
            
            size_t triangles_size = triangles.size();
            file.write(reinterpret_cast<const char*>(&triangles_size), sizeof(triangles_size));
            file.write(reinterpret_cast<const char*>(triangles.data()), triangles_size * sizeof(glm::ivec3));
            
            file.close();
        }
    }

    void readBinary(const std::filesystem::path& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (file.is_open()) {
            size_t vertices_size;
            file.read(reinterpret_cast<char*>(&vertices_size), sizeof(vertices_size));
            vertices.resize(vertices_size);
            file.read(reinterpret_cast<char*>(vertices.data()), vertices_size * sizeof(glm::vec2));
            
            size_t triangles_size;
            file.read(reinterpret_cast<char*>(&triangles_size), sizeof(triangles_size));
            triangles.resize(triangles_size);
            file.read(reinterpret_cast<char*>(triangles.data()), triangles_size * sizeof(glm::ivec3));
            
            file.close();
        }
    }
};


/// <summary>
/// Scene parameters.
/// </summary>
struct SceneParams {
    float in_disp_min;
    float in_disp_max;
    float out_disp_min;
    float out_disp_max;
    float iod_mm;
    float px_size_mm;
    float screen_distance_mm;
    float near_plane_mm;
    float far_plane_mm;
    int bilateral_size;
    float bilateral_joint_sigma;
    float grid_viz_im_scale;
    float grid_viz_tri_scale;
    float warp_scale;
};

/// <summary>
/// Structure for the forward warp image.
/// RGB image and a binary mask (0 or 1 as float).
/// 1 -> valid pixel, 0 -> missing pixel (hole).
/// </summary>
struct ImageWithMask {
    ImageRGB image;
    ImageFloat mask;

    void saveBinary(const std::filesystem::path& filepath) {
        image.saveBinary(filepath.string() + "_image.bin");
        mask.saveBinary(filepath.string() + "_mask.bin");
    }

    void readBinary(const std::filesystem::path& filepath) {
        image.readBinary(filepath.string() + "_image.bin");
        mask.readBinary(filepath.string() + "_mask.bin");
    }
};

/// <summary>
/// Function wrappers for passing the bilinear sampling as an argument.
/// </summary>
typedef std::function<float(const ImageFloat&, const glm::vec2&)> BilinearSamplerFloat;
typedef std::function<glm::vec3(const ImageRGB&, const glm::vec2&)> BilinearSamplerRGB;

/// <summary>
/// Prints helpful information about OpenMP.
/// </summary>
void printOpenMPStatus() 
{
#ifdef _OPENMP
    // https://stackoverflow.com/questions/38281448/how-to-check-the-version-of-openmp-on-windows
    std::cout << "OpenMP version " << _OPENMP << " is ENABLED with " << omp_get_max_threads() << " threads." << std::endl;
#else
    std::cout << "OpenMP is DISABLED." << std::endl;
#endif
}

/// <summary>
/// Returns (un-normalized) pdf of a Gaussian distribution with given sigma (std dev) and mean (mu).
/// </summary>
/// <param name="x">Where to sample</param>
/// <param name="sigma">std dev</param>
/// <param name="mu">mean</param>
/// <returns>unnormalied pdf value</returns>
float gauss(const float x, const float sigma = 1.0f, const float mu = 0.0f)
{
    auto exponent = (x - mu) / sigma;
    return std::exp(-0.5f * exponent * exponent);
}

/// <summary>
/// Check on which side of of a line (a,b) a point pt lies.
/// Equivalent to clockwise/counter-clockwise test for a given triangle (pt,a,b).
/// </summary>
/// <param name="p1"></param>
/// <param name="p2"></param>
/// <param name="p3"></param>
/// <returns></returns>
float triangleSign(const glm::vec2& pt, const glm::vec2& a, const glm::vec2& b)
{
    return (pt.x - b.x) * (a.y - b.y) - (a.x - b.x) * (pt.y - b.y);
}

/// <summary>
/// Checks if a point is inside a triangle by testing each of the three edges.
/// A point is inside triangle (or any convex polygon) if it is on the same side of all edges.
/// The test passes also for points that lie exactly on the edge or vertex.
/// Adapted from: https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
/// </summary>
/// <param name="pt"></param>
/// <param name="v1"></param>
/// <param name="v2"></param>
/// <param name="v3"></param>
/// <returns>true if inside triangle (or on its edge or vertex)</returns>
inline bool isPointInsideTriangle(const glm::vec2& pt, const glm::vec2& v1, const glm::vec2& v2, const glm::vec2& v3)
{
    float d1, d2, d3;
    bool has_neg, has_pos;

    d1 = triangleSign(pt, v1, v2);
    d2 = triangleSign(pt, v2, v3);
    d3 = triangleSign(pt, v3, v1);

    has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(has_neg && has_pos);
}

/// <summary>
/// Returns barycentric coordinates of a point inside a triangle abc.
/// </summary>
/// <param name="p"></param>
/// <param name="a"></param>
/// <param name="b"></param>
/// <param name="c"></param>
/// <returns></returns>
auto barycentricCoordinates(const glm::vec2& p, const glm::vec2& a, const glm::vec2& b, const glm::vec2& c)
{
    glm::dvec2 v0 = b - a, v1 = c - a, v2 = p - a;
    auto d00 = glm::dot(v0, v0);
    auto d01 = glm::dot(v0, v1);
    auto d11 = glm::dot(v1, v1);
    auto d20 = glm::dot(v2, v0);
    auto d21 = glm::dot(v2, v1);
    auto denom = d00 * d11 - d01 * d01;

    auto bc = glm::vec3();
    if (glm::abs(denom) >= 1e-8) {
        bc.y = float((d11 * d20 - d01 * d21) / denom);
        bc.z = float((d00 * d21 - d01 * d20) / denom);
    }
    bc.x = 1.0f - bc.y - bc.z;
    return bc;
}

/// <summary>
/// Read the disparity from a PNG file.
/// Applies:
/// - offset from the disp.txt file.
/// - correct scaling factor based on the image size.
/// More about file format: https://vision.middlebury.edu/stereo/data/scenes2005/
/// </summary>
/// <param name="depth_filename"></param>
/// <returns></returns>
ImageFloat loadDisparity(std::filesystem::path depth_filename, const float scale_factor = 1.0f)
{
    auto disparity = ImageFloat(depth_filename);

    // Determine offset from disp.txt if available.
    float disp_offset = 0.0f;
    auto txt_filename = depth_filename.parent_path() / "dmin.txt";
    if (std::filesystem::is_regular_file(txt_filename)) {
        std::ifstream fs(txt_filename);
        fs >> disp_offset;
    }
    disp_offset = 0;

    // Rescale to pixels.
    #pragma omp parallel for
    for (auto i = 0; i < disparity.data.size(); i++) {
        // Is it valid?
        if (disparity.data[i] == 0.0f) {
            disparity.data[i] = INVALID_VALUE;
        } else {
            // Rescale back to 8bit, remove offset, divide by scaling factor.
            disparity.data[i] = (disparity.data[i] * 255.0f + disp_offset) * scale_factor;
        }
    }

    return disparity;
}

/// <summary>
/// Converts RGB colors to HSV color space.
/// </summary>
/// <param name="rgb">RGB image</param>
/// <returns>HSV image</returns>
glm::vec3 rgbToHsv(const glm::vec3& rgb)
{
    auto hsv = glm::vec3();
    hsv.z = std::max(rgb.r, std::max(rgb.g, rgb.b));
    float m = std::min(rgb.r, std::min(rgb.g, rgb.b));
    float c = hsv.z - m;
    if (c != 0) {
        hsv.y = c / hsv.z;
        auto delta = (hsv.z - rgb) / c;
        delta = delta - glm::vec3(delta.b, delta.r, delta.g);
        delta.r += 2;
        delta.g += 4;
        if (rgb.r >= hsv.z) {
            hsv.x = delta.b;
        } else if (rgb.g >= hsv.z) {
            hsv.x = delta.r;
        } else {
            hsv.x = delta.g;
        }
        hsv.x = (hsv.x / 6);
        hsv.x -= std::floor(hsv.x);
    }
    return hsv;
}

/// <summary>
/// Converts HSV color to RGB color space.
/// </summary>
/// <param name="hsv">HSV image</param>
/// <returns>RGB image</returns>
glm::vec3 hsvToRgb(const glm::vec3& hsv)
{
    auto hue = hsv.x;
    auto r = abs(hue * 6 - 3) - 1;
    auto g = 2 - abs(hue * 6 - 2);
    auto b = 2 - abs(hue * 6 - 4);
    auto rgb_hue = glm::clamp(glm::vec3(r, g, b), 0.0f, 1.0f);

    // Apply saturation and value.
    return ((rgb_hue - 1.0f) * hsv.y + 1.0f) * hsv.z;
}


/// <summary>
/// Color maps the disparity for nicer viewing.
/// Makes regions with zero disparity white,
/// regions with negative/crossed disparity (=near) blue,
/// and regions with positive disparity (=far) yellow.
/// </summary>
/// <param name="disparity"></param>
/// <param name="max_val"></param>
/// <returns></returns>
ImageRGB disparityToColor(const ImageFloat& disparity, const float min_val = 0.0f, const float max_val = 1.0f) {
    const glm::vec3 color_neg = glm::vec3(26, 56, 89) / 255.0f;
    const glm::vec3 color_zero = glm::vec3(255, 255, 255) / 255.0f;
    const glm::vec3 color_pos = glm::vec3(247, 190, 29) / 255.0f;

    auto res = ImageRGB(disparity.width, disparity.height);
    
    #pragma omp parallel for
    for (auto i = 0; i < disparity.data.size(); i++) {
        auto val = glm::clamp((disparity.data[i] - min_val) / (max_val - min_val), 0.0f, 1.0f);
        if (val < 0.0) {
            res.data[i] = glm::mix(color_zero, color_neg, -val);
        } else {
            res.data[i] = glm::mix(color_zero, color_pos, val);        
        }
    }

    return res;
}

/// <summary>
/// Bresenham's line drawing algorithm.
/// https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
/// </summary>
/// <param name="im"></param>
/// <param name="x0"></param>
/// <param name="y0"></param>
/// <param name="x1"></param>
/// <param name="y1"></param>
/// <param name="color"></param>
void plotLine(ImageRGB& im,
    const int x0, const int y0, const int x1, const int y1,
    const glm::vec3 color = glm::vec3(0)) {

    auto dx = abs(x1 - x0);
    auto sx = x0 < x1 ? 1 : -1;
    auto dy = -abs(y1 - y0);
    auto sy = y0 < y1 ? 1 : -1;
    auto error = dx + dy;

    auto x = x0;
    auto y = y0;
    
    while (true) {
        if (x >= 0 && x < im.width && y >= 0 && y < im.height) {
            im.data[y * im.width + x] = color;
        }
        if (x == x1 && y == y1) {
            break;
        }
        auto e2 = 2 * error;
        if (e2 >= dy) {
            if (x == x1) {
                break;
            }
            error = error + dy;
            x = x + sx;
        }
        if (e2 <= dx) {
            if (y == y1) {
                break;
            }
            error = error + dx;            
            y = y + sy;
        }
    }
}

/// <summary>
/// Plots the mesh grid.
/// Only approximate - does not handle the
/// coordinates entirely correctly.
/// </summary>
/// <param name="mesh"></param>
/// <param name="w"></param>
/// <param name="h"></param>
/// <returns>image of the grid</returns>
ImageRGB plotGridMesh(const Mesh& mesh, const glm::ivec2& resolution, const float scale_factor = 1.0f)
{
    const auto colors = std::array {
        glm::vec3(1, 0, 0),
        glm::vec3(0, 1, 0),
        glm::vec3(0, 0, 1),
        glm::vec3(1, 1, 0),
        glm::vec3(1, 0, 1),
        glm::vec3(0, 1, 1),
    };

    auto res = ImageRGB(resolution.x, resolution.y);
    std::fill(res.data.begin(), res.data.end(), glm::vec3(1.0f));
    auto scaler = scale_factor;

    auto counter = 0;
    for (auto i = 0; i < mesh.triangles.size(); i++) {
        auto tri = mesh.triangles[i];
        for (auto j = 0; j < 3; j++) {
            auto a = mesh.vertices[tri[j]];
            auto b = mesh.vertices[tri[(j + 1) % 3]];
            auto ai = glm::ivec2(glm::round(a * scaler));
            auto bi = glm::ivec2(glm::round(b * scaler));

            auto color = colors[counter % colors.size()] * 0.5f;
            plotLine(res, ai.x, ai.y, bi.x, bi.y, color);
            counter++;
        }
    }

    return res;
}

/// <summary>
/// Switches from Attribute layout to Plane layout.
/// </summary>
/// <param name="image">image in attribute order</param>
/// <returns>image in plane order</returns>
ImageFloatPlane3 imageVec3ToPlane3(const ImageVec3& image)
{
    auto result = ImageFloatPlane3({ image.width, image.height }, { image.width, image.height }, { image.width, image.height });
#pragma omp parallel for
    for (int i = 0; i < image.data.size(); i++) {
        for (auto j = 0; j < 3; j++) {
            result[j].data[i] = image.data[i][j];
        }
    }
    return result;
}

/// <summary>
/// Switches from Attribute layout to Plane layout.
/// </summary>
/// <param name="image">image in attribute order</param>
/// <returns>image in plane order</returns>
ImageVec3 imagePlane3ToVec3(const ImageFloatPlane3& image)
{
    auto result = ImageVec3(image.X.width, image.X.height);
#pragma omp parallel for
    for (int i = 0; i < image.X.data.size(); i++) {
        for (auto j = 0; j < 3; j++) {
            result.data[i][j] = image[j].data[i];
        }
    }
    return result;
}


