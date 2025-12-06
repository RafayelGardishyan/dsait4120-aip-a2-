#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <tuple>
#include <vector>

#include "helpers.h"

/*
 * Utility functions.
 */

template <typename T>
int getIndex(const Image<T>& image, int px, int py)
{
    int x = std::max(0, std::min(image.width - 1, px));
    int y = std::max(0, std::min(image.height - 1, py));

    return x + (y * image.width);
}

/// <summary>
/// Bilinearly sample ImageFloat or ImageRGB using image coordinates [x,y].
/// </summary>
/// <typeparam name="T">template type, can be ImageFloat or ImageRGB</typeparam>
/// <param name="image">input image</param>
/// <param name="pos">x,y position in px units where first pixel center is [0.5,0.5]</param>
/// <returns>interpolated pixel value (float or glm::vec3)</returns>
template <typename T>
inline T sampleBilinear(const Image<T>& image, const glm::vec2& pos_px)
{
    //
    // Write a code that bilinearly interpolates values from a generic image (can contain either float or glm::vec3).
    //
    // The pos_px input represents the (x,y) pixel coordinates of the sampled point where:
    //   [0, 0] = The left top corner of the left top (=first) pixel.
    //   [width, height] = The right bottom corner of the right bottom (=last) pixel.
    //   [0, height] = The left bottom corner of the left bottom pixel.
    //
    //
    // Note: The method is templated by parameter "T". This will be either float or glm::vec3 depending on whether the method
    // is called with ImageFloat or ImageRGB. Use either "T" or "auto" to define your variables and use glm::functions to handle both types.
    // Example:
    //    auto value = image.data[0] * 3; // both float and glm:vec3 support baisc operators
    //    T rounded_value = glm::round(image.data[0]); // glm::round will handle both glm::vec3 and float.
    // Use glm API for further reference: https://glm.g-truc.net/0.9.9/api/a00241.html
    //

    glm::vec2 cpx = pos_px + glm::vec2(-0.5);

    int x1 = std::floor(cpx.x);
    int x2 = x1 + 1;
    int y1 = std::floor(cpx.y);
    int y2 = y1 + 1;

    float dx = cpx.x - x1;
    float dy = cpx.y - y1;

    T ipx1 = (1 - dx) * image.data[getIndex(image, x1, y1)] + dx * image.data[getIndex(image, x2, y1)];
    T ipx2 = (1 - dx) * image.data[getIndex(image, x1, y2)] + dx * image.data[getIndex(image, x2, y2)];

    return (1 - dy) * ipx1 + dy * ipx2;
}

/*
  Core functions.
*/

/// <summary>
/// Applies a joint bilateral filter to the given disparity image guided by an RGB guide image. Also known as cross-bilateral filter.
/// Ignores pixels that are marked as invalid.
/// </summary>
/// <param name="disparity">The image to be filtered.</param>
/// <param name="guide">The image guide used for calculating the tonal distances between pixel values.</param>
/// <param name="size">The kernel size, which is always odd.</param>
/// <param name="guide_sigma">Sigma value of the gaussian guide kernel.</param>
/// <returns>ImageFloat, the filtered disparity.</returns>
ImageFloat jointBilateralFilter(const ImageFloat& disparity, const ImageRGB& guide, const int size, const float guide_sigma)
{
    // The filter size is always odd.
    assert(size % 2 == 1);

    // We assume both images have matching dimensions.
    assert(disparity.width == guide.width && disparity.height == guide.height);

    // Rule of thumb for gaussian's std dev.
    const float sigma = (size - 1) / 2 / 3.2f;

    // Empty output image.
    auto result = ImageFloat(disparity);

    int width = disparity.width;
    int height = disparity.height;

    //
    // Implement a joint/cross bilateral filter of the disparity image guided by the guide RGB image.
    // Ignore all contributing pixels where disparity == INVALID_VALUE.
    //

    //
    // Notes:
    //   * If a pixel has no neighbor (all were skipped), assign INVALID_VALUE to the output.
    //   * Parallelize the code using OpenMP directives.

    // auto example = gauss(0.5f, 1.2f); // This is just an example of computing Normal pdf for x=0.5 and std.dev=1.2.

    int half_size = (size - 1) / 2;

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {

            auto p = guide.data[getIndex(guide, x, y)];

            float res = 0.0;
            float w = 0.0;

            for (int i = -half_size; i <= half_size; i++) {
                for (int j = -half_size; j <= half_size; j++) {
                    int nx = x + i;
                    int ny = y + j;

                    if (nx < 0 || nx >= disparity.width || ny < 0 || ny >= disparity.height)
                        continue;

                    auto vx = disparity.data[getIndex(disparity, nx, ny)];
                    if (vx == INVALID_VALUE)
                        continue;

                    auto pn = guide.data[getIndex(guide, nx, ny)];
                    float norm = glm::length(pn - p);
                    float w_i = (gauss(nx - x, sigma) * gauss(ny - y, sigma)) * gauss(norm, guide_sigma);

                    res += vx * w_i;
                    w += w_i;
                }
            }

            if (w > 0.0f)
                result.data[getIndex(result, x, y)] = res / w;
            else
                result.data[getIndex(result, x, y)] = INVALID_VALUE;
        }
    }

    // Return filtered disparity.
    return result;
}

/// <summary>
/// In-place normalizes and an ImageFloat image to be between 0 and 1.
/// All values marked as invalid will stay marked as invalid.
/// </summary>
/// <param name="scalar_image"></param>
/// <returns></returns>
void normalizeValidValues(ImageFloat& scalar_image)
{
    //
    // Find minimum and maximum among the VALID image values.
    // Linearly rescale the VALID image values to the [0,1] range (in-place).
    // The INVALID values remain INVALID (they are ignored).
    //
    // Note #1: Pixel is INVALID as long as value == INVALID_VALUE.
    // Note #2: This modified the input image in-place => no "return".
    //

    float min = INFINITY;
    float max = -INFINITY;

#pragma omp parallel for reduction(min : min) reduction(max : max)
    for (int i = 0; i < scalar_image.data.size(); i++) {
        float val = scalar_image.data[i];
        if (val == INVALID_VALUE)
            continue;
        min = std::min(min, val);
        max = std::max(max, val);
    }

    float maxmin = max - min;

#pragma omp parallel for
    for (int i = 0; i < scalar_image.data.size(); i++) {
        if (scalar_image.data[i] == INVALID_VALUE)
            continue;
        scalar_image.data[i] = (scalar_image.data[i] - min) / maxmin;
    }
}

/// <summary>
/// Converts a disparity image to a normalized depth image.
/// Ignores invalid disparity values.
/// </summary>
/// <param name="disparity">disparity in arbitrary units</param>
/// <returns>linear depth scaled from 0 to 1</returns>
ImageFloat disparityToNormalizedDepth(const ImageFloat& disparity)
{
    auto depth = ImageFloat(disparity.width, disparity.height);

    //
    // If disparity of a pixel is invalid, set its depth also invalid (INVALID_VALUE).
    // We guarantee that all valid disparities > 0.
    //

#pragma omp parallel for
    for (int i = 0; i < disparity.data.size(); i++) {
        float val = disparity.data[i];
        depth.data[i] = val == INVALID_VALUE ? INVALID_VALUE : 1.0f / val;
    }

    // Rescales valid depth values to [0,1] range.
    normalizeValidValues(depth);

    return depth;
}

/// <summary>
/// Convert linear normalized depth to target pixel disparity.
/// Invalid pixels
/// </summary>
/// <param name="depth">Normalized depth image (values in [0,1])</param>
/// <param name="iod_mm">Inter-ocular distance in mm.</param>
/// <param name="px_size_mm">Pixel size in mm.</param>
/// <param name="screen_distance_mm">Screen distance from eyes in mm.</param>
/// <param name="near_plane_mm">Near plane distance from eyes in mm.</param>
/// <param name="far_plane_mm">Far plane distance from eyes in mm.</param>
/// <returns>screen disparity in pixels</returns>
ImageFloat normalizedDepthToDisparity(
    const ImageFloat& depth, const float iod_mm,
    const float px_size_mm, const float screen_distance_mm,
    const float near_plane_mm, const float far_plane_mm)
{
    auto px_disparity = ImageFloat(depth.width, depth.height);

    //
    // Note:
    //    * All distances are measured orthogonal on the screen and such that pixel size is assumed constant across the screen (ignores the eccentricity).
    //    * Invalid pixels (depth==INVALID_VALUE) are to be marked invalid on the output as well.
    //

#pragma omp parallel for
    for (int i = 0; i < depth.data.size(); i++) {
        if (depth.data[i] == INVALID_VALUE) {
            px_disparity.data[i] = INVALID_VALUE;
            continue;
        }

        auto depth_abs = near_plane_mm + (depth.data[i] * (far_plane_mm - near_plane_mm));

        px_disparity.data[i] = (iod_mm / px_size_mm) * ((depth_abs - screen_distance_mm) / depth_abs);
    }

    return px_disparity; // returns disparity measured in pixels
}

/// <summary>
/// Creates a warping grid for an image of specified height and weight.
/// It produces vertex buffer which stores 2D positions of pixel corners,
/// and index buffer which defines triangles by triplets of indices into
/// the vertex buffer (the three vertices form a triangle).
///
/// </summary>
/// <param name="width">Image width.</param>
/// <param name="height">Image height.</param>
/// <returns>Mesh, containing a vertex buffer and triangle index buffer.</returns>
Mesh createWarpingGrid(const int width, const int height)
{

    // Build vertex buffer.
    auto num_vertices = (width + 1) * (height + 1);
    auto vertices = std::vector<glm::vec2>(num_vertices);

    //
    //    YOUR CODE GOES HERE
    //

    // Build index buffer.
    auto num_pixels = width * height;
    auto num_triangles = num_pixels * 2;
    auto triangles = std::vector<glm::ivec3>(num_triangles);

    //
    //    YOUR CODE GOES HERE
    //
    // Combine the vertex and index buffers into a mesh.
    return Mesh { std::move(vertices), std::move(triangles) };
}

/// <summary>
/// Warps a grid based on the given disparity and scaling_factor.
/// </summary>
/// <param name="grid">The original mesh.</param>
/// <param name="disparity">Disparity for each PIXEL.</param>
/// <param name="scaling_factor">Global scaling factor for the disparity.</param>
/// <returns>Mesh, the warped grid.</returns>
Mesh warpGrid(Mesh& grid, const ImageFloat& disparity, const float scaling_factor, const BilinearSamplerFloat& sampleBilinear)
{
    // Create a copy of the input mesh (all values are copied).
    auto new_grid = Mesh { grid.vertices, grid.triangles };

    const float EDGE_EPSILON = 1e-5f * disparity.width;

    // Here is an example use of the bilinear interpolation (using the provided function argument).
    auto interpolated_value = sampleBilinear(disparity, glm::vec2(1.0f, 1.0f));

    // !!! Recommended test: sampleBilinear(disparity, glm::vec2(1.0f, 1.0f)); should return mean of the 4 pixels for any 2x2 image !!!

    //
    //    YOUR CODE GOES HERE
    //

    return new_grid;
}

/// <summary>
/// Forward-warps an image based on the given disparity and warp_factor.
/// </summary>
/// <param name="src_image">Source image.</param>
/// <param name="src_depth">Depth image used for Z-Testing</param>
/// <param name="disparity">Disparity of the source image in pixels.</param>
/// <param name="warp_factor">Multiplier of the disparity.</param>
/// <returns>ImageWithMask, containing the forward-warped image and a mask image. Mask=1 for valid pixels, Mask=0 for holes</returns>
ImageWithMask forwardWarpImage(const ImageRGB& src_image, const ImageFloat& src_depth, const ImageFloat& disparity, const float warp_factor)
{
    // The dimensions of src image, src depth and disparity maps all match.
    assert(src_image.width == disparity.width && src_image.height == disparity.height);
    assert(src_image.width == disparity.width && src_depth.height == src_depth.height);

    // Create a new image and depth map for the output.
    auto dst_image = ImageRGB(src_image.width, src_image.height);
    auto dst_mask = ImageFloat(src_depth.width, src_depth.height);
    // Fill the destination depth mask map with zero.
    std::fill(dst_mask.data.begin(), dst_mask.data.end(), 0.0f);
    auto dst_depth = ImageFloat(src_depth.width, src_depth.height);
    // Fill the destination depth map with a very large number.
    std::fill(dst_depth.data.begin(), dst_depth.data.end(), std::numeric_limits<float>::max());

    //
    // Note: Parallelize the code using OpenMP directives.
    //

    int w = src_image.width;
    int h = src_image.height;

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            auto src_pixel = src_image.data[getIndex(src_image, x, y)];
            auto src_pixel_depth = src_depth.data[getIndex(src_depth, x, y)];
            auto pix_disparity = disparity.data[getIndex(disparity, x, y)];

            int nx = x + std::floor((pix_disparity * warp_factor) + 0.5);

#pragma omp critical
            {
                if (!(pix_disparity == INVALID_VALUE || src_pixel_depth == INVALID_VALUE || nx < 0 || nx >= w || dst_depth.data[getIndex(dst_depth, nx, y)] <= src_pixel_depth)) {
                    dst_image.data[getIndex(dst_image, nx, y)] = src_pixel;
                    dst_depth.data[getIndex(dst_depth, nx, y)] = src_pixel_depth;
                    dst_mask.data[getIndex(dst_mask, nx, y)] = 1.0f;
                }
            }
        }
    }

    // Return the warped image.
    return ImageWithMask(dst_image, dst_mask);
}

/// <summary>
/// Applies Gaussian filter on the given image to fill the holes
/// indicated by a binary mask (mask==0 -> missing pixel).
/// Other pixels remain unchanged.
/// </summary>
/// <param name="img_forward">The image to be filtered and its mask.</param>
/// <param name="size">The kernel size. It is always odd.</param>
/// <returns>ImageRGB, the filtered forward warping image.</returns>
ImageRGB inpaintHoles(const ImageWithMask& img, const int size)
{
    // The filter size is always odd.
    assert(size % 2 == 1);

    // Rule of thumb for gaussian's std dev.
    const float sigma = (size - 1) / 2 / 3.2f;

    // The output is initialized by copy of the input.
    auto result = ImageRGB(img.image);

    int w = img.image.width;
    int h = img.image.height;
    int hs = (size - 1) / 2;

#pragma omp parallel for collapse(2)
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < w; y++) {

            if (img.mask.data[getIndex(img.mask, x, y)] == 1)
                continue;

            auto res = glm::vec3(0.0);
            float weight = 0.0;

            for (int i = -hs; i <= hs; i++) {
                for (int j = -hs; j <= hs; j++) {
                    int nx = x + i;
                    int ny = y + j;

                    if (nx < 0 || ny < 0 || nx >= w || ny >= h)
                        continue;

                    if (img.mask.data[getIndex(img.mask, nx, ny)] == 0)
                        continue;

                    float w_i = gauss(nx, sigma, x) * gauss(ny, sigma, ny);
                    res += img.image.data[getIndex(img.image, nx, ny)] * w_i;
                    weight += w_i;
                }
            }

            result.data[getIndex(img.image, x, y)] = res / weight;
        }
    }

    // Return inpainted image.
    return result;
}

/// <summary>
/// Backward-warps an image using a warped mesh.
/// </summary>
/// <param name="src_image">Source image.</param>
/// <param name="src_depth">Depth image used for Z-Testing</param>
/// <param name="src_grid">Source grid.</param>
/// <param name="dst_grid">The warped grid.</param>
/// <param name="sampleBilinear">a function that bilinearly samples ImageFloat</param>
/// <param name="sampleBilinearRGB">a function that bilinearly samples ImageRGB</param>
/// <returns>ImageRGB, the backward-warped image.</returns>
ImageRGB backwardWarpImage(const ImageRGB& src_image, const ImageFloat& src_depth, const Mesh& src_grid, const Mesh& dst_grid,
    const BilinearSamplerFloat& sampleBilinear, const BilinearSamplerRGB& sampleBilinearRGB)
{
    // The dimensions of src image and depth match.
    assert(src_image.width == src_depth.width && src_image.height == src_depth.height);
    // We assume that both grids have the same size and also the same order (ie., there is 1:1 triangle pairing).
    // This implies that the content of index buffers of both meshes are exactly equal (we do not test it here).
    assert(src_grid.triangles.size() == dst_grid.triangles.size());

    // Create a new image and depth map for the output.
    auto dst_image = ImageRGB(src_image.width, src_image.height);
    auto dst_depth = ImageFloat(src_depth.width, src_depth.height);
    // Fill the destination depth map with a very large number.
    std::fill(dst_depth.data.begin(), dst_depth.data.end(), 1e20f);

    // Example of testing point [0.1, 0.2] is inside a triangle.
    bool is_point_inside = isPointInsideTriangle(glm::vec2(0.1, 0.2), glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(0, 1));

    // Example of computing barycentric coordinates of a point [0.1, 0.2] inside a triangle.
    glm::vec3 bc = barycentricCoordinates(glm::vec2(0.1, 0.2), glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(0, 1));

    //
    //    YOUR CODE GOES HERE
    //

    // Return the warped image.
    return dst_image;
}

/// <summary>
/// Returns an anaglyph image.
/// </summary>
/// <param name="image_left">left RGB image</param>
/// <param name="image_right">right RGB image</param>
/// <param name="saturation">color saturation to apply</param>
/// <returns>ImageRGB, the anaglyph image.</returns>
ImageRGB createAnaglyph(const ImageRGB& image_left, const ImageRGB& image_right, const float saturation)
{
    // An empty image for the resulting anaglyph.
    auto anaglyph = ImageRGB(image_left.width, image_left.height);

    // Example: RGB->HSV->RGB should be approx identity.
    auto rgb_orig = glm::vec3(0.2, 0.6, 0.4);
    auto rgb_should_be_same = hsvToRgb(rgbToHsv(rgb_orig)); // expect rgb == rgb_2 (up to numerical precision)

    for (int i = 0; i < anaglyph.data.size(); i++) {
        auto hsv_left = rgbToHsv(image_left.data[i]);
        auto hsv_right = rgbToHsv(image_right.data[i]);

        hsv_left.y *= saturation;
        hsv_right.y *= saturation;

        auto rgb_left = hsvToRgb(hsv_left);
        auto rgb_right = hsvToRgb(hsv_right);

        glm::vec3 final{
            rgb_left.r,
            rgb_right.g,
            rgb_right.b
        };

        anaglyph.data[i] = final;
    }

    // Returns a single analgyph image.
    return anaglyph;
}

/// <summary>
/// Rotates a grid counter-clockwise around the center by a given angle in degrees.
/// </summary>
/// <param name="grid">The original mesh.</param>
/// <param name="center">The center of the rotation (in pixel coords).</param>
/// <param name="angle">Angle in degrees.</param>
/// <returns>Mesh, the rotated grid.</returns>
Mesh rotatedWarpGrid(Mesh& grid, const glm::vec2& center, const float& angle)
{
    // Create a copy of the input mesh (all values are copied).
    auto new_grid = Mesh { grid.vertices, grid.triangles };

    const float DEGREE2RADIANS = 0.0174532925f;

    //
    //    YOUR CODE GOES HERE
    //

    return new_grid;
}

/// <summary>
/// Rotate an image using backward warping based on the provided meshes.
/// </summary>
/// <param name="image">input image</param>
/// <param name="src_grid">original grid</param>
/// <param name="dst_grid">rotated grid</param>
/// <param name="sampleBilinear">a function that bilinearly samples ImageFloat</param>
/// <param name="sampleBilinearRGB">a function that bilinearly samples ImageRGB</param>
/// <returns>rotated image, has the same size as input</returns>
ImageRGB rotateImage(const ImageRGB& image, const Mesh& src_grid, const Mesh& dst_grid, const BilinearSamplerFloat& sampleBilinear, const BilinearSamplerRGB& sampleBilinearRGB)
{

    //
    // Unused pixels should be black.
    // Pixel that fall outside of the image should be discarded.
    //
    //    YOUR CODE GOES HERE
    //
    return ImageRGB(1, 1); // replace
}