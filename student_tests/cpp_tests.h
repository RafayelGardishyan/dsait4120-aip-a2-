#include "grading_helpers.h"
#include "your_code_here.h"
#include "helpers.h"
// Suppress warnings in third-party code.
#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <catch2/catch_all.hpp>
DISABLE_WARNINGS_POP()
#include <algorithm>
#include <cmath>
#include <ostream>
#include <vector>

static const std::filesystem::path dataDirPath { DATA_DIR };
static const std::filesystem::path rawDataDirPath { dataDirPath / "raw_data" };
static const std::filesystem::path outDirPath { OUTPUT_DIR };

static const SceneParams SceneParams_Mini = {
    4,
    0,
    0,
    2,
    2.0f,
    1.0f,
    4.0f,
    2.0f,
    6.0f,
    5,
    0.02f,
    50.0f,
    0.95f,
    1.0f,
};
static const SceneParams SceneParams_Middlebury_Art = {
    120,
    50,
    0,
    30,
    64.0f,
    0.25f,
    600.0f,
    550.0f,
    650.0f,
    19,
    0.05f,
    1.0f,
    30.0f,
    -1.0f,
};

// Mini case
const ImageRGB mini_image = ImageRGB(dataDirPath / "mini/image.png");
const ImageFloat mini_disparity = loadDisparity(dataDirPath / "mini/disparity.png");
SceneParams mini_scene_params = SceneParams_Mini;
auto mini_variables = []() {
    auto vars = TestVariables();
    vars.readBinary(rawDataDirPath / "mini_variables");
    return vars;
}();
ImageFloat& mini_disparity_filtered = mini_variables.disparity_filtered;
ImageFloat& mini_linear_depth = mini_variables.linear_depth;
ImageFloat& mini_target_disparity = mini_variables.target_disparity;
Mesh& mini_src_grid = mini_variables.src_grid;
Mesh& mini_dst_grid = mini_variables.dst_grid;

// Middlebury case
const ImageRGB Middlebury_image = ImageRGB(dataDirPath / "art/view1.png");
const ImageFloat Middlebury_disparity = loadDisparity(dataDirPath / "art/disp1.png", 0.5f);
SceneParams Middlebury_scene_params = SceneParams_Middlebury_Art;
auto Middlebury_variables = []() {
    auto vars = TestVariables();
    vars.readBinary(rawDataDirPath / "Middlebury_variables");
    return vars;
}();

ImageFloat& Middlebury_disparity_filtered = Middlebury_variables.disparity_filtered;
ImageFloat& Middlebury_linear_depth = Middlebury_variables.linear_depth;
ImageFloat& Middlebury_target_disparity = Middlebury_variables.target_disparity;
Mesh& Middlebury_src_grid = Middlebury_variables.src_grid;
Mesh& Middlebury_dst_grid = Middlebury_variables.dst_grid;

// Reindeer case
const ImageRGB Reindeer_image = ImageRGB(dataDirPath / "reindeer/view1.png");
const ImageFloat Reindeer_disparity = loadDisparity(dataDirPath / "reindeer/disp1.png", 0.5f);
SceneParams Reindeer_scene_params = SceneParams_Middlebury_Art; // NOTE: Using the same params as for the other picture
auto Reindeer_variables = []() {
    auto vars = TestVariables();
    vars.readBinary(rawDataDirPath / "Reindeer_variables");
    return vars;
}();
ImageFloat& Reindeer_disparity_filtered = Reindeer_variables.disparity_filtered;
ImageFloat& Reindeer_linear_depth = Reindeer_variables.linear_depth;
ImageFloat& Reindeer_target_disparity = Reindeer_variables.target_disparity;
Mesh& Reindeer_src_grid = Reindeer_variables.src_grid;
Mesh& Reindeer_dst_grid = Reindeer_variables.dst_grid;


void checkjointBilateralFilter(const ImageFloat& disparity, const ImageRGB& guide, const int radius, const float guide_sigma, bool test_tmo, const std::string& id = "")
{
    auto reference_output = ImageFloat();
    reference_output.readBinary(rawDataDirPath / ("reference_checkJointBilateralFilter_" + id + "_output.bin"));
    auto user_output = jointBilateralFilter(disparity, guide, radius, guide_sigma);

    if (test_tmo) {
        auto reference_output_rgb = disparityToColor(reference_output, SceneParams_Middlebury_Art.in_disp_min, SceneParams_Middlebury_Art.in_disp_max);
        auto user_output_rgb = disparityToColor(user_output, SceneParams_Middlebury_Art.in_disp_min, SceneParams_Middlebury_Art.in_disp_max);  
        CHECK(calcImageRMSE(reference_output_rgb, user_output_rgb) == APPROX_FLOAT(0.0f));
    } else {
        CHECK(calcImageRMSE(reference_output, user_output) == APPROX_FLOAT(0.0f));
    }
}

TEST_CASE("1_jointBilateralFilter")
{
    SECTION("MiniImage")
    {
        checkjointBilateralFilter(mini_disparity, mini_image, SceneParams_Mini.bilateral_size, SceneParams_Mini.bilateral_joint_sigma, false, "MiniImage");
    }

    SECTION("MiddleburyImage")
    {
        SceneParams scene_params = SceneParams_Middlebury_Art;

        checkjointBilateralFilter(Middlebury_disparity, Middlebury_image, scene_params.bilateral_size, scene_params.bilateral_joint_sigma, true, "MiddleburyImage");
    }

    SECTION("ReindeerImage")
    {
        SceneParams scene_params = Reindeer_scene_params;

        checkjointBilateralFilter(Reindeer_disparity, Reindeer_image, scene_params.bilateral_size, scene_params.bilateral_joint_sigma, true, "ReindeerImage");
    }
}

void checknormalizeValidValues(const ImageFloat& disparity, const std::string& id = "")
{
    auto reference_depth = ImageFloat(disparity.width, disparity.height);
    auto user_depth = ImageFloat(disparity.width, disparity.height);

    auto num_pixels = disparity.width * disparity.height;
#pragma omp parallel for
    for (auto i = 0; i < num_pixels; i++) {
        if (disparity.data[i] != INVALID_VALUE) {
            reference_depth.data[i] = 1.0f / disparity.data[i];
            user_depth.data[i] = 1.0f / disparity.data[i];
        }
    }

    reference_depth.readBinary(rawDataDirPath / ("reference_checkNormalizeValidValues_" + id + "_output.bin"));
    normalizeValidValues(user_depth);

    CHECK(calcImageRMSE(reference_depth, user_depth) == APPROX_FLOAT(0.0f));
}

TEST_CASE("2a_normalizeValidValues")
{
    SECTION("MiniImage")
    {
        checknormalizeValidValues(mini_disparity_filtered, "MiniImage");
    }

    SECTION("MiddleburyImage")
    {
        checknormalizeValidValues(Middlebury_disparity_filtered, "MiddleburyImage");
    }

    SECTION("ReindeerImage")
    {
        checknormalizeValidValues(Reindeer_disparity_filtered, "ReindeerImage");
    }

}

void checkdisparityToNormalizedDepth(const ImageFloat& disparity, const std::string& id = "")
{
    auto reference_depth = ImageFloat();
    reference_depth.readBinary(rawDataDirPath / ("reference_checkDisparityToNormalizedDepth_" + id + "_output.bin"));
    auto user_depth = disparityToNormalizedDepth(disparity);
    
    CHECK(calcImageRMSE(reference_depth, user_depth) == APPROX_FLOAT(0.0f));
}


TEST_CASE("2b_disparityToNormalizedDepth")
{
    SECTION("MiniImage")
    {
        checkdisparityToNormalizedDepth(mini_disparity_filtered, "MiniImage");
    }

    SECTION("MiddleburyImage")
    {
        checkdisparityToNormalizedDepth(Middlebury_disparity_filtered, "MiddleburyImage");
    }

    SECTION("ReindeerImage")
    {
        checkdisparityToNormalizedDepth(Reindeer_disparity_filtered, "ReindeerImage");
    }

}

void checkforwardWarpImage(const ImageRGB& src_image, const ImageFloat& src_depth, const ImageFloat& disparity, const float warp_factor, const std::string& id = "")
{
    auto reference_warped_forward_a = ImageWithMask();
    auto reference_warped_forward_b = ImageWithMask();
    reference_warped_forward_a.readBinary(rawDataDirPath / ("reference_forwardWarpImage_" + id + "_a"));
    reference_warped_forward_b.readBinary(rawDataDirPath / ("reference_forwardWarpImage_" + id + "_b"));
    auto user_warped_forward = forwardWarpImage(src_image, src_depth, disparity, warp_factor);

    CHECK(std::min(calcImageRMSE(reference_warped_forward_a.image, user_warped_forward.image),
              calcImageRMSE(reference_warped_forward_b.image, user_warped_forward.image))
        == APPROX_FLOAT_5e(0.0f));

    CHECK(std::min(calcImageRMSE(reference_warped_forward_a.mask, user_warped_forward.mask),
              calcImageRMSE(reference_warped_forward_b.mask, user_warped_forward.mask))
        == APPROX_FLOAT_5e(0.0f));
}

TEST_CASE("2.5_forwardWarpImage")
{
    SECTION("MiniImage")
    {
        checkforwardWarpImage(mini_image, mini_linear_depth, mini_disparity_filtered, SceneParams_Mini.warp_scale, "MiniImage");
    }

    SECTION("MiddleburyImage")
    {
        checkforwardWarpImage(Middlebury_image, Middlebury_linear_depth, Middlebury_disparity_filtered, SceneParams_Middlebury_Art.warp_scale, "MiddleburyImage");
    }

    SECTION("ReindeerImage")
    {
        checkforwardWarpImage(Reindeer_image, Reindeer_linear_depth, Reindeer_disparity_filtered, Reindeer_scene_params.warp_scale, "ReindeerImage");
    }
}

void checkinpaintHoles(const ImageWithMask& img, const int size, const std::string& id = "")
{
    auto reference_inpainted_img = ImageRGB();
    reference_inpainted_img.readBinary(rawDataDirPath / ("reference_inpaintHoles_" + id + "_output.bin"));
    auto user_inpainted_img = inpaintHoles(img, size);

    CHECK(calcImageRMSEWithMask(reference_inpainted_img, user_inpainted_img, img.mask) == APPROX_FLOAT_5e(0.0f));
}

TEST_CASE("2.6_checkinpaintHoles")
{
    SECTION("MiniImage")
    {
        auto reference_warped_forward = ImageWithMask();
        reference_warped_forward.readBinary(rawDataDirPath / "reference_inpaintHoles_MiniImage_input.bin");
        checkinpaintHoles(reference_warped_forward, SceneParams_Mini.bilateral_size, "MiniImage");
    }

    SECTION("MiddleburyImage")
    {
        auto reference_warped_forward = ImageWithMask();
        reference_warped_forward.readBinary(rawDataDirPath / "reference_inpaintHoles_MiddleburyImage_input.bin");
        checkinpaintHoles(reference_warped_forward, SceneParams_Middlebury_Art.bilateral_size, "MiddleburyImage");
    }

    SECTION("ReindeerImage")
    {
        auto reference_warped_forward = ImageWithMask();
        reference_warped_forward.readBinary(rawDataDirPath / "reference_inpaintHoles_ReindeerImage_input.bin");
        checkinpaintHoles(reference_warped_forward, Reindeer_scene_params.bilateral_size, "ReindeerImage");
    }
}

void checkcreateWarpingGrid(const int width, const int height, const SceneParams scene_params, const std::string& id = "")
{   
    auto reference_grid = Mesh();
    reference_grid.readBinary(rawDataDirPath / ("reference_createWarpingGrid_" + id + "_output.bin"));
    auto reference_gridImage = plotGridMesh(reference_grid, { width * scene_params.grid_viz_im_scale, height * scene_params.grid_viz_im_scale }, scene_params.grid_viz_tri_scale);
    auto user_grid = createWarpingGrid(width, height);
    auto user_gridImage = plotGridMesh(user_grid, { width * scene_params.grid_viz_im_scale, height * scene_params.grid_viz_im_scale }, scene_params.grid_viz_tri_scale);
    
    CHECK(calcImageRMSE(reference_gridImage, user_gridImage) == APPROX_FLOAT(0.0f));
}

TEST_CASE("3_createWarpingGrid")
{
    SECTION("MiniImage")
    {
        checkcreateWarpingGrid(mini_image.width, mini_image.height, SceneParams_Mini, "MiniImage");
    }

    SECTION("MiddleburyImage")
    {
        checkcreateWarpingGrid(Middlebury_image.width, Middlebury_image.height, SceneParams_Middlebury_Art, "MiddleburyImage");
    }

    SECTION("ReindeerImage")
    {
        checkcreateWarpingGrid(Reindeer_image.width, Reindeer_image.height, Reindeer_scene_params, "ReindeerImage");
    }
}



template <typename T>
void checksampleBilinear(const Image<T>& image, const std::string& id = "")
{
    int nCount = 10;
    std::vector<glm::vec2> test_positions;
    // Generate test positions along the diagonal
    for (int i = 0; i < nCount; i++) {
        float factor = (1.0f / (nCount - 1)) * i;
        test_positions.push_back(glm::vec2(factor * image.width, factor * image.height));
    }

    for (int i = 0; i < nCount;i++)
    {
        glm::vec2 rel_pos = test_positions[i];

        float reference_sample = loadFloatFromFile(rawDataDirPath / ("reference_checkSampleBilinear_" + id + "_pos_" + std::to_string(i) + "_output.bin"));
        auto user_sample = sampleBilinear(image, rel_pos);

        std::stringstream ss;
        ss << "Testing sampling position [" << rel_pos.x <<  "; " << rel_pos.y << "] in an image with size [" << image.width << "; " << image.height << "]";

        INFO(ss.str()); // Adds context to test output
        auto pass = reference_sample == APPROX_FLOAT_EPS(user_sample, 1e-4f);
        CHECK(reference_sample == APPROX_FLOAT_EPS(user_sample, 1e-4f));
        if (!pass) {
            // Log detailed message for failing cases
            CAPTURE(ss.str(), user_sample, reference_sample);
        }
    }
}

TEST_CASE("4a_sampleBilinear")
{
    SECTION("MiniImage")
    {
        checksampleBilinear(mini_disparity_filtered, "MiniImage");
    }

    SECTION("MiddleburyImage")
    {
        checksampleBilinear(Middlebury_disparity_filtered, "MiddleburyImage");
    }

}

void checkwarpGrid(Mesh& grid, ImageFloat& disparity, const SceneParams scene_params, const std::string& id = "")
{   
    auto reference_grid = Mesh();
    reference_grid.readBinary(rawDataDirPath / ("reference_warpGrid_" + id + "_output.bin"));
    auto reference_gridImage = plotGridMesh(reference_grid, { disparity.width * scene_params.grid_viz_im_scale, disparity.height * scene_params.grid_viz_im_scale }, scene_params.grid_viz_tri_scale);
    
    // Note: Assumes sampleBilinear works correctly.
    auto user_grid = warpGrid(grid, disparity, scene_params.warp_scale, sampleBilinear<float>);
    auto user_gridImage = plotGridMesh(user_grid, { disparity.width * scene_params.grid_viz_im_scale, disparity.height * scene_params.grid_viz_im_scale }, scene_params.grid_viz_tri_scale);

    CHECK(calcImageRMSE(reference_gridImage, user_gridImage) == APPROX_FLOAT(0.0f));
}

TEST_CASE("4b_warpGrid")
{
    SECTION("MiniImage")
    {
        checkwarpGrid(mini_src_grid, mini_disparity_filtered, mini_scene_params, "MiniImage");
    }

    SECTION("MiddleburyImage")
    {
        checkwarpGrid(Middlebury_src_grid, Middlebury_disparity_filtered, Middlebury_scene_params, "MiddleburyImage");
    }

    SECTION("ReindeerImage")
    {
        checkwarpGrid(Reindeer_src_grid, Reindeer_disparity_filtered, Reindeer_scene_params, "ReindeerImage");
    }

}

void checkrotatedWarpGrid(const ImageRGB& src_image, Mesh& grid, const glm::vec2& center, const float& angle, const SceneParams scene_params, const std::string& id = "")
{
    auto reference_grid = Mesh();
    reference_grid.readBinary(rawDataDirPath / ("reference_rotatedWarpGrid_reference_grid_" + id + "_output.bin"));
    auto user_grid = rotatedWarpGrid(grid, center, angle);

    auto dummy_depth = ImageFloat(src_image.width, src_image.height);
    auto reference_result = ImageRGB();
    reference_result.readBinary(rawDataDirPath / ("reference_rotatedWarpGrid_reference_result_" + id + "_output.bin"));

    // Note: Assumes that backwardWarpImage and sampleBilinear work correctly.
    auto user_result = backwardWarpImage(src_image, dummy_depth, grid, user_grid, sampleBilinear<float>, sampleBilinear<glm::vec3>);

    CHECK(calcImageRMSE(reference_result, user_result) == APPROX_FLOAT(0.0f));
}

TEST_CASE("4c_rotatedWarpGrid")
{
    SECTION("MiniImage")
    {
        glm::vec2 rel_pos = glm::vec2(0.5f, 0.5f);
        glm::vec2 center = glm::vec2(mini_disparity_filtered.width, mini_disparity_filtered.height) * rel_pos;
        float rotate_angle = 45.0f;
        checkrotatedWarpGrid(mini_image, mini_src_grid, center, rotate_angle, mini_scene_params, "MiniImage");
    }

    SECTION("MiddleburyImage")
    {
        glm::vec2 rel_pos = glm::vec2(0.5f, 0.5f);
        glm::vec2 center = glm::vec2(Middlebury_disparity_filtered.width, Middlebury_disparity_filtered.height) * rel_pos;

        float rotate_angle = 45.0f;
        checkrotatedWarpGrid(Middlebury_image, Middlebury_src_grid, center, rotate_angle, Middlebury_scene_params, "MiddleburyImage");
    }

    SECTION("ReindeerImage")
    {
        glm::vec2 rel_pos = glm::vec2(0.4f, 0.7f);
        glm::vec2 center = glm::vec2(Reindeer_disparity_filtered.width, Reindeer_disparity_filtered.height) * rel_pos;
        float rotate_angle = 10.0f;
        checkrotatedWarpGrid(Reindeer_image, Reindeer_src_grid, center, rotate_angle, Reindeer_scene_params, "ReindeerImage");
    }

}

void checkbackwardWarpImage(const ImageRGB& src_image, const ImageFloat& src_depth, Mesh& src_grid, Mesh& dst_grid, const std::string& id = "")
{
    auto reference_warped_backward = ImageRGB();
    reference_warped_backward.readBinary(rawDataDirPath / ("reference_backwardWarpImage_" + id + "_output.bin"));

    // Note: Assumes that sampleBilinear works correctly.
    auto user_warped_backward = backwardWarpImage(src_image, src_depth, src_grid, dst_grid, sampleBilinear<float>, sampleBilinear<glm::vec3>);

    CHECK(calcImageRMSE(reference_warped_backward, user_warped_backward) == APPROX_FLOAT_EPS(0.0f, 0.1f));
}

TEST_CASE("5_backwardWarpImage")
{
    SECTION("MiniImage")
    {
        checkbackwardWarpImage(mini_image, mini_linear_depth, mini_src_grid, mini_dst_grid, "MiniImage");
    }

    SECTION("MiddleburyImage")
    {
        checkbackwardWarpImage(Middlebury_image, Middlebury_linear_depth, Middlebury_src_grid, Middlebury_dst_grid, "MiddleburyImage");
    }

    SECTION("ReindeerImage")
    {
        checkbackwardWarpImage(Reindeer_image, Reindeer_linear_depth, Reindeer_src_grid, Reindeer_dst_grid, "ReindeerImage");
    }

}

void checknormalizedDepthToDisparity(const ImageFloat& linear_depth, const SceneParams scene_params, const std::string& id = "")
{
    auto reference_target_disparity = ImageFloat();
    reference_target_disparity.readBinary(rawDataDirPath / ("reference_normalizedDepthToDisparity_" + id + "_output.bin"));

    auto user_target_disparity = normalizedDepthToDisparity(
        linear_depth,
        scene_params.iod_mm,
        scene_params.px_size_mm,
        scene_params.screen_distance_mm,
        scene_params.near_plane_mm,
        scene_params.far_plane_mm);

    CHECK(calcImageRMSE(reference_target_disparity, user_target_disparity) == APPROX_FLOAT(0.0f));
}

TEST_CASE("6_normalizedDepthToDisparity")
{
    SECTION("MiniImage")
    {
        checknormalizedDepthToDisparity(mini_linear_depth, mini_scene_params, "MiniImage");
    }

    SECTION("MiddleburyImage")
    {
        checknormalizedDepthToDisparity(Middlebury_linear_depth, Middlebury_scene_params, "MiddleburyImage");
    }

    SECTION("ReindeerImage")
    {
        checknormalizedDepthToDisparity(Reindeer_linear_depth, Reindeer_scene_params, "ReindeerImage");
    }

}

void checkcreateAnaglyph(const ImageRGB& src_image, const ImageFloat& linear_depth, const ImageFloat& target_disparity, Mesh& src_grid, const float saturation, const std::string& id = "")
{
    std::vector<ImageRGB> image_pair;
    image_pair.push_back(ImageRGB());
    image_pair.push_back(ImageRGB());
    image_pair[0].readBinary(rawDataDirPath / ("reference_createAnaglyph_" + id + "_left.bin"));
    image_pair[1].readBinary(rawDataDirPath / ("reference_createAnaglyph_" + id + "_right.bin"));

    auto reference_anaglyph = ImageRGB();
    reference_anaglyph.readBinary(rawDataDirPath / ("reference_createAnaglyph_" + id + "_output.bin"));
    auto user_anaglyph = createAnaglyph(image_pair[0], image_pair[1], saturation);

    CHECK(calcImageRMSE(reference_anaglyph, user_anaglyph) == APPROX_FLOAT(0.0f));
}

TEST_CASE("7_createAnaglyph")
{
    SECTION("MiniImage")
    {
        checkcreateAnaglyph(mini_image, mini_linear_depth, mini_target_disparity, mini_src_grid, 0.3f, "MiniImage");
    }

    SECTION("MiddleburyImage")
    {
        checkcreateAnaglyph(Middlebury_image, Middlebury_linear_depth, Middlebury_target_disparity, Middlebury_src_grid, 0.3f, "MiddleburyImage");
    }

    SECTION("ReindeerImage")
    {
        checkcreateAnaglyph(Reindeer_image, Reindeer_linear_depth, Reindeer_target_disparity, Reindeer_src_grid, 0.3f, "ReindeerImage");
    }

}

void checkrotateImage(const ImageRGB& src_image, const ImageFloat& src_depth, Mesh& src_grid, const glm::vec2& center, const float rotate_angle, const std::string& id = "")
{
    auto dst_grid_rotate = Mesh();
    dst_grid_rotate.readBinary(rawDataDirPath / ("reference_rotateImage_" + id + "_dst_grid_rotate.bin"));

    auto dummy_depth = ImageFloat(src_image.width, src_image.height);

    auto reference_rotateImage = ImageRGB();
    reference_rotateImage.readBinary(rawDataDirPath / ("reference_rotateImage_" + id + "_output.bin"));

    // Note: Assumes that sampleBilinear works correctly.
    auto user_rotateImage = rotateImage(src_image, src_grid, dst_grid_rotate, sampleBilinear<float>, sampleBilinear<glm::vec3>);

    bool dimensions_match = user_rotateImage.width == reference_rotateImage.width
        && user_rotateImage.height == reference_rotateImage.height;
    CHECK(dimensions_match);
    
    if (dimensions_match) {
        CHECK(calcImageRMSE(reference_rotateImage, user_rotateImage) == APPROX_FLOAT_EPS(0.0f, 0.1f));
    }
}

TEST_CASE("8_rotateImage")
{
    SECTION("MiniImage")
    {
        glm::vec2 rel_pos = glm::vec2(0.5f);
        float rotate_angle = 45;
        glm::vec2 center = glm::vec2(mini_disparity_filtered.width, mini_disparity_filtered.height) * rel_pos;

        checkrotateImage(mini_image, mini_linear_depth, mini_src_grid, center, rotate_angle, "MiniImage");
    }

    SECTION("MiddleburyImage")
    {
        glm::vec2 rel_pos = glm::vec2(0.5f);
        float rotate_angle = 45;
        glm::vec2 center = glm::vec2(Middlebury_disparity_filtered.width, Middlebury_disparity_filtered.height) * rel_pos;

        checkrotateImage(Middlebury_image, Middlebury_linear_depth, Middlebury_src_grid, center, rotate_angle, "MiddleburyImage");
    }

    SECTION("ReindeerImage")
    {
        glm::vec2 rel_pos = glm::vec2(0.8f,0.3f);
        float rotate_angle = 15;
        glm::vec2 center = glm::vec2(Reindeer_disparity_filtered.width, Reindeer_disparity_filtered.height) * rel_pos;

        checkrotateImage(Reindeer_image, Reindeer_linear_depth, Reindeer_src_grid, center, rotate_angle, "ReindeerImage");
    }

}