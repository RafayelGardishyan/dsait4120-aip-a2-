#include "your_code_here.h"

static const std::filesystem::path dataDirPath { DATA_DIR };
static const std::filesystem::path outDirPath { OUTPUT_DIR };

/// <summary>
/// Enum of the provided scenes.
/// </summary>
enum InputSelection : int {
    Mini = 0,
    Middlebury = 1,
};

/// <summary>
/// Feel free to edit these params or add new.
/// </summary>
static const SceneParams SceneParams_Mini = {
    4, 0, 0, 2,
    2.0f, 1.0f, 4.0f, 2.0f, 6.0f,
    5, 0.02f, 50.0f, 0.95f, 1.0f,
};
static const SceneParams SceneParams_Middlebury = {
    105, 35, 0, 28,
    64.0f, 0.25f, 590.0f, 550.0f, 670.0f,
    19, 0.05f, 1.0f, 30.0f, -1.0f,
};

/// <summary>
/// Main method. Runs default tests. Feel free to modify it, add more tests and experiments,
/// change the input images etc. The file is not part of the solution. All solutions have to 
/// implemented in "your_code_here.h".
/// </summary>
/// <returns>0</returns>
int main()
{
    // Do not add any noise to the saved images.
    std::srand(unsigned(4733668));
    const float im_write_noise_level = 0.0f;

    std::chrono::steady_clock::time_point time_start, time_end;
    printOpenMPStatus();
    
    
    // 0. Load inputs from files. 
    ImageRGB image;
    ImageFloat src_disparity;
    SceneParams scene_params;
     
    // Change your inputs here!
    //const auto input_select = InputSelection::Mini;
    const auto input_select = InputSelection::Middlebury;

    switch (input_select) {
        case InputSelection::Mini:
            // The 5x4 mini image.
            image = ImageRGB(dataDirPath / "mini/image.png");
            src_disparity = loadDisparity(dataDirPath / "mini/disparity.png");
            scene_params = SceneParams_Mini;
            break;
        case InputSelection::Middlebury:
            // The Middleburry Stereo dataset.
            // Find plenty more in: https://vision.middlebury.edu/stereo/data/scenes2005/HalfSize/zip-2views/
            // Note that we use the half-sized images for faster testing.
            image = ImageRGB(dataDirPath / "moebius/view1.png");
            src_disparity = loadDisparity(dataDirPath / "moebius/disp1.png", 0.5f);
            scene_params = SceneParams_Middlebury;
            break;
        default:
            throw std::runtime_error("Invalid scene ID.");
    }


    // Test save the inputs.
    image.writeToFile(outDirPath / "0_src_image.png", 1.0f, im_write_noise_level);
    disparityToColor(src_disparity, scene_params.in_disp_min, scene_params.in_disp_max).writeToFile(outDirPath / "0_src_disparity.png");


    // 1. Filter depth map (guided filter). We do this to remove (most of) the holes.
    time_start = std::chrono::steady_clock::now();
    auto disparity_filtered = jointBilateralFilter(src_disparity, image, scene_params.bilateral_size, scene_params.bilateral_joint_sigma);
    time_end = std::chrono::steady_clock::now();
    std::cout << "[1] Bilateral filter | Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;
    disparityToColor(disparity_filtered, scene_params.in_disp_min, scene_params.in_disp_max).writeToFile(outDirPath / "1_disparity_filtered.png", 1.0f, im_write_noise_level);


    // 2. Disparity to depth. We need depth to do Z-testing.
    time_start = std::chrono::steady_clock::now();
    auto linear_depth = disparityToNormalizedDepth(disparity_filtered);
    time_end = std::chrono::steady_clock::now();
    std::cout << "[2] Disparity -> Depth | Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;
    linear_depth.writeToFile(outDirPath / "2_linear_depth.png", 1.0f, im_write_noise_level);

    // 2.1 Forward warp the image. We use the original (unscaled) disparity to test the warping.
    // This has the benefit that we have a GT for this from the dataset.
    // This test will show us why forward warping is not the best fit.
    // Outputs structure with .image and .mask members.
    time_start = std::chrono::steady_clock::now();
    ImageWithMask warped_forward = forwardWarpImage(image, linear_depth, disparity_filtered, scene_params.warp_scale);
    time_end = std::chrono::steady_clock::now();
    std::cout << "[2.1] Forward warp the image | Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;
    warped_forward.image.writeToFile(outDirPath / "2_1_warped_forward.png", 1.0f, im_write_noise_level);       
    warped_forward.mask.writeToFile(outDirPath / "2_1_warped_forward_mask.png", 1.0f, im_write_noise_level);       


    // 2.2 Inpaint the holes in the forward warping.
    // We use a very simple filter to replace pixels with mask ~ 0.
    time_start = std::chrono::steady_clock::now();
    auto inpainted_forward = inpaintHoles(warped_forward, scene_params.bilateral_size);
    time_end = std::chrono::steady_clock::now();
    std::cout << "[2.2] Inpaint the holes | Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;
    inpainted_forward.writeToFile(outDirPath / "2_2_inpainted_warped_forward.png", 1.0f, im_write_noise_level);

    // 3. Now we convert the depth back to disparity.
    // We do this to have control over the disparity magnitude and position.
    // This allows us to tune a comfortably viewable stereoscopic effect.
    time_start = std::chrono::steady_clock::now();
    auto target_disparity = normalizedDepthToDisparity(
        linear_depth,
        scene_params.iod_mm,
        scene_params.px_size_mm,
        scene_params.screen_distance_mm,
        scene_params.near_plane_mm,
        scene_params.far_plane_mm);
    time_end = std::chrono::steady_clock::now();
    std::cout << "[3] Depth -> Target disparity | Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;
    disparityToColor(target_disparity, scene_params.out_disp_min, scene_params.out_disp_max).writeToFile(outDirPath / "3_target_disparity.png", 1.0f, im_write_noise_level);

    // 4. Repeat forward warping twice to obtain left and right stereo pair.
    // We use the disparity scaling warp_factor to obtain mirrored effect for left and right image.
    time_start = std::chrono::steady_clock::now();
    std::vector<ImageRGB> image_pair;
    for (int i = 0; i < 2; i++) {
        // The total disparity is split in half between both images as each gets shifted in an opposite direction.
        auto warp_factor = i == 0 ? -0.5f : 0.5f;

        // Forward warp the image. We use the scaled disparity.
        auto img_forward = forwardWarpImage(image, linear_depth, target_disparity, warp_factor);
        // Inpaint the holes in the forward warping image.
        ImageRGB dst_image = inpaintHoles(img_forward, scene_params.bilateral_size);

        dst_image.writeToFile(outDirPath / (i == 0 ? "4_stereo_froward_left.png" : "4_stereo_froward_right.png"), 1.0f, im_write_noise_level);
        image_pair.push_back(std::move(dst_image));
    }
    time_end = std::chrono::steady_clock::now();
    std::cout << "[4] Create a stereo pair from froward warp image | Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;

    // 5. Combine the left/right image pair into an anaglyph stereoscopic image.
    time_start = std::chrono::steady_clock::now();
    auto forward_anaglyph = createAnaglyph(image_pair[0], image_pair[1], 0.3f);
    time_end = std::chrono::steady_clock::now();
    std::cout << "[5] Create anaglyph | Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;
    forward_anaglyph.writeToFile(outDirPath / "5_forward_anaglyph.png", 1.0f, im_write_noise_level);

	// --- Advanced section (mesh-based warping)---

    // 6. Create a grid that covers all pixels.
    time_start = std::chrono::steady_clock::now();
    auto src_grid = createWarpingGrid(image.width, image.height);
    time_end = std::chrono::steady_clock::now();
    std::cout << "[6] Create a grid | Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;
    plotGridMesh(src_grid, { image.width * scene_params.grid_viz_im_scale, image.height * scene_params.grid_viz_im_scale }, scene_params.grid_viz_im_scale * scene_params.grid_viz_tri_scale).writeToFile(outDirPath / "6_src_grid.png", 1.0f, im_write_noise_level);   

    // 7. Warp the grid by moving the vertices according to the disparity.
    time_start = std::chrono::steady_clock::now();
    // Note that we are passing your sampleBilinear() function as an argument. That allows us to replace it with reference
    // implementation during grading and therefore judge it and warpGrid() independently without propagating errors in one to the other.
    auto dst_grid = warpGrid(src_grid, disparity_filtered, scene_params.warp_scale, sampleBilinear<float>);
    time_end = std::chrono::steady_clock::now();
    std::cout << "[7] Warp the grid | Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;
    plotGridMesh(dst_grid, { image.width * scene_params.grid_viz_im_scale, image.height * scene_params.grid_viz_im_scale }, scene_params.grid_viz_im_scale * scene_params.grid_viz_tri_scale).writeToFile(outDirPath / "7_dst_grid.png", 1.0f, im_write_noise_level);   

    // 8. Resample the image using the warped grid (= mesh-based warping).
    // Compare the result to the warped_forward.png. Should have no holes.
    time_start = std::chrono::steady_clock::now();
    auto warped_backward = backwardWarpImage(image, linear_depth, src_grid, dst_grid, sampleBilinear<float>, sampleBilinear<glm::vec3>);
    time_end = std::chrono::steady_clock::now();
    std::cout << "[8] Backward warp the image | Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;
    warped_backward.writeToFile(outDirPath / "8_warped_backward.png", 1.0f, im_write_noise_level);

    // 9. Repeat the mesh-based warping twice to obtain left and right stereo pair.
    // We use the disparity scaling warp_factor to obtain mirrored effect for left and right image.
    time_start = std::chrono::steady_clock::now();
    image_pair.clear();
    for (int i = 0; i < 2; i++) {
        // The total disparity is split in half between both images as each gets shifted in an opposite direction.
        auto warp_factor = i == 0 ? -0.5f : 0.5f;

        // Warp the grid. We can reuse src_grid because it does not change.
        auto dst_grid = warpGrid(src_grid, target_disparity, warp_factor, sampleBilinear<float>);

        // Resample the image.
        ImageRGB dst_image = backwardWarpImage(image, linear_depth, src_grid, dst_grid, sampleBilinear<float>, sampleBilinear<glm::vec3>);
        dst_image.writeToFile(outDirPath / (i == 0 ? "9_stereo_backward_left.png" : "9_stereo_backward_right.png"), 1.0f, im_write_noise_level);
        image_pair.push_back(std::move(dst_image));
    }
    time_end = std::chrono::steady_clock::now();
    std::cout << "[9] Create a stereo pair from backward warp image | Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;

    // 10. Combine the left/right image pair into an anaglyph stereoscopic image.
    time_start = std::chrono::steady_clock::now();
    auto backward_anaglyph = createAnaglyph(image_pair[0], image_pair[1], 0.3f);
    time_end = std::chrono::steady_clock::now();
    std::cout << "[10] Create anaglyph from backward warp image| Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;
    backward_anaglyph.writeToFile(outDirPath / "10_anaglyph_backward.png", 1.0f, im_write_noise_level);

    	
	//
    // --- Independent exercise (no support from TAs) ----
    // 

    // 11. Rotate the grid by 45 deg counterclockwise around the center of the image.
    time_start = std::chrono::steady_clock::now();
    auto dst_grid_rotate = rotatedWarpGrid(src_grid, glm::vec2(disparity_filtered.width * 0.5f, disparity_filtered.height * 0.5f), 45.0f);
    time_end = std::chrono::steady_clock::now();
    std::cout << "[11] Warp the grid | Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;
    
    // 12. Resample the image using the warped grid (= mesh-based warping).
    // Compare the result to the warped_forward.png. Should have no holes.
    time_start = std::chrono::steady_clock::now();
    auto rotated_warped_backward = rotateImage(image, src_grid, dst_grid_rotate, sampleBilinear<float>, sampleBilinear<glm::vec3>);
    time_end = std::chrono::steady_clock::now();
    std::cout << "[12] Backward warp the image | Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;
    rotated_warped_backward.writeToFile(outDirPath / "12_rotated_warped_backward.png", 1.0f, im_write_noise_level);

	std::cout << "All done!" << std::endl;

	
    return 0;
}
