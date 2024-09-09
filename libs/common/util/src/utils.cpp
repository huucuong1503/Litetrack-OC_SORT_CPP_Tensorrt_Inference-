#include <utils.h>

Utils::Utils(){
    // std::cout<<"nice"std::endl;
}
std::vector<float> Utils::transform_image_to_crop(const std::vector<float>& box_in, 
                                        const std::vector<float>& box_extract, 
                                        float resize_factor, 
                                        const std::vector<float>& crop_sz, 
                                        bool normalize)
{
        // Calculate the center of the box_extract
    std::vector<float> box_extract_center = { 
        box_extract[0] + 0.5f * box_extract[2], 
        box_extract[1] + 0.5f * box_extract[3] 
    };

    // Calculate the center of the box_in
    std::vector<float> box_in_center = { 
        box_in[0] + 0.5f * box_in[2], 
        box_in[1] + 0.5f * box_in[3] 
    };

    // Calculate the transformed center of the box_out
    std::vector<float> box_out_center = { 
        (crop_sz[0] - 1) / 2.0f + (box_in_center[0] - box_extract_center[0]) * resize_factor, 
        (crop_sz[1] - 1) / 2.0f + (box_in_center[1] - box_extract_center[1]) * resize_factor 
    };

    // Calculate the width and height of the transformed box_out
    std::vector<float> box_out_wh = { 
        box_in[2] * resize_factor, 
        box_in[3] * resize_factor 
    };

    // Construct the output box_out
    std::vector<float> box_out = { 
        box_out_center[0] - 0.5f * box_out_wh[0], 
        box_out_center[1] - 0.5f * box_out_wh[1], 
        box_out_wh[0], 
        box_out_wh[1] 
    };

    // Normalize if requested
    if (normalize) {
        box_out[0] /= crop_sz[0];
        box_out[1] /= crop_sz[1];
        box_out[2] /= crop_sz[0];
        box_out[3] /= crop_sz[1];
    }

    return box_out;
}


std::vector<float> Utils::transform_bbox_to_crop(const std::vector<float>& box_in, float resize_factor, 
                                        const std::string& crop_type, const std::vector<float>& box_extract)
{
    std::vector<float> crop_sz(2);

    // Determine crop size based on crop type
    if (crop_type == "template") {
        crop_sz[0] = crop_sz[1] = 128.0f; // Example size, replace with actual template size
    } else if (crop_type == "search") {
        crop_sz[0] = crop_sz[1] = 256.0f; // Example size, replace with actual search size
    } else {
        throw std::invalid_argument("Unsupported crop type");
    }

    // Convert input boxes
    std::vector<float> local_box_in = box_in;
    std::vector<float> local_box_extract;

    if (box_extract.empty()) {
        local_box_extract = local_box_in;
    } else {
        local_box_extract = box_extract;
    }

    // Transform the bounding box coordinates
    std::vector<float> template_bbox = transform_image_to_crop(local_box_in, local_box_extract, resize_factor, crop_sz, true);

    // Reshape output to [1, 1, 4]
    return template_bbox;
}


std::vector<float> Utils::map_box_back(const std::vector<float>& pred_box, float resize_factor, std::vector<float>& state)
{
    // Extract state variables
    float cx_prev = state[0] + 0.5f * state[2];
    float cy_prev = state[1] + 0.5f * state[3];
    
    // Extract predicted box variables
    float cx = pred_box[0];
    float cy = pred_box[1];
    float w = pred_box[2];
    float h = pred_box[3];
    
    // Compute half_side
    float half_side = 0.5f * 256.0f / resize_factor;
    
    // Compute real coordinates
    float cx_real = cx + (cx_prev - half_side);
    float cy_real = cy + (cy_prev - half_side);
    
    // Compute and return the result box
    std::vector<float> result_box = {
        cx_real - 0.5f * w,
        cy_real - 0.5f * h,
        w,
        h
    };
    
    return result_box;
}


void Utils::clip_box(const std::vector<int>& box, int H, int W, std::vector<float>&state,int margin)
{
    // Extract bounding box coordinates
    int x1 = box[0];
    int y1 = box[1];
    int w = box[2];
    int h = box[3];
    
    // Calculate the bottom-right corner of the box
    int x2 = x1 + w;
    int y2 = y1 + h;

    // Clip the coordinates to be within the margins and image boundaries
    x1 = std::min(std::max(0, x1), W - margin);
    x2 = std::min(std::max(margin, x2), W);
    y1 = std::min(std::max(0, y1), H - margin);
    y2 = std::min(std::max(margin, y2), H);
    
    // Recalculate width and height
    w = std::max(margin, x2 - x1);
    h = std::max(margin, y2 - y1);

    // Return the clipped bounding box as a vector
    state= {x1, y1, w, h};
}


std::vector<std::vector<float>> Utils::box_xywh_to_xyxy(const std::vector<std::vector<float>>& boxes_xywh)
{
    std::vector<std::vector<float>> boxes_xyxy;
    
    for (const auto& box : boxes_xywh) {
        float x1 = box[0];
        float y1 = box[1];
        float w = box[2];
        float h = box[3];
        
        std::vector<float> converted_box = {x1, y1, x1 + w, y1 + h};
        boxes_xyxy.push_back(converted_box);
    }
    
    return boxes_xyxy;
}


cv::cuda::GpuMat Utils::sample_target(
    const cv::cuda::GpuMat& im, 
    const std::vector<float>& target_bb, 
    float search_area_factor, 
    int output_sz, 
    const cv::cuda::GpuMat& mask,
    float* resize_factor)
{
    // Extract bounding box values
    float x = target_bb[0];
    float y = target_bb[1];
    float w = target_bb[2];
    float h = target_bb[3];

    // Calculate the size of the crop
    int crop_sz = static_cast<int>(std::ceil(std::sqrt(w * h) * search_area_factor));

    if (crop_sz < 1) {
        throw std::runtime_error("Too small bounding box.");
    }

    // Determine crop coordinates
    int x1 = static_cast<int>(std::round(x + 0.5 * w - 0.5 * crop_sz));
    int x2 = x1 + crop_sz;
    int y1 = static_cast<int>(std::round(y + 0.5 * h - 0.5 * crop_sz));
    int y2 = y1 + crop_sz;

    // Handle padding if the crop goes outside the image boundaries
    int x1_pad = std::max(0, -x1);
    int x2_pad = std::max(x2 - im.cols + 1, 0);
    int y1_pad = std::max(0, -y1);
    int y2_pad = std::max(y2 - im.rows + 1, 0);

    // Crop the image
    cv::Rect crop_rect(x1 + x1_pad, y1 + y1_pad, crop_sz - x1_pad - x2_pad, crop_sz - y1_pad - y2_pad);
    cv::cuda::GpuMat im_crop;
    im(crop_rect).copyTo(im_crop);

    // Pad the cropped image
    cv::cuda::GpuMat im_crop_padded;
    cv::cuda::copyMakeBorder(im_crop, im_crop_padded, y1_pad, y2_pad, x1_pad, x2_pad, cv::BORDER_CONSTANT);

    // Handle the attention mask
    cv::cuda::GpuMat att_mask(im_crop_padded.size(), CV_8U, cv::Scalar(255));
    if (x2_pad > 0 || y2_pad > 0) {
        att_mask(cv::Rect(x1_pad, y1_pad, crop_sz - x1_pad - x2_pad, crop_sz - y1_pad - y2_pad)).setTo(cv::Scalar(0));
    }

    if (!mask.empty()) {
        // Crop and pad the mask
        cv::cuda::GpuMat mask_crop = mask(crop_rect);
        cv::cuda::GpuMat mask_crop_padded;
        cv::cuda::copyMakeBorder(mask_crop, mask_crop_padded, y1_pad, y2_pad, x1_pad, x2_pad, cv::BORDER_CONSTANT);

        // Resize the cropped image and mask if output size is provided
        if (output_sz > 0) {
            *resize_factor = static_cast<float>(output_sz) / crop_sz;
            cv::cuda::GpuMat im_resized, mask_resized;
            cv::cuda::resize(im_crop_padded, im_resized, cv::Size(output_sz, output_sz));
            cv::cuda::resize(att_mask, att_mask, cv::Size(output_sz, output_sz), 0, 0, cv::INTER_LINEAR);

            cv::cuda::resize(mask_crop_padded, mask_resized, cv::Size(output_sz, output_sz), 0, 0, cv::INTER_LINEAR);
            mask_crop_padded = mask_resized;

            return im_resized;  // Returning the resized image
        } else {
            return im_crop_padded;  // Returning the padded cropped image
        }
    } else {
        // Resize the cropped image if output size is provided
        if (output_sz > 0) {
            *resize_factor = static_cast<float>(output_sz) / crop_sz;
            cv::cuda::GpuMat im_resized;
            cv::cuda::resize(im_crop_padded, im_resized, cv::Size(output_sz, output_sz));
            cv::cuda::resize(att_mask, att_mask, cv::Size(output_sz, output_sz), 0, 0, cv::INTER_LINEAR);
            return im_resized;  // Returning the resized image
        } else {
            return im_crop_padded;  // Returning the padded cropped image
        }
    }
}

void Utils::cal_bbox(const std::vector<float>& score_map_ctr, int feat_size,
            const std::vector<float>& size_map, const std::vector<float>& offset_map,
            std::vector<float>& bbox, bool return_score)
{
    int map_height = 16;
    int map_width = 16;

    // Calculate the maximum scores and their indices
    max_score = -std::numeric_limits<float>::infinity();
    int max_idx = 0;

    for (int i = 0; i < map_height * map_width; ++i) {
        if (score_map_ctr[i] > max_score) {
            max_score = score_map_ctr[i];
            max_idx = i;
        }
    }

    int idx_y = max_idx / feat_size;
    int idx_x = max_idx % feat_size;

    std::cout << "idy: " << max_idx<<",  max score:  "<<max_score << std::endl;

    // Get the sizes and offsets
    int size_idx = max_idx;
// .....................................................
    float size_width = size_map[size_idx];
    float size_height = size_map[size_idx + map_height*map_width];
    float offset_x = offset_map[size_idx];
    float offset_y = offset_map[size_idx + map_height*map_width];
// .....................................................
    // Calculate bounding boxes
    float x_center = (idx_x + offset_x) / feat_size;
    float y_center = (idx_y + offset_y) / feat_size;
    float width_bb = size_width;
    float height_bb = size_height;

    bbox = {x_center, y_center, width_bb, height_bb};

    if (return_score) {
        std::cout << "Max Score: " << max_idx << std::endl;
    }
}

cv::cuda::GpuMat Utils::process(const cv::cuda::GpuMat& img_arr)
{
    // Define the mean and std (same as in the Python code)
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);
    // hwc_to_chw(img_arr,img_float)
    cv::cuda::GpuMat img_float,img_normalized;
    img_arr.convertTo(img_float, CV_32F, 1.0 / 255.0);

    // Step 1: Convert the image from [0, 255] range to [0, 1] (normalize by 255.0)

    // Step 2: Normalize the image by subtracting the mean and dividing by std
    cv::cuda::subtract(img_float, mean, img_normalized);  // Subtract mean
    cv::cuda::divide(img_normalized, std, img_normalized);  // Divide by std
    // Return the normalized and permuted image
    return img_normalized;
}
