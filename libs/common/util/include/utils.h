#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>  // For std::min and std::max
#include <limits>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>
class Utils {
public:
    Utils();
    
    std::vector<float> transform_image_to_crop(const std::vector<float>& box_in, 
                                            const std::vector<float>& box_extract, 
                                            float resize_factor, 
                                            const std::vector<float>& crop_sz, 
                                            bool normalize = false); 
    
    std::vector<float> transform_bbox_to_crop(const std::vector<float>& box_in, float resize_factor, 
                                            const std::string& crop_type, const std::vector<float>& box_extract = {});

    std::vector<float> map_box_back(const std::vector<float>& pred_box, float resize_factor, std::vector<float>& state);
    
    void clip_box(const std::vector<int>& box, int H, int W, std::vector<float>&state,int margin = 0);
    
    std::vector<std::vector<float>> box_xywh_to_xyxy(const std::vector<std::vector<float>>& boxes_xywh);

    cv::cuda::GpuMat sample_target(
        const cv::cuda::GpuMat& im, 
        const std::vector<float>& target_bb, 
        float search_area_factor, 
        int output_sz = -1, 
        const cv::cuda::GpuMat& mask = cv::cuda::GpuMat(),
        float* resize_factor = nullptr);

    void cal_bbox(const std::vector<float>& score_map_ctr, int feat_size,
                const std::vector<float>& size_map, const std::vector<float>& offset_map,
                std::vector<float>& bbox, bool return_score = true);

    cv::cuda::GpuMat process(const cv::cuda::GpuMat& img_arr);
    float max_score;
    };