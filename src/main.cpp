#include <vector>
#include <opencv2/opencv.hpp>
#include <forward.h>
#include <utils.h>
#include <chrono>
#include <OCSort.hpp>
#include <Eigen/Dense>
#include <string>
// #include <thread>
// #include <fstream>
// #include <iostream>



cv::cuda::GpuMat search;
std::vector<float> pred_box;
std::vector<float> mapbox;
TensorRTInference inference;
Utils utils_func;
ocsort::OCSort tracker = ocsort::OCSort(0.45, 50, 3, 0.22136877277096445, 1, "giou", 0.2, false);

// auto write_vector_to_file = [](const std::vector<float>& vec, const std::string& filename) {
//         std::ofstream file(filename);
//         if (!file.is_open()) {
//             std::cerr << "Failed to open file " << filename << std::endl;
//             return;
//         }
//         for (const auto& value : vec) {
//             file << value << " ";
//         }
//         file.close();
//     };

Eigen::Matrix<float, Eigen::Dynamic, 6> Vector2Matrix(std::vector<std::vector<float>> data) {
    Eigen::Matrix<float, Eigen::Dynamic, 6> matrix(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < data[0].size(); ++j) {
            matrix(i, j) = data[i][j];
        }
    }
    return matrix;
}

template<typename AnyCls>
std::ostream& operator<<(std::ostream& os, const std::vector<AnyCls>& v) {
    os << "{";
    for (auto it = v.begin(); it != v.end(); ++it) {
        os << "(" << *it << ")";
        if (it != v.end() - 1) os << ", ";
    }
    os << "}";
    return os;
}

void processFrame(const cv::Mat& frame, std::vector<float>& output, ocsort::OCSort& tracker,std::vector<float>& state) {
    std::vector<float> row;

    row.push_back(output[0]);
    row.push_back(output[1]);
    row.push_back(output[0] + output[2]);
    row.push_back(output[1] + output[3]);
    row.push_back(output[4]);
    row.push_back(output[5]);

    std::vector<std::vector<float>> data;
    // cv::Rect box;
    cv::Rect box((int)output[0], (int)output[1], (int)output[2], (int)output[3]); // Example rectangle (x, y, width, height)

    data.push_back(row);

    if (!data.empty()) {
        std::vector<Eigen::RowVectorXf> res = tracker.update(Vector2Matrix(data));

        cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
        if (utils_func.max_score<0.35){
            state[0]=(float)tracker.posX;
            state[1]=(float)tracker.posY;
            state[2]=(float)tracker.posW;
            state[3]=(float)tracker.posH;
        }
        // std::string classString = '(' + std::to_string(output[4]).substr(0, 4) + ')';
        // cv::putText(frame, classString, cv::Point(box.x + 5, box.y + box.height - 10), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0), 1, 0);
        
        std::cout<<"J newX: "<<tracker.posX<<"  J newY: "<<tracker.posY<<"  J newW: "<<tracker.posW<<"  J newH: "<<tracker.posH<<std::endl;
        // cv::putText(frame, cv::format("ID:%d", ID), cv::Point(j[0], j[1] - 5), 0, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        cv::rectangle(frame, cv::Rect(tracker.posX, tracker.posY, tracker.posW, tracker.posH), cv::Scalar(0, 0, 255), 1);

        // for (auto j : res) {
        //     float conf = j[6];
        //     std::cout<<"J newX: "<<tracker.posX<<"  J newY: "<<tracker.posY<<"  J newW: "<<tracker.posW<<"  J newH: "<<tracker.posH<<std::endl;
        //     // cv::putText(frame, cv::format("ID:%d", ID), cv::Point(j[0], j[1] - 5), 0, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        //     cv::rectangle(frame, cv::Rect(tracker.posX, tracker.posY, tracker.posW, tracker.posH), cv::Scalar(0, 0, 255), 1);
        // }

        data.clear();
    }
}


void initialize(const cv::cuda::GpuMat& image, const std::vector<float>& info_bbox) {
    // Parameters for template_factor and template_size
    float template_factor = 2.0f; // Replace with actual value
    int template_size = 128;      // Replace with actual value

    // Sample the target
    float resize_factor = 1.0f;
    cv::cuda::GpuMat z_patch_arr, z_amask_arr;
    z_patch_arr = utils_func.sample_target(image, info_bbox, template_factor, template_size, cv::cuda::GpuMat(), &resize_factor);
    // std::cout<<"resize factor initalize:   "<<resize_factor<<std::endl;
    // Process the template
    cv::cuda::GpuMat template_mat = utils_func.process(z_patch_arr);

    // Initialize other variables
    cv::cuda::GpuMat box_mask_z; // This remains uninitialized

    // Transform bounding box to crop
    std::vector<float> template_bbox = utils_func.transform_bbox_to_crop(info_bbox, resize_factor, "template", {});
    std::vector<std::vector<float>> bbox_xywh = {template_bbox};
    std::vector<std::vector<float>> bbox_xyxy = utils_func.box_xywh_to_xyxy(bbox_xywh);
    // std::cout <<"x_center: "<<bbox_xyxy[0][0]<<"  -  y_center:  "<< bbox_xyxy[0][1]<<"  -  width_center:  "<<bbox_xyxy[0][2]<<"  -  height_center:  "<< bbox_xyxy[0][3]<<"  resize:  "<<resize_factor<<std::endl;

    if (!inference.loadDataForZ(bbox_xyxy[0].data(),template_mat)) {
        std::cout<<"cannot load loadData"<<std::endl;
    }

    // std::cout<<"done load data"<<std::endl;
    inference.inferenceForZ();
    // std::cout<<"succesfully load model"<<std::endl;


}
void track_center(const cv::cuda::GpuMat& search_image,std::vector<float>& state) {
    // Parameters for template_factor and template_size
        // Get the size of the GpuMat
    int W = search_image.cols;    // Width of the image
    int H = search_image.rows;   // Height of the image
    float search_factor = 4.0f; // Replace with actual value
    int search_size = 256;      // Replace with actual value

    // Sample the target
    float resize_factor = 1.0f;
    cv::cuda::GpuMat x_patch_arr;
    
    x_patch_arr = utils_func.sample_target(search_image, state, search_factor, search_size, cv::cuda::GpuMat(), &resize_factor);
    // std::cout<<"search size factor:  "<<resize_factor<<"   width  "<<W<<"  height  "<<H<<std::endl;

    search=utils_func.process(x_patch_arr);
    if (!inference.loadDataForward(search)) {
        std::cout<<"cannot load loadData"<<std::endl;
    }

    // std::cout<<"done load data"<<std::endl;
    inference.inferenceForward();
    // // Calculate the size of the array

    // // Convert the array to a vector
    std::vector<float> score_map(inference.score_map, inference.score_map + 256);
    std::vector<float> size_map(inference.size_map, inference.size_map + 512);
    std::vector<float> offset_map(inference.offset_map, inference.offset_map + 512);

    
    utils_func.cal_bbox(score_map, 16, size_map, offset_map, pred_box, false);
    // std::vector<float> pred_box = {0.4958, 0.4957, 0.1819, 0.3510};
// 0.4958, 0.4957, 0.1819, 0.3510
    std::transform(pred_box.begin(), pred_box.end(), pred_box.begin(), [&search_size,&resize_factor](auto& c){return (c*search_size)/resize_factor;});
    mapbox=utils_func.map_box_back(pred_box,resize_factor,state);
    // std::cout <<"mapboxX: "<<mapbox[0]<<"  -  mapboxY:  "<< mapbox[1]<<"  -  W:  "<<mapbox[2]<<"  -  H:  "<< mapbox[3]<<std::endl;
    
    std::vector<int> intVec(mapbox.size());
    
    std::transform(mapbox.begin(), mapbox.end(), intVec.begin(),[](float val) { return static_cast<int>(val); });
    utils_func.clip_box(intVec,H,W,state,10);
    std::cout <<"stateX: "<<state[0]<<"  -  stateY:  "<< state[1]<<"  -  W:  "<<state[2]<<"  -  H:  "<< state[3]<<std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file><bouding box>" << std::endl;
        return 1;
    }
    std::string inputVideoPath = argv[1];
    std::string outputVideoPath = argv[2];
    int x = std::stoi(argv[3]);
    int y = std::stoi(argv[4]);
    int width = std::stoi(argv[5]);
    int height = std::stoi(argv[6]);
    //....................................... Example usage sample target.......................................................
    if (!inference.loadModelForZ("/home/rtr/cuong/LiteTrackInferenceCPP/myCPP/models/B8_model/b8_op15_Z.trt")) {
        std::cout<<"cannot load model"<<std::endl;
    }
    if (!inference.loadModelForward("/home/rtr/cuong/LiteTrackInferenceCPP/myCPP/models/B8_model/b8_op15_hann.trt")) {
        std::cout<<"cannot load model"<<std::endl;
    }
    cv::VideoCapture capture(inputVideoPath);    
    // // Define bounding box and parameters
    // std::vector<float> target_bb = {438, 443, 98, 195};  // Bounding box [x, y, w, h]
    std::vector<float> target_bb = {(float)x, (float)y, (float)width, (float)height};  // Bounding box [x, y, w, h]
    float resize_factor = 1.0f;
    std::vector<float> state=target_bb;

    // auto start = std::chrono::high_resolution_clock::now();
    
    cv::cuda::GpuMat d_img;
    bool isRunning = true;  // Variable for controlling the loop
    bool init=false;

    int frameWidth = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = capture.get(cv::CAP_PROP_FPS);
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G'); // Codec used to save the video

    cv::VideoWriter writer(outputVideoPath, codec, fps, cv::Size(frameWidth, frameHeight));

    while (isRunning) {
        cv::Mat frame;
        if (!capture.read(frame)) {
            std::cout << "\n Cannot read the video file. Please check your video.\n";
            isRunning = false;
            break;
        }
        if (init==false){
            init=true;
            d_img.upload(frame);
            initialize(d_img,target_bb);
        }
        d_img.upload(frame);
        track_center(d_img,state);
        
        std::vector<float> output=state;

        std::vector<float> scoreID = {utils_func.max_score, 1};

        output.insert(output.end(), scoreID.begin(), scoreID.end());
        
        processFrame(frame, output, tracker,state);
        // cv::Rect rect((int)state[0], (int)state[1], (int)state[2], (int)state[3]); // Example rectangle (x, y, width, height)
        
        // cv::rectangle(frame, rect, (255, 0, 0), 2);
        writer.write(frame);
        // Check for termination condition (e.g., press Esc key)
        if (cv::waitKey(1) == 27)
            isRunning = false;
    }
    writer.release();
    capture.release();
    // auto end = std::chrono::high_resolution_clock::now();

        // Calculate the duration in microseconds
    // std::chrono::duration<double, std::micro> duration = end - start;

    // Output the time taken
    // std::cout << "Time taken: " << duration.count()/100 << " microseconds" << std::endl;

}