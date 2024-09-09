// #include "../detector/inference.h"
#include <OCSort.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <thread>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

std::mutex dataMutex;

/**
@brief Convert Vector to Matrix
@param data
@return Eigen::Matrix<float, Eigen::Dynamic, 6>
*/
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

void processFrame(const cv::Mat& frame, std::vector<float>& output, ocsort::OCSort& tracker) {
    std::vector<std::vector<float>> data;
    cv::Rect box;

    data.push_back(output);

    if (!data.empty()) {
        std::vector<Eigen::RowVectorXf> res = tracker.update(Vector2Matrix(data));

        cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);

        std::string classString = '(' + std::to_string(output[4]).substr(0, 4) + ')';
        cv::putText(frame, classString, cv::Point(box.x + 5, box.y + box.height - 10), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0), 1, 0);

        for (auto j : res) {
            int ID = int(j[4]);
            int Class = int(j[5]);
            float conf = j[6];
            cv::putText(frame, cv::format("ID:%d", ID), cv::Point(j[0], j[1] - 5), 0, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            cv::rectangle(frame, cv::Rect(j[0], j[1], j[2] - j[0] + 1, j[3] - j[1] + 1), cv::Scalar(0, 0, 255), 1);
        }

        data.clear();
    }
}


void videoODThread() {
    bool runGPU = true;

    std::vector<std::string> classes;

    ocsort::OCSort tracker = ocsort::OCSort(0, 50, 1, 0.22136877277096445, 1, "giou", 0.3941737016672115, true);

    std::vector<float> output={438.0000,443.0000,536.0,638.0,0.5,1};

    cv::VideoCapture capture("/home/rtr/cuong/tracking/LiteTrack/test.mp4");

    bool isRunning = true;  // Variable for controlling the loop

    while (isRunning) {
        cv::Mat frame;
        std::cout << "readinggggggg,.....\n";
        if (!capture.read(frame)) {
            std::cout << "\n Cannot read the video file. Please check your video.\n";
            isRunning = false;
            break;
        }

        // Perform inference on the frame
        // output = detector.runInference(frame);

        // Acquire lock before accessing shared data
        std::lock_guard<std::mutex> lock(dataMutex);

        // Process the frame and shared data
        processFrame(frame, output, tracker);

        // Display the frame
        // cv::imshow("VideoOD", frame);

        // Check for termination condition (e.g., press Esc key)
        if (cv::waitKey(1) == 27)
            isRunning = false;
    }
}

int main() {

    std::thread videoThread(videoODThread);

    // Wait for the video thread to finish
    videoThread.join();

    std::cout << "End of program." << std::endl;
    return 0;
}
