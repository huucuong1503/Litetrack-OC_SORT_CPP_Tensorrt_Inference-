#ifndef FORWARDZ_H
#define FORWARDZ_H

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>

class Logger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;
};

class TensorRTInference {
public:
    TensorRTInference();
    bool loadModelForZ(const std::string& engineFilePathZ);
    bool loadDataForZ(const float* input0, const cv::cuda::GpuMat& input1);
    void inferenceForZ(int avg_time_set = 1);

    bool loadModelForward(const std::string& engineFilePathForWard);
    bool loadDataForward(const cv::cuda::GpuMat& input1);
    void inferenceForward(int avg_time_set = 1);
    
    void freeResources();
    ~TensorRTInference();
    float size_map[1 * 2 * 16 * 16] = {0};
    float score_map[1 * 1 * 16 * 16] = {0};
    float offset_map[1 * 2 * 16 * 16] = {0};


private:
    std::string engineFilePathZ;
    std::vector<char> engineDataZ;
    std::unique_ptr<Logger> logger;
    nvinfer1::IRuntime* runtimeZ = nullptr;
    nvinfer1::ICudaEngine* engineZ = nullptr;
    nvinfer1::IExecutionContext* contextZ = nullptr;

    std::string engineFilePath;
    std::vector<char> engineData;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;

    const int batchSize = 1;
    void* buffers[3] = {nullptr};
    void* buffersFor[5] = {nullptr};

    cv::Mat convert_search;
    cv::Mat blob; 

    int inputIndex0;
    int inputIndex1;
    int outputIndex1;
    int outputIndex2;
    int outputIndex3;
    int input_trans_out;
};

#endif // FORWARDZ_H
