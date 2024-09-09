#include "forward.h"
#include <iostream>
#include <fstream>
#include <chrono>
void Logger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept {
    if (severity != nvinfer1::ILogger::Severity::kINFO) {
        std::cout << msg << std::endl;
    }
}

TensorRTInference::TensorRTInference(){
    logger = std::make_unique<Logger>();
}

bool TensorRTInference::loadModelForZ(const std::string& engineFilePathZ){
    std::ifstream engineFile(engineFilePathZ, std::ios::binary);
    if (!engineFile) {
        std::cerr << "Error loading engine file!" << std::endl;
        return false;
    }

    engineFile.seekg(0, engineFile.end);
    size_t engineSize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);
    engineDataZ.resize(engineSize);
    engineFile.read(engineDataZ.data(), engineSize);
    engineFile.close();

    runtimeZ = nvinfer1::createInferRuntime(*logger);
    if (!runtimeZ) {
        std::cerr << "Error creating TensorRT runtime!" << std::endl;
        return false;
    }

    engineZ = runtimeZ->deserializeCudaEngine(engineDataZ.data(), engineSize, nullptr);
    if (!engineZ) {
        std::cerr << "Error deserializing CUDA engine!" << std::endl;
        return false;
    }

    contextZ = engineZ->createExecutionContext();
    if (!contextZ) {
        std::cerr << "Error creating execution context!" << std::endl;
        return false;
    }

    return true;
}

bool TensorRTInference::loadModelForward(const std::string& engineFilePath){
    // std::cout << "Loading model..................  "<<std::endl;
    
    std::ifstream engineFile(engineFilePath, std::ios::binary);
    if (!engineFile) {
        std::cerr << "Error loading engine file!" << std::endl;
        return false;
    }

    engineFile.seekg(0, engineFile.end);
    size_t engineSize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);
    engineData.resize(engineSize);
    engineFile.read(engineData.data(), engineSize);
    engineFile.close();
    // std::cout << "Done model -> Runtime..................  "<<std::endl;

    runtime = nvinfer1::createInferRuntime(*logger);
    if (!runtime) {
        std::cerr << "Error creating TensorRT runtime!" << std::endl;
        return false;
    }
    // std::cout << "Done runtime -> engine..................  "<<std::endl;

    engine = runtime->deserializeCudaEngine(engineData.data(), engineSize, nullptr);
    if (!engine) {
        std::cerr << "Error deserializing CUDA engine!" << std::endl;
        return false;
    }
    // std::cout << "done engine -> context..................  "<<std::endl;

    context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Error creating execution context!" << std::endl;
        return false;
    }
    // std::cout << "succesfully loading model..................  "<<std::endl;

    return true;
}

bool TensorRTInference::loadDataForZ(const float* input0, const cv::cuda::GpuMat& input1) {
    inputIndex0 = engineZ->getBindingIndex("input.5");
    inputIndex1 = engineZ->getBindingIndex("onnx::Gemm_1");
    outputIndex1 = engineZ->getBindingIndex("1016");
    input_trans_out=engineZ->getBindingIndex("1016");


    cudaMalloc(&buffers[inputIndex0], 1 * 3 * 128 * 128 * sizeof(float));  // Input 1
    cudaMalloc(&buffers[inputIndex1], 1 * 4 * sizeof(float));              // Input 2
    cudaMalloc(&buffers[outputIndex1], 1 * 64 * 768 * sizeof(float));       // Output

    // Copy inputs to the GPU

    cv::Mat convert_template;

    input1.download(convert_template);
    cv::Mat blob = cv::dnn::blobFromImage(convert_template,1.0,cv::Size(),cv::Scalar(),false);

    cudaMemcpy(buffers[inputIndex0], blob.ptr<float>(),  3 * 128 * 128 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(buffers[inputIndex1], input0,   4 * sizeof(float), cudaMemcpyHostToDevice);
    
    float* floatBuffer = new float[3*128*128]; 
    cudaMemcpy(floatBuffer, buffers[inputIndex0],   3*128*128 * sizeof(float), cudaMemcpyDeviceToHost);
    // std::memcpy(floatBuffer, blob.ptr<float>(), blob.total() * blob.elemSize());
    // for (int i=0;i<10;i++)
    // std::cout << "template buffer : "<<i << floatBuffer[i]  << std::endl;

    return true;
}

bool TensorRTInference::loadDataForward(const cv::cuda::GpuMat& input1) {

    // std::cout << "Loading binding......: "<<std::endl;
    inputIndex0 = engine->getBindingIndex("z");
    inputIndex1 = engine->getBindingIndex("input.1");

    outputIndex1 = engine->getBindingIndex("offset_map");
    outputIndex2 = engine->getBindingIndex("size_map");
    outputIndex3 = engine->getBindingIndex("1118");
    // std::cout << "Done binding......: "<<std::endl;


     cudaMalloc(&buffersFor[inputIndex0], 1 *64 * 768 * sizeof(float));  // Input 1
     cudaMalloc(&buffersFor[inputIndex1], 1 * 3 * 256 * 256 * sizeof(float));     // Input 2

     cudaMalloc(&buffersFor[outputIndex1], 1 * 2 * 16 * 16 * sizeof(float));       // Output
     cudaMalloc(&buffersFor[outputIndex2], 1 * 2 * 16 * 16 * sizeof(float));       // Output
     cudaMalloc(&buffersFor[outputIndex3], 1 * 1 * 16 * 16 * sizeof(float));       // Output

    // Copy inputs to the GPU

    // std::vector<float> output(1 * 64 * 768);
    
    buffersFor[inputIndex0]=buffers[input_trans_out];
    
    input1.download(convert_search);
    cv::Mat blob = cv::dnn::blobFromImage(convert_search, 1.0);
    cudaMemcpy(buffersFor[inputIndex1], blob.ptr<float>(),  3 * 256 * 256 * sizeof(float), cudaMemcpyHostToDevice);
    
    return true;
}

void TensorRTInference::inferenceForZ(int avg_time_set) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    for (int i = 0; i < avg_time_set; ++i) {
        contextZ->executeV2(buffers);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << "Inference Time: " << elapsedTime / avg_time_set << " ms" << std::endl;
    float fps = 1000.0f / (elapsedTime / avg_time_set);
    std::cout << "FPS: " << fps << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void TensorRTInference::inferenceForward(int avg_time_set) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 5; ++i) {
    if (buffersFor[i] == nullptr) {
        std::cerr << "Error: buffers[" << i << "] is not initialized." << std::endl;
    }
}
    
    for (int i = 0; i < avg_time_set; ++i) {
        context->executeV2(buffersFor);
    }

     cudaMemcpy(offset_map, buffersFor[outputIndex1], 512 * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(size_map, buffersFor[outputIndex2], 512 * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(score_map, buffersFor[outputIndex3], 256 * sizeof(float), cudaMemcpyDeviceToHost);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // std::cout << "Inference Time: " << elapsedTime / avg_time_set << " ms" << std::endl;
    float fps = 1000.0f / (elapsedTime / avg_time_set);
    // std::cout << "FPS: " << fps << std::endl;

    // Copy the output from GPU to CPU

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void TensorRTInference::freeResources() {
    for (void* buf : buffers) {
        if (buf) cudaFree(buf);
    }

    if (context) context->destroy();
    if (engine) engine->destroy();
    if (runtime) runtime->destroy();

    if (contextZ) contextZ->destroy();
    if (engineZ) engineZ->destroy();
    if (runtimeZ) runtimeZ->destroy();
}

TensorRTInference::~TensorRTInference() {
    // freeResources();
}

int main() {
}
