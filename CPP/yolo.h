#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#define BBOX_CONF_THRESH 0.5
#define BATCH_SIZE 1
#define BBOX_NUM 1
#define BOXES_SIZE 400
#define CLASS_SCORES 100
#define CLASS_INDEXS 100
#define IMG_SIZE 640
#define IMG_CHANNELS 3

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class Yolo {
public:
    Yolo(const std::string& model_path);
    ~Yolo();
    void Infer(const std::vector<cv::Mat>& images, std::vector<std::vector<Object>>& all_detections);
    void draw_objects(const std::vector<cv::Mat>& images, const std::vector<std::vector<Object>>& all_detections);

private:
    void decodeOutputs(std::vector<Object>& objects, int bbox_num, float* Boxes, int* ClassIndexs, float* ClassScores);

    void* buffs[5];
    int* num_dets;
    float* det_boxes;
    float* det_scores;
    int* det_classes; // 改为 int*

    nvinfer1::ICudaEngine* engine;
    nvinfer1::IRuntime* runtime;
    nvinfer1::IExecutionContext* context;
    cudaStream_t stream;
};
