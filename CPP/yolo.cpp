#include "yolo.h"
#include "warp_affine.h"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntimeCommon.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cassert>

namespace {
struct Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) std::cout << msg << std::endl;
    }
} gLogger;

const float color_list[80][3] = {
    {0.000, 0.447, 0.741}, {0.850, 0.325, 0.098}, {0.929, 0.694, 0.125}, {0.494, 0.184, 0.556},
    {0.466, 0.674, 0.188}, {0.301, 0.745, 0.933}, {0.635, 0.078, 0.184}, {0.300, 0.300, 0.300},
    {0.600, 0.600, 0.600}, {1.000, 0.000, 0.000}, {1.000, 0.500, 0.000}, {0.749, 0.749, 0.000},
    {0.000, 1.000, 0.000}, {0.000, 0.000, 1.000}, {0.667, 0.000, 1.000}, {0.333, 0.333, 0.000},
    {0.333, 0.667, 0.000}, {0.333, 1.000, 0.000}, {0.667, 0.333, 0.000}, {0.667, 0.667, 0.000},
    {0.667, 1.000, 0.000}, {1.000, 0.333, 0.000}, {1.000, 0.667, 0.000}, {1.000, 1.000, 0.000},
    {0.000, 0.333, 0.500}, {0.000, 0.667, 0.500}, {0.000, 1.000, 0.500}, {0.333, 0.000, 0.500},
    {0.333, 0.333, 0.500}, {0.333, 0.667, 0.500}, {0.333, 1.000, 0.500}, {0.667, 0.000, 0.500},
    {0.667, 0.333, 0.500}, {0.667, 0.667, 0.500}, {0.667, 1.000, 0.500}, {1.000, 0.000, 0.500},
    {1.000, 0.333, 0.500}, {1.000, 0.667, 0.500}, {1.000, 1.000, 0.500}, {0.000, 0.333, 1.000},
    {0.000, 0.667, 1.000}, {0.000, 1.000, 1.000}, {0.333, 0.000, 1.000}, {0.333, 0.333, 1.000},
    {0.333, 0.667, 1.000}, {0.333, 1.000, 1.000}, {0.667, 0.000, 1.000}, {0.667, 0.333, 1.000},
    {0.667, 0.667, 1.000}, {0.667, 1.000, 1.000}, {1.000, 0.000, 1.000}, {1.000, 0.333, 1.000},
    {1.000, 0.667, 1.000}, {0.333, 0.000, 0.000}, {0.500, 0.000, 0.000}, {0.667, 0.000, 0.000},
    {0.833, 0.000, 0.000}, {1.000, 0.000, 0.000}, {0.000, 0.167, 0.000}, {0.000, 0.333, 0.000},
    {0.000, 0.500, 0.000}, {0.000, 0.667, 0.000}, {0.000, 0.833, 0.000}, {0.000, 1.000, 0.000},
    {0.000, 0.000, 0.167}, {0.000, 0.000, 0.333}, {0.000, 0.000, 0.500}, {0.000, 0.000, 0.667},
    {0.000, 0.000, 0.833}, {0.000, 0.000, 1.000}, {0.000, 0.000, 0.000}, {0.143, 0.143, 0.143},
    {0.286, 0.286, 0.286}, {0.429, 0.429, 0.429}, {0.571, 0.571, 0.571}, {0.714, 0.714, 0.714},
    {0.857, 0.857, 0.857}, {0.000, 0.447, 0.741}, {0.314, 0.717, 0.741}, {0.50, 0.5, 0}
};
} // namespace

Yolo::Yolo(const std::string& model_path) {
    num_dets = new int[BATCH_SIZE * BBOX_NUM];
    det_boxes = new float[BATCH_SIZE * BOXES_SIZE];
    det_scores = new float[BATCH_SIZE * CLASS_SCORES];
    det_classes = new int[BATCH_SIZE * CLASS_INDEXS];

    std::ifstream ifile(model_path, std::ios::in | std::ios::binary);
    if (!ifile) {
        std::cerr << "read serialized file failed\n";
        std::abort();
    }
    ifile.seekg(0, std::ios::end);
    const int mdsize = ifile.tellg();
    ifile.clear();
    ifile.seekg(0, std::ios::beg);
    std::vector<char> buf(mdsize);
    ifile.read(&buf[0], mdsize);
    ifile.close();

    runtime = nvinfer1::createInferRuntime(gLogger);
    initLibNvInferPlugins(&gLogger, "");
    engine = runtime->deserializeCudaEngine((void*)&buf[0], mdsize);
    context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "create execution context failed\n";
        std::abort();
    }

    cudaError_t state;
    state = cudaMalloc(&buffs[0], BATCH_SIZE * IMG_CHANNELS * IMG_SIZE * IMG_SIZE * sizeof(float));
    if (state) std::abort();
    state = cudaMalloc(&buffs[1], BATCH_SIZE * BBOX_NUM * sizeof(int));
    if (state) std::abort();
    state = cudaMalloc(&buffs[2], BATCH_SIZE * BOXES_SIZE * sizeof(float));
    if (state) std::abort();
    state = cudaMalloc(&buffs[3], BATCH_SIZE * CLASS_SCORES * sizeof(float));
    if (state) std::abort();
    state = cudaMalloc(&buffs[4], BATCH_SIZE * CLASS_INDEXS * sizeof(int));
    if (state) std::abort();
    state = cudaStreamCreate(&stream);
    if (state) std::abort();
}

Yolo::~Yolo() {
    cudaStreamSynchronize(stream);
    cudaFree(buffs[0]);
    cudaFree(buffs[1]);
    cudaFree(buffs[2]);
    cudaFree(buffs[3]);
    cudaFree(buffs[4]);
    cudaStreamDestroy(stream);
    if (context) context->destroy();
    if (engine) engine->destroy();
    if (runtime) runtime->destroy();
    delete[] num_dets;
    delete[] det_boxes;
    delete[] det_scores;
    delete[] det_classes;
}

void Yolo::Infer(const std::vector<cv::Mat>& images, std::vector<std::vector<Object>>& all_detections) {
    cudaError_t state;
    float scale = 1.0f;
    int img_w = images[0].cols;
    int img_h = images[0].rows;

    uint8_t* psrc_device = nullptr;
    size_t src_size = img_w * img_h * 3 * BATCH_SIZE;
    state = cudaMalloc((void**)&psrc_device, src_size);
    if (state) std::abort();

    for (int image_index = 0; image_index < BATCH_SIZE; ++image_index) {
        cv::Mat image = images[image_index].clone();
        size_t offset = img_w * img_h * 3 * image_index;
        state = cudaMemcpyAsync(psrc_device + offset, image.data, src_size / BATCH_SIZE, cudaMemcpyHostToDevice, stream);
        if (state) std::abort();
        scale = warp_affine_bilinear(
            psrc_device + offset, img_w * 3, img_w, img_h,
            (float*)buffs[0] + image_index * IMG_CHANNELS * IMG_SIZE * IMG_SIZE, IMG_SIZE * 3, IMG_SIZE, IMG_SIZE,
            114, stream);
    }
    cudaDeviceSynchronize();
    cudaFree(psrc_device);
    context->enqueueV2(buffs, stream, nullptr);

    state = cudaMemcpyAsync(num_dets, buffs[1], BATCH_SIZE * BBOX_NUM * sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (state) std::abort();
    state = cudaMemcpyAsync(det_boxes, buffs[2], BATCH_SIZE * BOXES_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (state) std::abort();
    state = cudaMemcpyAsync(det_scores, buffs[3], BATCH_SIZE * CLASS_SCORES * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (state) std::abort();
    state = cudaMemcpyAsync(det_classes, buffs[4], BATCH_SIZE * CLASS_INDEXS * sizeof(int), cudaMemcpyDeviceToHost, stream); // <-- sizeof(int)
    if (state) std::abort();
    cudaStreamSynchronize(stream);

    int x_offset = (IMG_SIZE / scale - img_w) / 2;
    int y_offset = (IMG_SIZE / scale - img_h) / 2;

    all_detections.resize(BATCH_SIZE);
    for (int batch_num = 0; batch_num < BATCH_SIZE; ++batch_num) {
        int detecotr_count = 0;
        std::vector<float> Boxes(BOXES_SIZE);
        std::vector<int> ClassIndexs(CLASS_INDEXS);  
        std::vector<float> ClassScores(CLASS_SCORES);

        for (size_t i = 0; i < num_dets[batch_num * BBOX_NUM]; ++i) {
            if (det_scores[batch_num * CLASS_SCORES + i] > BBOX_CONF_THRESH) {
                float x0 = (det_boxes[(batch_num * BOXES_SIZE) + i * 4]) / scale - x_offset;
                float y0 = (det_boxes[(batch_num * BOXES_SIZE) + i * 4 + 1]) / scale - y_offset;
                float x1 = (det_boxes[(batch_num * BOXES_SIZE) + i * 4 + 2]) / scale - x_offset;
                float y1 = (det_boxes[(batch_num * BOXES_SIZE) + i * 4 + 3]) / scale - y_offset;

                x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
                y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
                x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
                y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

                Boxes[detecotr_count * 4] = x0;
                Boxes[detecotr_count * 4 + 1] = y0;
                Boxes[detecotr_count * 4 + 2] = x1 - x0;
                Boxes[detecotr_count * 4 + 3] = y1 - y0;
                ClassIndexs[detecotr_count] = det_classes[batch_num * CLASS_INDEXS + i];
                ClassScores[detecotr_count] = det_scores[batch_num * CLASS_SCORES + i];
                detecotr_count++;
            }
        }
        std::vector<Object> objects(detecotr_count);
        decodeOutputs(objects, detecotr_count, Boxes.data(), ClassIndexs.data(), ClassScores.data());
        all_detections[batch_num] = objects;
    }
}

void Yolo::decodeOutputs(std::vector<Object>& objects, int bbox_num, float* Boxes, int* ClassIndexs, float* ClassScores) {
    for (int i = 0; i < bbox_num; i++) {
        objects[i].label = ClassIndexs[i];
        objects[i].prob = ClassScores[i];
        objects[i].rect.x = Boxes[i * 4 + 0];
        objects[i].rect.y = Boxes[i * 4 + 1];
        objects[i].rect.width = Boxes[i * 4 + 2];
        objects[i].rect.height = Boxes[i * 4 + 3];
    }
}

void Yolo::draw_objects(const std::vector<cv::Mat>& images, const std::vector<std::vector<Object>>& all_detections) {
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };
    for (size_t i = 0; i < images.size(); i++) {
        cv::Mat img = images[i].clone();
        for (const auto& obj : all_detections[i]) {
            int class_idx = obj.label;
            cv::Scalar color(color_list[class_idx][0] * 255, color_list[class_idx][1] * 255, color_list[class_idx][2] * 255);
            float c_mean = cv::mean(color)[0];
            cv::Scalar txt_color = (c_mean > 127) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

            cv::rectangle(img, obj.rect, color, 2);

            char text[256];
            sprintf(text, "%s %.1f%%", class_names[class_idx], obj.prob * 100);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

            cv::Scalar txt_bk_color = color * 0.7;
            int x = obj.rect.x;
            int y = obj.rect.y + 1;
            if (y > img.rows) y = img.rows;

            cv::rectangle(img, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                          txt_bk_color, -1);

            cv::putText(img, text, cv::Point(x, y + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
        }
        cv::imwrite("./result_" + std::to_string(i) + ".jpg", img);
    }
}
