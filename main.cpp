#include "yolo.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

int main(int argc, char** argv) {
    if (argc == 5 && std::string(argv[1]) == "-model_path" && std::string(argv[3]) == "-image_path") {
        std::string model_path = argv[2];
        std::string image_path = argv[4];
        Yolo yolo(model_path);

        std::vector<cv::Mat> images;
        std::vector<std::string> image_names;
        cv::glob(image_path, image_names);
        for (int i = 0; i < BATCH_SIZE && i < image_names.size(); i++) {
            images.push_back(cv::imread(image_names[i]));
        }

        std::vector<std::vector<Object>> all_detections;
        yolo.Infer(images, all_detections);
        yolo.draw_objects(images, all_detections);
        std::cout << "--> success!" << std::endl;
    } else {
        std::cerr << "--> arguments not right!" << std::endl;
        std::cerr << "--> yolo -model_path ./output.trt -image_path ./demo.jpg" << std::endl;
        return -1;
    }
    return 0;
}
