# Jeston-Inference YOLO (C++ / TensorRT)

简介
---
这是一个基于 TensorRT 与 CUDA 的 YOLO 推理示例，针对 aarch64（Jetson）平台优化。包含图像预处理（warp affine）、TensorRT 引擎加载与推理、以及结果可视化（保存为 result_*.jpg）。
该工程适配多Batch推理,修改BATCH_SIZE为需要的size即可

主要文件
---
- main.cpp：程序入口，解析参数并运行推理流程。
- yolo.h / yolo.cpp：TensorRT 引擎加载、推理、结果解析与绘图。
- warp_affine.cu / warp_affine.h：GPU 上的仿射与双线性插值预处理。
- CMakeLists.txt：构建脚本，默认仅支持 aarch64（Jetson）。

先决条件
---
- aarch64 平台（Jetson 系列）
- CUDA（与 Jetson 系统匹配）
- TensorRT（包含头文件与库，例如 /usr/lib/aarch64-linux-gnu）,适配版本为8.*,10以上因为API变动需要做一点修改
- OpenCV（用于读取/写入图像）
- CMake >= 3.10, g++ / nvcc

构建（示例）
---
```bash
cd /Jeston-Inference/cpp
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```
注意：CMakeLists.txt 当前会在非 aarch64 系统上报错。如果需要在 x86 上构建，请修改 CMakeLists.txt 中平台检查与路径。

运行示例
---
编译成功后，运行：
```bash
./yolo -model_path ./output.trt -image_path ./imgs/
```
- model_path：TensorRT 序列化引擎文件（.trt）
- image_path： glob 模式（main.cpp 使用 cv::glob），示例可为 `./imgs/ `
- 结果保存为 `./result_0.jpg` 等文件

常见配置
---
- 在 yolo.h 中可以调整 IMG_SIZE、BATCH_SIZE、阈值等。
- CMakeLists.txt 中为 Jetson 指定了 CUDA 与 TensorRT 路径，若系统路径不同请调整 `CUDA_INCLUDE_DIR`、`CUDA_LIB_DIR`、`TENSORRT_ROOT`。
- 插件与自定义算子：warp_affine.cu 被编译为 myplugins，共享库会与主程序链接。
