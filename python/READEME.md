# Jeston-Inference (Ultralytics with TensorRT)

简要说明：本目录包含一个基于 TensorRT 的推理示例（infer.py），用于加载 TensorRT engine 并对图片做目标检测。

必备环境
- Ubuntu（推荐 Jetson 系列使用 JetPack）
- CUDA / cuDNN / TensorRT（与目标 JetPack 版本匹配）
- Python 3.6+（在 Jetson 上推荐随 JetPack 提供的 Python）

推荐 Python 包（与项目测试通过的版本）
```bash
pip3 install --no-cache-dir pycuda==2021.1
pip3 install numpy==1.21.6
```

使用方法（最小示例）
1. 准备好 TensorRT 引擎文件（例如：yolo11n.engine），放在与 infer.py 相同目录，或在代码中传入正确路径。
2. 准备一张测试图片，命名为 test.jpg（或修改 infer.py 中的路径）。
3. 运行：
```bash
python3 infer.py
```
运行后结果图像将保存到 ./result/ 下，文件名格式为 0.jpg, 1.jpg 等。
