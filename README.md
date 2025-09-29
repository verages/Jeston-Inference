# YOLO 系列 End2End 推理（Jetson）

简介
该项目包含用于在 Jetson 系列设备上运行的 YOLO End2End 推理相关说明与注意事项，配套 C++ 推理代码（兼容多 batch，但默认单 batch）。

快速要点
- 支持多 Batch 推理（默认单 Batch）。
- 需要手动设置的参数：BBOX_CONF_THRESH（置信度阈值）、BATCH_SIZE（batch 大小）。
- 原项目由单帧推理改写为多帧版本，已测试通过。
- 多 Batch 导出与验证仍在测试中，后续更新。
- 使用Ultralytics中的v8,v11模型,CPP代码理论上为通用代码,其余版本请自行验证

环境与前置依赖
- Python（用于模型导出，建议 3.8+）
- ultralytics（用于导出 ONNX）
- TensorRT（用于 trtexec 导出与引擎生成）
- onnx、onnx-simplifier（可选，视导出流程）

导出流程（推荐）
1. 从 ultralytics 导出 ONNX：
```python
# 示例
from ultralytics import YOLO
model = YOLO("yolo11n.pt")      # 加载预训练模型
model.fuse()
model.export(format='onnx', simplify=True, opset=13)  # 导出 ONNX
```

1. 对 ONNX 添加 NMS 模块（仅支持 YOLOv8 / YOLOv11）：
```
python yolo_nms_export.py --model_path yolo11n.onnx --output_path yolo11n_nms.onnx --num_classes 80
```

TensorRT 导出（单帧）
- 单帧导出示例：
```
trtexec --onnx=yolo11n_nms.onnx --saveEngine=yolo11n.engine --fp16
```
关键参数说明
- BBOX_CONF_THRESH：检测框置信度阈值；在后处理代码中设置（例如 0.25/0.3），影响召回与精度平衡。
- BATCH_SIZE：推理时的 batch 大小；默认 1。


注意事项（摘要）
- 请根据项目代码中提示修改 BBOX_CONF_THRESH 与 BATCH_SIZE。
- 多 Batch 功能仍需自行验证；使用前务必在目标设备上完整测试。
- 本 README 侧重导出与部署流程，具体推理源码请参考仓库中的 C++ 源文件。
