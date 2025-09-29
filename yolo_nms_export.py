import argparse
import onnx_graphsurgeon as gs
import numpy as np
import onnx

class YOLOPostprocess:
    def __init__(self, model_path, num_classes=80, top_k=1000, keep_top_k=20,
                 score_threshold=0.20, iou_threshold=0.45):
        self.num_classes = num_classes
        self.model_path = model_path
        self.graph = gs.import_onnx(onnx.load(model_path))
        
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold


    def _create_transpose_node(self):
        model_output = self.graph.outputs[0]
        origin_output = [node for node in self.graph.nodes if model_output in node.outputs][0]
        print("Origin output node name:", origin_output.name)
        
        output_t = gs.Variable(name="output_t", shape=(1, 8400, 4 + self.num_classes), dtype=np.float32)
        return gs.Node(op="Transpose", inputs=[origin_output.outputs[0]], outputs=[output_t], attrs={"perm":[0,2,1]})
    
    def _create_split_nodes(self, output_t):
        split_sizes = gs.Constant("split_size", values=np.array([1, 1, 1, 1, self.num_classes], dtype=np.int64))
        
        x = gs.Variable(name="x_center", shape=(1,8400,1), dtype=np.float32)
        y = gs.Variable(name="y_center", shape=(1,8400,1), dtype=np.float32)
        w = gs.Variable(name="w", shape=(1,8400,1), dtype=np.float32)
        h = gs.Variable(name="h", shape=(1,8400,1), dtype=np.float32)
        label_conf = gs.Variable(name='label_conf', shape=(1,8400,self.num_classes), dtype=np.float32)
        
        split_node = gs.Node(op="Split",
                           inputs=[output_t, split_sizes],
                           outputs=[x, y, w, h, label_conf],
                           attrs={"axis": 2})
        
        return split_node, x, y, w, h, label_conf
    
    def _create_box_transform_nodes(self, w, h):
        div_val = gs.Constant("div_val", values=np.array([2], dtype=np.float32))
        div_val_ = gs.Constant("div_val_", values=np.array([-2], dtype=np.float32))
        
        w_ = gs.Variable(name="w_half_", shape=(1,8400,1), dtype=np.float32)
        wplus = gs.Variable(name="w_half_plus", shape=(1,8400,1), dtype=np.float32)
        h_ = gs.Variable(name="h_half_", shape=(1,8400,1), dtype=np.float32)
        hplus = gs.Variable(name="h_half_plus", shape=(1,8400,1), dtype=np.float32)
        
        transform_nodes = [
            gs.Node(op="Div", inputs=[w, div_val_], outputs=[w_]),
            gs.Node(op="Div", inputs=[w, div_val], outputs=[wplus]),
            gs.Node(op="Div", inputs=[h, div_val_], outputs=[h_]),
            gs.Node(op="Div", inputs=[h, div_val], outputs=[hplus])
        ]
        
        return transform_nodes, w_, wplus, h_, hplus
    
    def _create_nms_attrs(self):
        return {
            "background_class": np.int32(-1),
            "max_output_boxes": np.int32(self.keep_top_k),
            "score_threshold": np.float32(self.score_threshold),
            "iou_threshold": np.float32(self.iou_threshold),
            "box_coding": np.int32(0),
            "score_type": np.int32(0),
            "score_activation": np.int32(0), # 必须添加！0=无激活，1=sigmoid
            "class_agnostic": np.int32(0),
            "plugin_version": "1"
        }

    def _add_nms_node(self):
        batch_size = self.graph.inputs[0].shape[0]
        tensors = self.graph.tensors()
        boxes_tensor = tensors["boxes"]
        confs_tensor = tensors["label_conf"]
        
        # Reshape boxes 为 [batch, num_boxes, 4]
        new_shape = gs.Constant("new_shape", values=np.array([-1, 8400, 4], dtype=np.int64))
        boxes_reshaped = gs.Variable(name="boxes_reshaped", dtype=np.float32, shape=[batch_size, 8400, 4])
        reshape_node = gs.Node(
            op="Reshape",
            inputs=[boxes_tensor, new_shape],
            outputs=[boxes_reshaped]
        )
        
        num_detections = gs.Variable(name="num_detections", dtype=np.int32, shape=[batch_size, 1])
        nmsed_boxes = gs.Variable(name="nmsed_boxes", dtype=np.float32, shape=[batch_size, self.keep_top_k, 4])
        nmsed_scores = gs.Variable(name="nmsed_scores", dtype=np.float32, shape=[batch_size, self.keep_top_k])
        nmsed_classes = gs.Variable(name="nmsed_classes", dtype=np.int32, shape=[batch_size, self.keep_top_k]) 
        
        new_outputs = [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes]
        
        nms_node = gs.Node(
            op="EfficientNMS_TRT",
            attrs=self._create_nms_attrs(),
            inputs=[boxes_reshaped, confs_tensor],
            outputs=new_outputs
        )
        
        self.graph.nodes.append(reshape_node)
        self.graph.nodes.append(nms_node)
        self.graph.outputs = new_outputs
        return self.graph.cleanup().toposort()

    def process(self):
        output_t_node = self._create_transpose_node()
        
        split_node, x, y, w, h, label_conf = self._create_split_nodes(output_t_node.outputs[0])
        
        transform_nodes, w_, wplus, h_, hplus = self._create_box_transform_nodes(w, h)
        
        x1 = gs.Variable(name="x1", shape=(1,8400,1), dtype=np.float32)
        y1 = gs.Variable(name="y1", shape=(1,8400,1), dtype=np.float32)
        x2 = gs.Variable(name="x2", shape=(1,8400,1), dtype=np.float32)
        y2 = gs.Variable(name="y2", shape=(1,8400,1), dtype=np.float32)
        
        final_nodes = [
            gs.Node(op="Add", inputs=[x, w_], outputs=[x1]),
            gs.Node(op="Add", inputs=[x, wplus], outputs=[x2]),
            gs.Node(op="Add", inputs=[y, h_], outputs=[y1]),
            gs.Node(op="Add", inputs=[y, hplus], outputs=[y2])
        ]
        
        boxes = gs.Variable(name="boxes", shape=(1, 8400, 4), dtype=np.float32) 
        output_nodes = [
            gs.Node(op="Concat", inputs=[x1, y1, x2, y2], outputs=[boxes], attrs={"axis": 2})
        ]
        
        all_nodes = [output_t_node, split_node] + \
                   transform_nodes + final_nodes + output_nodes
        self.graph.nodes.extend(all_nodes)
        
        self.graph.outputs = [boxes, label_conf]
        self.graph.cleanup().toposort()

    def save(self, output_path, add_nms=True):
        if add_nms:
            self._add_nms_node()
        onnx.save(gs.export_onnx(self.graph), output_path)

def main():
    parser = argparse.ArgumentParser(description="YOLOv8&YOLOv11 Postprocess with NMS")
    parser.add_argument("--model_path", type=str,default='yolo11n.onnx', help="Path to the YOLOv8 ONNX model")
    parser.add_argument("--num_classes", type=int, default=80, help="Number of classes in the model")
    parser.add_argument("--output_path", type=str, default="yolo11n_nms.onnx", help="Path to save the modified ONNX model")
    parser.add_argument("--top_k", type=int, default=1000, help="Number of boxes for NMS")
    parser.add_argument("--keep_top_k", type=int, default=100, help="Boxes to keep per image")
    parser.add_argument("--score_threshold", type=float, default=0.20, help="Score threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.65, help="IOU threshold")
    parser.add_argument("--add_nms", type=bool, default=True, help="Whether to add NMS")
    
    args = parser.parse_args()
    
    processor = YOLOPostprocess(
        model_path=args.model_path, 
        num_classes=args.num_classes,
        top_k=args.top_k,
        keep_top_k=args.keep_top_k,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold
    )
    processor.process()
    processor.save(output_path=args.output_path, add_nms=args.add_nms)


if __name__ == "__main__":
    main()
