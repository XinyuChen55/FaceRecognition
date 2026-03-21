import os
import torch

from third_party.insightface.recognition.arcface_torch.backbones import get_model

ckpt_path = "third_party/insightface/recognition/arcface_torch/work_dirs/ms1mv2_r50/model.pt"
state_dict = torch.load(ckpt_path, map_location="cpu")

model = get_model("r50", fp16=False)
model.load_state_dict(state_dict)
model.eval()

example_inputs = (torch.randn(1, 3, 112, 112),)
onnx_path = "assets/outputs/ms1mv2_r50.onnx"
os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

torch.onnx.export(model, example_inputs, onnx_path)