import os
import torch
import torch.nn as nn
import time

from third_party.insightface.recognition.arcface_torch.backbones import get_model
from third_party.insightface.recognition.arcface_torch.eval.verification import test, load_bin

ckpt_path = "third_party/insightface/recognition/arcface_torch/work_dirs/ms1mv2_r50/model.pt"
state_dict = torch.load(ckpt_path, map_location="cpu")
model = get_model("r50")
model.load_state_dict(state_dict)
model.eval()

size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)

model_int8 = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
model_int8.eval()

quant_ckpt_path = "assets/outputs/best_model_dynamic_quant.pt"
torch.save(model_int8.state_dict(), quant_ckpt_path)
int_size_mb = os.path.getsize(quant_ckpt_path) / (1024 * 1024)

print(f"FP32 model size: {size_mb: .2f} MB")
print(f"INT8 model sizeL {int_size_mb: .2f} MB")

input0 = torch.randn(1, 3, 112, 112)

def times(model, x, warmup = 20, runs = 100):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(x)
    end = time.perf_counter()
    return (end-start) / runs

old_time = times(model, input0)
new_time = times(model_int8, input0)

print(f"FP32 average inference time: {old_time: .3f} s")
print(f"INT8 average inference time: {new_time: .3f} s")
print(f"Speedup: {old_time / new_time: .3f}x")


data_path = "data/MS1M/lfw.bin"
image_size = [112, 112]
batch_size = 32
data_set = load_bin(data_path, image_size)

_, _, old_acc, old_std, _, _ = test(data_set, model, batch_size) 
_, _, new_acc, new_std, _, _ = test(data_set, model_int8, batch_size)

print(f"FP32 LFW Accuracy: {old_acc: .4f} +- {old_std: .4f}")
print(f"INT8 LFW Accuracy: {new_acc: .4f} +- {new_std: .4f}")
