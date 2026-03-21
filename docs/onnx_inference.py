import numpy as np
import onnxruntime as ort
import torch

from third_party.insightface.recognition.arcface_torch.backbones import get_model

ckpt_path = "third_party/insightface/recognition/arcface_torch/work_dirs/ms1mv2_r50/model.pt"
state_dict = torch.load(ckpt_path, map_location="cpu")
onnx_path = "assets/outputs/ms1mv2_r50.onnx"

model = get_model("r50", fp16=False)
model.load_state_dict(state_dict)
model.eval()

#execute onnx model with onnx runtime
example_inputs = (torch.rand(1, 3, 112, 112),)
onnx_inputs = [tensor.numpy(force=True) for tensor in example_inputs]

session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(session.get_inputs(), onnx_inputs)}

ort_out = session.run(None, onnxruntime_input)[0]

#compare PyTorch results with the ones from the onnx time
torch_out = model(*example_inputs)

assert len(torch_out) == len(ort_out)
for torch_o, ort_o in zip(torch_out, ort_out):
    torch.testing.assert_close(torch_o, torch.tensor(ort_o))

    max_abs_diff = np.max(np.abs(torch_o.detach().cpu().numpy() - ort_o))
    mean_abs_diff = np.mean(np.abs(torch_o.detach().cpu().numpy() - ort_o))
    print(f"PyTorch output shape: {torch_o.shape}")
    print(f"ONNX output shape: {ort_o.shape}")
    print(f"Max abs diff: {max_abs_diff:.8f}")
    print(f"Mean abs diff: {mean_abs_diff:.8f}")

print("PyTorch and ONNX Runtime output matched!")
print(f"Output length:{len(ort_out)}")