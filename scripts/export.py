import torch
import os
from torchvision import models


model_save_path = '/mnt/data/finetuned_resnet18.pth'
if not os.path.exists(model_save_path):
    print(f"Model file not found: {model_save_path}")
    exit(1)

print(f"Loading model from {model_save_path}...")
model = models.resnet18()
model.fc = torch.nn.Linear(512, 10)  

# 모델 로드
try:
    model.load_state_dict(torch.load(model_save_path))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

model.eval()

# TorchScript로 변환(trace 방식)
try:
    print("Converting model to TorchScript using trace...")
    example_input = torch.rand(1, 3, 28, 28)  
    traced_model = torch.jit.trace(model, example_input)
    print("Model converted to TorchScript using trace successfully.")
except Exception as e:
    print(f"Error converting model to TorchScript using trace: {e}")
    exit(1)

# 모델및 config.pbtxt 저장 경로
export_path = '/mnt/data/triton-models/resnet18/1'
config_path = '/mnt/data/triton-models/resnet18'

os.makedirs(export_path, exist_ok=True)

# TorchScript 모델 저장
torch.jit.save(traced_model, os.path.join(export_path, 'model.pt'))
print(f"Model exported to {export_path} in TorchScript format")

# config.pbtxt 파일 생성
config_content = """
name: "resnet18"
platform: "pytorch_libtorch"
max_batch_size: 32

input [
  {
    name: "INPUT__0"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ 3, 28, 28 ]  
  }
]

output [
  {
    name: "OUTPUT__0"
    data_type: TYPE_FP32
    dims: [ 10 ]  
  }
]
"""

with open(os.path.join(config_path, 'config.pbtxt'), 'w') as config_file:
    config_file.write(config_content)

print(f"Config file created at {os.path.join(config_path, 'config.pbtxt')}")
