import grpc
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
import time  

# Triton gRPC Client 초기화
triton_client = grpcclient.InferenceServerClient(url='triton-service:8001')

# sample imput data
input_data = np.random.rand(1, 1, 28, 28).astype(np.float32)

# 1채널 -> 3채널
input_data = np.repeat(input_data, 3, axis=1)  


inputs = []
inputs.append(grpcclient.InferInput('INPUT__0', input_data.shape, np_to_triton_dtype(input_data.dtype)))
inputs[0].set_data_from_numpy(input_data)


model_name = "resnet18"
outputs = []
outputs.append(grpcclient.InferRequestedOutput('OUTPUT__0'))

# inference 및 latency 체크
start_time = time.time()

result = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

end_time = time.time()

latency = (end_time - start_time) * 1000

# inference 결과 로깅
output_data = result.as_numpy('OUTPUT__0')

print(f"Inference latency: {latency:.2f} ms")
print(f"Random 10 inference results: {output_data[:10]}")
