import torch

def check_pytorch_gpu():
    print("\n=== PyTorch GPU检测 ===")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"可用GPU数量: {device_count}")
        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("未检测到可用GPU")

check_pytorch_gpu()