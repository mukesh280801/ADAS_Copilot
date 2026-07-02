from ultralytics import YOLO
import torch

print("=" * 50)
print("CUDA Available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("=" * 50)

# Load YOLOv8 Nano
model = YOLO("yolov8n.pt")

# Test Detection
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    device=0,
    show=True,
    conf=0.5
)

print("✅ YOLO Detection Complete")