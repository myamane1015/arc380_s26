import cv2
import torch
from torchvision.transforms import Compose

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

encoder = "vits" 
model = DepthAnything.from_pretrained(
    f"LiheYoung/depth_anything_{encoder}14"
).eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        image_interpolation_method=cv2.INTER_CUBIC,
        resize_method="lower_bound",
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

# ---- load image ----
image = cv2.imread("C:\\Users\\miyum\\Downloads\\ARC380\\arc380_s26_fork\\arc380_s26\\color.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

input_tensor = transform({"image": image})["image"]
input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(device)

# ---- inference ----
with torch.no_grad():
    depth = model(input_tensor)

depth_map = depth.squeeze().cpu().numpy()
import matplotlib.pyplot as plt

# Normalize depth map to 0-1 range
depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

# Display with colormap
plt.figure(figsize=(10, 8))
plt.imshow(depth_normalized, cmap='viridis')
plt.colorbar(label='Depth')
plt.title('Depth Map')
plt.axis('off')
plt.tight_layout()
plt.show()