import cv2
import torch
from torchvision.transforms import Compose
from cv2 import aruco
import sys
import os
import numpy as np

sys.path.append(os.path.abspath("Depth-Anything"))

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
img_path = r'C:\Users\arc380\Downloads\arc380_s26\arc380_s26\realsense_shared\base.png'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the predefined dictionary where our markers are printed from
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# Load the default detector parameters
detector_params = aruco.DetectorParameters()

# Create an ArucoDetector using the dictionary and detector parameters
detector = aruco.ArucoDetector(dictionary, detector_params)

# Run the detector on our input image
corners, ids, rejected = detector.detectMarkers(img)
#print(ids)
#print(corners)

# Define the dimensions of the output image
width = 10      # inches
height = 7.5    # inches
ppi = 96        # pixels per inch (standard resolution for most screens - can be any arbitrary value that still preserves information)

_ids = ids.flatten()

# Sort corners based on id
ids = ids.flatten()
#print(ids)

# Sort the corners based on the ids
corners = np.array([corners[i] for i in np.argsort(ids)])
#print(corners.shape)

# Remove dimensions of size 1
corners = np.squeeze(corners)
#print(corners)

# Sort the ids
ids = np.sort(ids)

# Extract source points corresponding to the exterior bounding box corners of the 4 markers
src_pts = np.array([corners[0][0], corners[1][1], corners[2][2], corners[3][3]], dtype='float32')

# Define destination points as the corners of the output image
dst_pts = np.array([[0, 0], [0, height*ppi], [width*ppi, height*ppi], [width*ppi, 0]], dtype='float32')

# Compute the perspective transformation matrix
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Apply the perspective transformation to the input image
corrected_img = cv2.warpPerspective(img, M, (img_rgb.shape[1], img_rgb.shape[0]))

# Crop the output image to the specified dimensions
corrected_img = corrected_img[:int(height*ppi), :int(width*ppi)]

input_tensor = transform({"image": corrected_img})["image"]
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