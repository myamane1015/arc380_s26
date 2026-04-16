import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
from cv2 import aruco
from matplotlib import pyplot as plt

# Load image
img_path = r'C:\Users\arc380\Downloads\arc380_s26\arc380_s26\palette_2\layer_1.JPG'
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
h, w, _ = corrected_img.shape


# Reshape to (num_pixels, 3)
pixels = corrected_img.reshape(-1, 3)
K = 5

# Fit GMM
gmm = GaussianMixture(n_components=K, covariance_type='tied')
gmm.fit(pixels)

# Predict cluster probabilities
labels = gmm.predict(pixels)

mask_k = (labels.reshape(h, w) == K).astype(np.uint8)

num_labels, comps = cv2.connectedComponents(mask_k)

for i in range(1, num_labels):  # skip background
    comp_mask = (comps == i)
    
    ys, xs = np.where(comp_mask)
    x_c = xs.mean()
    y_c = ys.mean()
    
    cv2.circle(corrected_img, (int(x_c), int(y_c)), 5, (0, 255, 0), -1)

plt.imshow(corrected_img)
plt.title(f'GMM Clustering (K={K})')
plt.show()