import cv2
import numpy as np
from matplotlib import pyplot as plt
from cv2 import aruco

img_path = 'raw_image.png'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

# Reshape our image data to a flattened list of RGB values
img_data = corrected_img.reshape((-1, 3))
img_data = np.float32(img_data)

# Define the number of clusters
k = 3

# Define the criteria for the k-means algorithm
# This is a tuple with three elements: (type of termination criteria, maximum number of iterations, epsilon/required accuracy)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Run the k-means algorithm
# Parameters: data, number of clusters, best labels, criteria, number of attempts, initial centers
_, labels, centers = cv2.kmeans(img_data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# The output of the k-means algorithm gives the centers as floating point values
# We need to convert these back to uint8 to be able to use them as pixel values
centers = np.uint8(centers)

# Rebuild the image using the labels and centers
kmeans_data = centers[labels.flatten()]
kmeans_img = kmeans_data.reshape(corrected_img.shape)
labels = labels.reshape(corrected_img.shape[:2])
    
# Identify the cluster that is closest to the dark green color
tan = np.array([0, 100, 0])
distances = np.linalg.norm(centers - tan, axis=1)
block_cluster_label = np.argmin(distances)

# Create a mask image for this label
# All pixels that belong to this cluster will be white, and all others will be black
mask_img = np.zeros(kmeans_img.shape[:2], dtype='uint8')
mask_img[labels == block_cluster_label] = 255

# Segment continuous regions
# Parameters: input image, contour retrieval mode, contour approximation method
contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

areas = [cv2.contourArea(contour) for contour in contours]
print(f'Area of each region: {areas}')