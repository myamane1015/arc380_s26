import cv2
import numpy as np
from cv2 import aruco
from matplotlib import pyplot as plt


def apply_canny(image_path, output_path=None):
    """
    Apply Sobel edge detection to an image.
    
    Args:
        image_path: Path to input image file
        output_path: Path to save output image (optional)
    
    Returns:
        Sobel edge detected image
    """
    # Read image in grayscale
    
    img_path = r'C:\Users\arc380\Downloads\arc380_s26\arc380_s26\palette_1\layer_0.JPG'
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
    
    # Convert to grayscale so contours can be extracted from a single-channel image
    corrected_gray = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(r"C:\Users\arc380\Downloads\arc380_s26\arc380_s26\realsense_shared\gray.jpg", corrected_gray)

    # Apply Canny edge detection
    edges = cv2.Canny(corrected_gray, 10, 40)

    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    filled = cv2.dilate(closed, kernel, iterations=2)
    filled = cv2.erode(filled, kernel, iterations=1)
    
    # Save output if path provided
    if output_path:
        cv2.imwrite(output_path, filled)
    
    return edges


if __name__ == "__main__":
    input_image = r"C:\Users\arc380\Downloads\arc380_s26\arc380_s26\realsense_shared\color.png"
    output_image = r"C:\Users\arc380\Downloads\arc380_s26\arc380_s26\realsense_shared\canny_output.jpg"

    result = apply_canny(input_image, output_image)
    print(f"Canny complete. Output saved to {output_image}")