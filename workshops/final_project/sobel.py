import cv2
import numpy as np
from cv2 import aruco


def apply_sobel_edge_detection(image_path, output_path=None):
    """
    Apply Sobel edge detection to an image.
    
    Args:
        image_path: Path to input image file
        output_path: Path to save output image (optional)
    
    Returns:
        Sobel edge detected image
    """
    # Read image in grayscale
    
    img_path = r'C:\Users\miyum\Downloads\ARC380\arc380_s26_fork\arc380_s26\color.png'
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

    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(corrected_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(corrected_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine gradients
    sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)
    max_val = np.max(sobel_edges)
    if max_val > 0:
        sobel_edges = np.uint8(255 * sobel_edges / max_val)
    else:
        sobel_edges = np.zeros_like(corrected_gray, dtype=np.uint8)
    
    # Find contours in the Sobel edge image
    contours, _ = cv2.findContours(sobel_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the bounding rectangle around all contours
    if contours:
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        # Draw bounding box on the Sobel image
        sobel_edges = cv2.rectangle(sobel_edges, (x, y), (x + w, y + h), 255, 2)
        
    # Draw all contours on the Sobel image
    cv2.drawContours(sobel_edges, contours, -1, 255, 2)
    
    # Save output if path provided
    if output_path:
        cv2.imwrite(output_path, sobel_edges)
    
    return sobel_edges


if __name__ == "__main__":
    input_image = r"C:\Users\miyum\Downloads\ARC380\arc380_s26_fork\arc380_s26\color.png"
    output_image = r"C:\Users\miyum\Downloads\ARC380\arc380_s26_fork\arc380_s26\sobel_output.jpg"

    result = apply_sobel_edge_detection(input_image, output_image)
    print(f"Sobel edge detection complete. Output saved to {output_image}")