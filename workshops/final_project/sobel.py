import cv2
import numpy as np

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
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine gradients
    sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_edges = np.uint8(255 * sobel_edges / np.max(sobel_edges))
    
    # Save output if path provided
    if output_path:
        cv2.imwrite(output_path, sobel_edges)
    
    return sobel_edges


if __name__ == "__main__":
    input_image = r"C:\Users\miyum\Downloads\ARC380\arc380_s26_fork\arc380_s26\color.png"
    output_image = r"C:\Users\miyum\Downloads\ARC380\arc380_s26_fork\arc380_s26\sobel_output.jpg"

    result = apply_sobel_edge_detection(input_image, output_image)
    print(f"Sobel edge detection complete. Output saved to {output_image}")