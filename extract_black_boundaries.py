import cv2
import numpy as np
import os

def extract_map_boundaries(input_path, output_path, smoothing_level=1):
    """
    Extract and smooth boundaries from a hand-drawn map while removing isolated points.
    
    Args:
        input_path (str): Path to input image
        output_path (str): Path for output image
        smoothing_level (int): Controls smoothing intensity (1-5, where 1=minimal, 5=maximum)
    """
    # Validate smoothing level
    smoothing_level = max(1, min(5, smoothing_level))
    
    # Read and preprocess image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image at {input_path}")
        return None
    
    # Convert to grayscale and invert
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    
    # Edge detection pipeline
    blurred = cv2.GaussianBlur(inverted, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 100)
    dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    
    # Remove small components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    clean_binary = np.zeros_like(dilated)
    
    # Keep only components larger than min_size
    min_size = 100
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            clean_binary[labels == i] = 255
    
    # Smooth the lines based on smoothing_level
    kernel_size = 2 * smoothing_level + 1  # Creates odd numbers: 3, 5, 7, 9, 11
    
    # Apply closing operation with adjustable kernel size
    closing_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(clean_binary, cv2.MORPH_CLOSE, closing_kernel)
    
    # Apply blur with adjustable kernel size
    smoothed = cv2.GaussianBlur(closed, (kernel_size, kernel_size), 0)
    
    # Threshold based on smoothing level to maintain line thickness
    threshold_value = 127 + (smoothing_level * 10)
    _, final = cv2.threshold(smoothed, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Create final image (black lines on white background)
    result = cv2.bitwise_not(final)
    
    # Save outputs
    cv2.imwrite(output_path, result)
    print(f"Map extraction complete: {output_path} (smoothing level: {smoothing_level})")
    
    # Create and save overlay for reference
    edges_color = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
    edges_color[np.where((edges_color == [255, 255, 255]).all(axis=2))] = [0, 0, 255]
    overlay = cv2.addWeighted(img.copy(), 0.7, edges_color, 0.3, 0)
    
    overlay_path = f"{os.path.splitext(output_path)[0]}_overlay{os.path.splitext(output_path)[1]}"
    cv2.imwrite(overlay_path, overlay)
    
    return output_path

# Example usage with different smoothing levels
input_path = "EmpireOfManDivided/original.jpeg"
output_path = "EmpireOfManDivided/EmpireOfManDividedp-Boundaries.jpeg"

# Use smoothing_level parameter to control smoothness (1-5)
# Example: smoothing_level=1 (minimal smoothing)
result = extract_map_boundaries(input_path, output_path, smoothing_level=2)

if result:
    print("Processing complete with overlay for verification.")
    print("Tip: Adjust smoothing_level between 1-5 to find the best result.")