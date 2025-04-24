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
    
    Returns:
        str: Path to the output image if successful, None otherwise
    """
    # Validate smoothing level
    smoothing_level = max(1, min(5, smoothing_level))
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Read and preprocess image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image at {input_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to isolate black elements
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    
    # Remove small components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean_binary = np.zeros_like(binary)
    
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
    edges_color[np.where((edges_color == [255, 255, 255]).all(axis=2))] = [0, 0, 255]  # Red overlay
    overlay = cv2.addWeighted(img.copy(), 0.7, edges_color, 0.3, 0)
    
    overlay_path = f"{os.path.splitext(output_path)[0]}_overlay{os.path.splitext(output_path)[1]}"
    cv2.imwrite(overlay_path, overlay)
    
    return output_path

def main():
    """
    Main function to run the script with command line arguments.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract boundaries from a map image.')
    parser.add_argument('input_path', help='Path to input image')
    parser.add_argument('output_path', help='Path for output image')
    parser.add_argument('--smoothing', type=int, default=2, 
                        help='Smoothing level (1-5, where 1=minimal, 5=maximum)')
    
    args = parser.parse_args()
    
    result = extract_map_boundaries(args.input_path, args.output_path, smoothing_level=args.smoothing)
    
    if result:
        print("Processing complete with overlay for verification.")
        print("Tip: Adjust smoothing_level between 1-5 to find the best result.")

if __name__ == "__main__":
    main()