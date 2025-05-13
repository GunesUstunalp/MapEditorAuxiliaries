import cv2
import numpy as np
import os
import argparse
import json
import time
from scipy.ndimage import distance_transform_edt
from collections import defaultdict

def remove_boundaries(input_path, output_path, colors_json_path=None):
    """
    Removes black boundaries from a colored map by assigning each boundary pixel
    to the nearest province.
    
    Args:
        input_path (str): Path to the colored map image with black boundaries
        output_path (str): Path for saving the map with boundaries removed
        colors_json_path (str, optional): Path to the JSON file containing province colors
        
    Returns:
        ndarray: Map with boundaries removed
    """
    # Start timing the process
    start_time = time.time()
    print(f"Starting boundary removal process...")
    
    # Load the colored map
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image at {input_path}")
        return None
    
    # Create masks for provinces and boundaries
    # Boundaries are black pixels
    height, width = img.shape[:2]
    boundaries_mask = np.all(img == [0, 0, 0], axis=2)
    provinces_mask = ~boundaries_mask
    
    print(f"Image dimensions: {width}x{height}")
    print(f"Found {np.sum(boundaries_mask)} boundary pixels to process")
    
    # Create a copy of the image for our result
    result = img.copy()
    
    # Load colors data if available (for verification)
    colors_data = None
    if colors_json_path and os.path.exists(colors_json_path):
        with open(colors_json_path, 'r') as f:
            colors_data = json.load(f)
        print(f"Loaded {len(colors_data)} province colors from {colors_json_path}")
    
    # Two alternative approaches:
    # 1. Fast method using distance transform
    # 2. Slower but more accurate method using nearest color
    
    # Method 1: Use distance transform to find nearest non-boundary pixel
    print("Using distance transform to assign boundary pixels to nearest provinces...")
    
    # Calculate distance transform for non-boundary regions
    dist_transform = distance_transform_edt(boundaries_mask)
    
    # For each province, create a mask and apply distance transform
    provinces_labels = np.zeros((height, width), dtype=np.int32)
    
    # Find unique colors in the image (excluding black)
    unique_colors = []
    for y in range(height):
        for x in range(width):
            if not boundaries_mask[y, x]:  # If it's a province pixel
                color = tuple(img[y, x])
                if color not in unique_colors:
                    unique_colors.append(color)
    
    print(f"Found {len(unique_colors)} unique province colors")
    
    # Create a mask for each province color
    for i, color in enumerate(unique_colors, 1):
        color_mask = np.all(img == color, axis=2)
        provinces_labels[color_mask] = i
    
    # Calculate Voronoi regions using distance transform
    voronoi = np.zeros((height, width, len(unique_colors)), dtype=np.float32)
    
    for i, color in enumerate(unique_colors):
        # Create a mask for this color
        color_mask = np.all(img == color, axis=2)
        
        # Calculate distance from each pixel to this color's province
        voronoi[:, :, i] = distance_transform_edt(~color_mask)
    
    # For each boundary pixel, find the nearest province
    nearest_province = np.argmin(voronoi, axis=2)
    
    # Replace boundary pixels with color of nearest province
    for y in range(height):
        for x in range(width):
            if boundaries_mask[y, x]:
                nearest_color_index = nearest_province[y, x]
                result[y, x] = unique_colors[nearest_color_index]
    
    # Verify that all boundary pixels have been replaced
    remaining_boundaries = np.all(result == [0, 0, 0], axis=2)
    if np.any(remaining_boundaries):
        print(f"Warning: {np.sum(remaining_boundaries)} boundary pixels remain!")
    else:
        print("All boundary pixels successfully assigned to provinces")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save result as PNG with no compression for lossless output
    cv2.imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"Map with boundaries removed saved to {output_path}")
    
    # Report timing
    elapsed_time = time.time() - start_time
    print(f"Boundary removal completed in {elapsed_time:.2f} seconds")
    
    return result

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Remove boundaries from a colored map')
    parser.add_argument('input', help='Path to the colored map image with boundaries')
    parser.add_argument('output', help='Path for saving the map with boundaries removed')
    parser.add_argument('--colors-json', help='Path to the JSON file containing province colors')
    
    args = parser.parse_args()
    
    # Process the map
    result = remove_boundaries(
        input_path=args.input, 
        output_path=args.output,
        colors_json_path=args.colors_json
    )
    
    if result is not None:
        print("Boundary removal complete!")

if __name__ == "__main__":
    main()