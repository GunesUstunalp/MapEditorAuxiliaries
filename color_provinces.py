import cv2
import numpy as np
import os
import argparse
import json

def color_provinces(input_path=None, output_path=None, image=None):
    """
    Colors every province (closed space) with unique colors.
    Uses strict binary approach and saves as lossless PNG.
    
    Args:
        input_path (str, optional): Path to the map image with borders
        output_path (str, optional): Path for saving the colored map (should end with .png)
        image (ndarray, optional): Input image as numpy array instead of loading from file
        
    Returns:
        ndarray: Colored provinces image
        int: Number of provinces found
        list: List of colors used (in BGR format)
    """
    # Handle either file input or direct image input
    if image is None and input_path:
        img = cv2.imread(input_path)
        if img is None:
            print(f"Error: Could not read image at {input_path}")
            return None, 0, []
    elif image is not None:
        img = image.copy()
    else:
        print("Error: Either input_path or image must be provided")
        return None, 0, []
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use a strict binary threshold - no interpolation
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Invert to get the regions (provinces)
    regions = cv2.bitwise_not(binary)
    
    # Use connected components with 4-connectivity to label regions
    num_labels, labels = cv2.connectedComponents(regions, connectivity=4)
    
    # Create output image - start with all white
    output = np.ones(img.shape, dtype=np.uint8) * 255
    
    # List to store all colors used
    colors_used = []
    
    # Color each province with a unique color using direct pixel assignment
    for i in range(1, num_labels):
        # Create a color with good separation in HSV space
        hue = (i * 67) % 180
        saturation = 200
        value = 220
        
        # Convert HSV to BGR
        hsv_color = np.array([[[hue, saturation, value]]], dtype=np.uint8)
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        
        # Add color to the list
        colors_used.append(bgr_color.tolist())
        
        # Create a binary mask for this label only
        province_mask = (labels == i)
        
        # Apply color directly without any interpolation
        output[province_mask] = bgr_color
    
    # Set all border pixels to black - use the original binary threshold
    output[binary == 255] = [0, 0, 0]
    
    # Save as PNG if output_path is provided
    if output_path:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, output, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"Map provinces colored: {output_path}")
        print(f"Found {num_labels-1} distinct provinces")
        
        # Save color information to JSON file
        colors_json_path = os.path.splitext(output_path)[0] + "_colors.json"
        with open(colors_json_path, 'w') as f:
            json.dump(colors_used, f, indent=2)
        print(f"Color information saved to: {colors_json_path}")
        
        # Check for duplicate colors
        check_duplicate_colors(colors_used)
    
    return output, num_labels-1, colors_used

def check_duplicate_colors(colors):
    """
    Check if there are any duplicate colors in the list.
    
    Args:
        colors (list): List of colors in BGR format
    
    Returns:
        bool: True if duplicates found, False otherwise
    """
    # Convert list of lists to tuples for hashability
    color_tuples = [tuple(color) for color in colors]
    
    # Check for duplicates
    unique_colors = set()
    duplicates = []
    
    for i, color in enumerate(color_tuples):
        if color in unique_colors:
            duplicates.append((i+1, color))
        else:
            unique_colors.add(color)
    
    # Report results
    if duplicates:
        print("\nERROR: Duplicate colors detected!")
        print(f"Found {len(duplicates)} duplicate colors:")
        for province_id, color in duplicates:
            print(f"  Province {province_id}: BGR={color}")
        return True
    else:
        print("\nNo duplicate colors found. All provinces have unique colors.")
        return False

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Color provinces in a map')
    parser.add_argument('input', help='Path to the input map image with borders')
    parser.add_argument('output', help='Path for saving the colored map (should end with .png)')
    
    args = parser.parse_args()
    
    # Process the map
    result, province_count, colors = color_provinces(input_path=args.input, output_path=args.output)
    if result is not None:
        print("Province coloring complete with strict binary coloring in lossless PNG format!")

if __name__ == "__main__":
    main()