import cv2
import numpy as np
import json
import os
import argparse

def find_province_centroids(colored_map_path, colors_json_path, output_json_path=None, display_width=8000, display_height=4000):
    """
    Find the centroid (center point) of each province in a colored map.
    Adjusts coordinates to be centered at (0,0) with specified display dimensions.
    
    Args:
        colored_map_path (str): Path to the colored map image
        colors_json_path (str): Path to the JSON file with province colors
        output_json_path (str, optional): Path to save the output JSON with centroids
        display_width (int): Width of the display area (default: 8000)
        display_height (int): Height of the display area (default: 4000)
        
    Returns:
        dict: Dictionary mapping province colors to their centroids {color: [x, y, 0]}
    """
    # Load the colored map
    print(f"Loading colored map from {colored_map_path}...")
    colored_map = cv2.imread(colored_map_path)
    if colored_map is None:
        raise ValueError(f"Could not read image from {colored_map_path}")
    
    # Get image dimensions
    img_height, img_width = colored_map.shape[:2]
    print(f"Image dimensions: {img_width}x{img_height}")
    
    # Calculate scaling factors
    scale_x = display_width / img_width
    scale_y = display_height / img_height
    print(f"Scaling factors: x={scale_x}, y={scale_y}")
    
    # Load the colors JSON file
    print(f"Loading colors JSON from {colors_json_path}...")
    with open(colors_json_path, 'r') as f:
        colors = json.load(f)
    
    # Create a dictionary to store the centroids
    province_centroids = {}
    
    # Compute centroids for each province color
    print(f"Finding centroids for {len(colors)} provinces...")
    for i, color in enumerate(colors, 1):
        # Convert color from list to tuple
        color_tuple = tuple(color)
        
        # Create a mask for pixels matching this color
        # Note: OpenCV uses BGR format, so we need to compare with the BGR color
        mask = np.all(colored_map == color_tuple, axis=2).astype(np.uint8) * 255
        
        # Verify that we found pixels with this color
        pixel_count = np.count_nonzero(mask)
        if pixel_count == 0:
            print(f"  Warning: No pixels found for province {i} with color {color_tuple}")
            continue
        
        # Find the contours of the province
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no contours were found, try a different approach
        if not contours:
            print(f"  No contours found for province {i}, using pixel coordinates instead")
            # Get the coordinates of all pixels with this color
            y_coords, x_coords = np.where(mask > 0)
            # Calculate the centroid as the mean of all pixel coordinates
            cx_raw = int(np.mean(x_coords))
            cy_raw = int(np.mean(y_coords))
        else:
            # Get the largest contour (in case there are multiple disconnected areas)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate the moments of the contour
            M = cv2.moments(largest_contour)
            
            # Calculate the centroid from the moments
            if M["m00"] > 0:
                cx_raw = int(M["m10"] / M["m00"])
                cy_raw = int(M["m01"] / M["m00"])
            else:
                # Fallback if moments method fails
                print(f"  Moments method failed for province {i}, using bounding box center")
                x, y, w, h = cv2.boundingRect(largest_contour)
                cx_raw = x + w // 2
                cy_raw = y + h // 2
        
        # Transform coordinates to be centered at (0,0) with specified scaling
        # First, shift origin to center of image
        cx_centered = cx_raw - (img_width / 2)
        cy_centered = (img_height / 2) - cy_raw  # Flip Y-axis so positive is up
        
        # Then scale to display dimensions
        cx_scaled = cx_centered * scale_x
        cy_scaled = cy_centered * scale_y
        
        # Store the centroid with z=0
        province_centroids[str(color_tuple)] = [cx_scaled, cy_scaled, 0]
        print(f"  Province {i}: raw centroid ({cx_raw}, {cy_raw}) -> transformed ({cx_scaled:.2f}, {cy_scaled:.2f}, 0)")
    
    # Save the centroids to a JSON file if specified
    if output_json_path:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        
        # Save the centroids
        with open(output_json_path, 'w') as f:
            json.dump(province_centroids, f, indent=2)
        
        print(f"Centroids saved to {output_json_path}")
    
    return province_centroids

def visualize_centroids(colored_map_path, centroids, output_path=None, display_width=8000, display_height=4000):
    """
    Create a visualization of the province centroids.
    Shows both the raw image coordinates and the transformed coordinates.
    
    Args:
        colored_map_path (str): Path to the colored map image
        centroids (dict): Dictionary mapping province colors to their centroids
        output_path (str, optional): Path to save the visualization
        display_width (int): Width of the display area
        display_height (int): Height of the display area
    """
    # Load the colored map
    colored_map = cv2.imread(colored_map_path)
    if colored_map is None:
        raise ValueError(f"Could not read image from {colored_map_path}")
    
    # Get image dimensions
    img_height, img_width = colored_map.shape[:2]
    
    # Calculate scaling factors
    scale_x = img_width / display_width
    scale_y = img_height / display_height
    
    # Create a copy for visualization
    vis_map = colored_map.copy()
    
    # Draw each centroid
    for i, (color_str, (cx_scaled, cy_scaled, _)) in enumerate(centroids.items(), 1):
        # Convert string back to tuple
        color = eval(color_str)
        
        # Transform back to image coordinates for visualization
        cx_centered = cx_scaled * scale_x
        cy_centered = cy_scaled * scale_y
        
        # Reverse the transformation
        cx_raw = int(cx_centered + (img_width / 2))
        cy_raw = int((img_height / 2) - cy_centered)
        
        # Ensure coordinates are within image bounds
        cx_raw = max(0, min(cx_raw, img_width - 1))
        cy_raw = max(0, min(cy_raw, img_height - 1))
        
        # Calculate a contrasting color for the marker
        marker_color = (255, 255, 255) if sum(color) < 380 else (0, 0, 0)
        
        # Draw a circle at the centroid
        cv2.circle(vis_map, (cx_raw, cy_raw), 5, marker_color, -1)
        cv2.circle(vis_map, (cx_raw, cy_raw), 7, marker_color, 2)
        
        # Add coordinate labels
        label = f"({cx_scaled:.0f}, {cy_scaled:.0f}, 0)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Create a background rectangle for the text
        cv2.rectangle(
            vis_map, 
            (cx_raw - 5, cy_raw + 10), 
            (cx_raw + text_width + 5, cy_raw + text_height + 15),
            (255, 255, 255),
            -1
        )
        
        # Add the text
        cv2.putText(
            vis_map, 
            label, 
            (cx_raw, cy_raw + text_height + 10), 
            font, 
            font_scale, 
            (0, 0, 0), 
            thickness
        )
    
    # Save the visualization if specified
    if output_path:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the visualization
        cv2.imwrite(output_path, vis_map)
        print(f"Visualization saved to {output_path}")
    
    return vis_map

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Find centroids of colored provinces')
    parser.add_argument('input', help='Path to the input colored map image')
    parser.add_argument('output', help='Path to save the output JSON with centroids')
    parser.add_argument('--colors-json', help='Path to the JSON file with province colors')
    parser.add_argument('--visualize', action='store_true', help='Create a visualization of the centroids')
    parser.add_argument('--display-width', type=int, default=8000, help='Width of the display area (default: 8000)')
    parser.add_argument('--display-height', type=int, default=4000, help='Height of the display area (default: 4000)')
    
    args = parser.parse_args()
    
    # If colors_json is not provided, try to infer it
    colors_json_path = args.colors_json
    if not colors_json_path:
        # Try to infer from input path
        base_dir = os.path.dirname(args.input)
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        colors_json_path = os.path.join(base_dir, f"{base_name}_colors.json")
        print(f"No colors JSON specified, trying: {colors_json_path}")
    
    # Find the centroids with coordinate transformation
    centroids = find_province_centroids(
        args.input, 
        colors_json_path, 
        args.output,
        display_width=args.display_width,
        display_height=args.display_height
    )
    
    # Create a visualization if requested
    if args.visualize:
        vis_path = os.path.splitext(args.output)[0] + "_visualization.png"
        visualize_centroids(
            args.input, 
            centroids, 
            vis_path,
            display_width=args.display_width,
            display_height=args.display_height
        )
    
    print(f"Found centroids for {len(centroids)} provinces")
    print(f"Coordinate system: Origin (0,0) at center, display dimensions {args.display_width}x{args.display_height}")
    print(f"Note: Y-axis is inverted so positive Y is up")

if __name__ == "__main__":
    main()