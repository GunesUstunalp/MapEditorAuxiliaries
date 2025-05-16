#!/usr/bin/env python3
"""
Province Adjacency Finder (Enhanced)

This script analyzes a map image where provinces are represented by unique colors
and finds which provinces are adjacent to each other. The result is saved as a JSON file
containing the adjacency information for each province.

This enhanced version includes:
- Improved JSON format with proper RGB object representation
- Better progress reporting
- Support for visualization of adjacencies
- Noise filtering and border detection

Usage:
    python province_adjacency_finder_enhanced.py <input_map.jpg> <output_adjacency.json>

The input map should have provinces colored with unique colors and no boundaries.
"""

import cv2
import numpy as np
import json
import os
import argparse
import time
from collections import defaultdict
import colorsys


def rgb_to_hex(color):
    """Convert RGB tuple to hex string."""
    return f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"


def find_province_adjacencies(input_path, output_path, min_shared_pixels=5):
    """
    Find all adjacent provinces in a colored map.
    
    Args:
        input_path (str): Path to the map image
        output_path (str): Path for the output JSON file
        min_shared_pixels (int): Minimum number of shared boundary pixels to consider provinces adjacent
                                (helps filter out noise and artifacts)
    
    Returns:
        dict: Adjacency information for each province color
    """
    print(f"Loading map from {input_path}...")
    start_time = time.time()
    
    # Load the image
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image from {input_path}")
    
    # Get dimensions
    height, width, _ = img.shape
    print(f"Map dimensions: {width}x{height}")
    
    # Identify all unique colors (provinces)
    print("Identifying unique provinces...")
    colors = {}
    for y in range(height):
        for x in range(width):
            color = tuple(map(int, img[y, x]))
            if color not in colors:
                colors[color] = {
                    "bgr": color,
                    "hex": rgb_to_hex(color),
                    "rgb": (color[2], color[1], color[0]),
                    "pixels": 0
                }
            colors[color]["pixels"] += 1
    
    print(f"Found {len(colors)} unique colors (provinces)")
    
    # Initialize adjacency count dictionary
    # This will track how many pixels of each color are adjacent to each other color
    adjacency_counts = defaultdict(lambda: defaultdict(int))
    
    # Define directions for neighbor checking (8-connectivity for more robustness)
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    
    # Check for adjacencies
    print("Analyzing province adjacencies...")
    progress_total = height
    
    # Process row by row for better memory efficiency
    for y in range(height):
        # Show progress
        if y % max(1, height // 100) == 0:
            progress = (y / progress_total) * 100
            elapsed = time.time() - start_time
            print(f"Progress: {progress:.1f}% - Elapsed time: {elapsed:.1f} seconds", end="\r")
        
        for x in range(width):
            color = tuple(map(int, img[y, x]))
            
            # Check all neighbors
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                
                # Skip if out of bounds
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                
                # Get neighbor color
                neighbor_color = tuple(map(int, img[ny, nx]))
                
                # If colors are different, they are adjacent
                if neighbor_color != color:
                    # Increment the adjacency count
                    adjacency_counts[color][neighbor_color] += 1
    
    print("\nProcessing adjacency data...")
    
    # Create final adjacency structure with improved format
    adjacency_dict = {}
    
    for color, color_info in colors.items():
        color_hex = color_info["hex"]
        
        # Get all adjacent colors with enough shared pixels
        adjacent_colors = []
        for adj_color, count in adjacency_counts[color].items():
            if count >= min_shared_pixels:
                # Add the adjacent color with its information
                adjacent_colors.append({
                    "bgr": list(adj_color),
                    "rgb": [adj_color[2], adj_color[1], adj_color[0]],
                    "hex": rgb_to_hex(adj_color),
                    "shared_pixels": count
                })
        
        # Sort adjacent colors by number of shared pixels (descending)
        adjacent_colors.sort(key=lambda x: x["shared_pixels"], reverse=True)
        
        # Create complete province info
        adjacency_dict[color_hex] = {
            "color": {
                "bgr": list(color),
                "rgb": color_info["rgb"],
                "hex": color_hex
            },
            "pixel_count": color_info["pixels"],
            "adjacent_provinces": adjacent_colors
        }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(adjacency_dict, f, indent=2)
    
    print(f"Adjacency information saved to {output_path}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    
    return adjacency_dict


def create_adjacency_visualization(input_path, adjacency_dict, output_path):
    """
    Create a visualization of the province adjacencies by highlighting borders.
    
    Args:
        input_path (str): Path to the original map image
        adjacency_dict (dict): Adjacency information for each province
        output_path (str): Path for the output visualization image
    """
    print("Creating adjacency visualization...")
    
    # Load the original map
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image from {input_path}")
    
    # Create a copy for visualization
    vis_img = img.copy()
    
    # Create a black border image
    border_img = np.zeros_like(img)
    
    # For each color and its adjacencies
    for _, province_info in adjacency_dict.items():
        # Get color BGR values
        bgr = province_info["color"]["bgr"]
        
        # Create a mask for this color
        mask = np.all(img == bgr, axis=2)
        
        # Find the border pixels by eroding the mask and finding the difference
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        border = mask.astype(np.uint8) - eroded
        
        # Set border pixels to white in the border image
        border_img[border == 1] = [255, 255, 255]
    
    # Overlay the border on the original image
    alpha = 0.7  # Original image weight
    beta = 0.3   # Border weight
    gamma = 0    # Scalar added to each sum
    
    # Apply weighted addition
    vis_img = cv2.addWeighted(vis_img, alpha, border_img, beta, gamma)
    
    # Save the visualization
    cv2.imwrite(output_path, vis_img)
    print(f"Visualization saved to {output_path}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Find province adjacencies in a colored map')
    parser.add_argument('input', help='Path to the map image')
    parser.add_argument('output', help='Path for the output JSON file')
    parser.add_argument('--min-shared-pixels', type=int, default=5,
                       help='Minimum shared boundary pixels to consider provinces adjacent (default: 5)')
    parser.add_argument('--create-visualization', action='store_true',
                       help='Create a visualization of the province adjacencies')
    parser.add_argument('--visualization-output', type=str, default=None,
                       help='Path for the visualization output (default: <output_basename>_visualization.png)')
    
    args = parser.parse_args()
    
    try:
        # Find province adjacencies
        adjacency_dict = find_province_adjacencies(
            args.input, 
            args.output,
            min_shared_pixels=args.min_shared_pixels
        )
        
        # Create visualization if requested
        if args.create_visualization:
            # Determine visualization output path
            vis_output = args.visualization_output
            if vis_output is None:
                base_name = os.path.splitext(args.output)[0]
                vis_output = f"{base_name}_visualization.png"
            
            create_adjacency_visualization(args.input, adjacency_dict, vis_output)
        
        print("\nProvince adjacency analysis completed successfully!")
        print(f"Found {len(adjacency_dict)} provinces with their adjacencies.")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
