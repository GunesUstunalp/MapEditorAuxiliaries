#!/usr/bin/env python3
"""
Province Adjacency Finder

This script analyzes a map image where provinces are represented by unique colors
and finds which provinces are adjacent to each other. The result is saved as a JSON file
containing the adjacency information for each province.

The output is a simple JSON structure where each key is a hex color and the value
is an array of hex colors representing adjacent provinces.

Usage:
    python province_adjacency_finder.py <input_map.jpg> <output_adjacency.json>

The input map should have provinces colored with unique colors and no boundaries.
"""

import cv2
import numpy as np
import json
import os
import argparse
import time
from collections import defaultdict


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
                    "hex": rgb_to_hex(color)
                }
    
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
        adjacent_hexes = []
        for adj_color, count in adjacency_counts[color].items():
            if count >= min_shared_pixels:
                # Add only the hex color
                adjacent_hexes.append(rgb_to_hex(adj_color))
        
        # Create simplified province info
        adjacency_dict[color_hex] = adjacent_hexes
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(adjacency_dict, f, indent=2)
    
    print(f"Adjacency information saved to {output_path}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    
    return adjacency_dict





def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Find province adjacencies in a colored map')
    parser.add_argument('input', help='Path to the map image')
    parser.add_argument('output', help='Path for the output JSON file')
    parser.add_argument('--min-shared-pixels', type=int, default=5,
                       help='Minimum shared boundary pixels to consider provinces adjacent (default: 5)')
    
    args = parser.parse_args()
    
    try:
        # Find province adjacencies
        adjacency_dict = find_province_adjacencies(
            args.input, 
            args.output,
            min_shared_pixels=args.min_shared_pixels
        )
        
        print("\nProvince adjacency analysis completed successfully!")
        print(f"Found {len(adjacency_dict)} provinces with their adjacencies.")
        print(f"Results saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()