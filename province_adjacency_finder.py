#!/usr/bin/env python3
"""
Province Adjacency Finder

This script analyzes a map image where provinces are represented by unique colors
and finds which provinces are adjacent to each other. It also calculates distances
between province centroids provided in a separate JSON file.

Usage:
    python province_adjacency_finder.py <input_map.jpg> <output_adjacency.json> --centroids <centroids.json>

The input map should have provinces colored with unique colors and no boundaries.
The centroids JSON file should contain the centroid coordinates for each province color.
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


def find_province_adjacencies(input_path, output_path, min_shared_pixels=1):
    """
    Find all adjacent provinces in a colored map and calculate distances between centroids.
    
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
    
    # Identify all unique colors (provinces) and collect pixel positions for centroid calculation
    print("Identifying unique provinces...")
    colors = {}
    province_pixels = defaultdict(list)
    
    for y in range(height):
        for x in range(width):
            color = tuple(map(int, img[y, x]))
            if color not in colors:
                colors[color] = {
                    "hex": rgb_to_hex(color)
                }
            # Store pixel position for centroid calculation
            province_pixels[color].append((x, y))
    
    print(f"Found {len(colors)} unique colors (provinces)")
    
    # Calculate centroids for each province
    province_centroids = {}
    for color, pixels in province_pixels.items():
        if pixels:
            # Calculate average x and y coordinates
            x_sum = sum(p[0] for p in pixels)
            y_sum = sum(p[1] for p in pixels)
            centroid = (x_sum / len(pixels), y_sum / len(pixels))
            province_centroids[color] = centroid
    
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
    
    print("\nProcessing adjacency data and calculating distances...")
    
    # Create final adjacency structure with improved format
    adjacency_dict = {}
    
    for color, color_info in colors.items():
        color_hex = color_info["hex"]
        
        # Get all adjacent colors with enough shared pixels
        adjacent_info = {}
        for adj_color, count in adjacency_counts[color].items():
            if count >= 1:
                # Add the hex color with distance information
                adj_color_hex = rgb_to_hex(adj_color)
                
                # Calculate Euclidean distance between centroids
                distance = 0
                if color in province_centroids and adj_color in province_centroids:
                    c1 = province_centroids[color]
                    c2 = province_centroids[adj_color]
                    distance = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                
                adjacent_info[adj_color_hex] = {
                    "shared_pixels": count,
                    "centroid_distance": distance
                }
        
        # Create simplified province info
        adjacency_dict[color_hex] = adjacent_info
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(adjacency_dict, f, indent=2)
    
    print(f"Adjacency information saved to {output_path}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    
    return adjacency_dict
    
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
    parser.add_argument('--centroids', required=True, help='Path to the JSON file containing province centroids')
    
    args = parser.parse_args()
    
    try:
        # Find province adjacencies
        adjacency_dict = find_province_adjacencies(
            args.input, 
            args.output,
            args.centroids
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