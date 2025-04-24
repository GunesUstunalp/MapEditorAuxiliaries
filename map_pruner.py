import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
import argparse

def load_map(filepath):
    """Load a map image as a binary array."""
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image from {filepath}")
    
    # Ensure binary (255 for lines, 0 for background)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Convert to 0 and 1 format for easier processing
    binary = binary // 255
    
    # Print some stats about the image
    print(f"Image loaded: shape={img.shape}, min={np.min(img)}, max={np.max(img)}")
    print(f"Unique values in binary image: {np.unique(binary)}")
    print(f"Number of white pixels: {np.sum(binary)}")
    
    return binary

def identify_junctions_and_endpoints(binary_map):
    """
    Identify junctions (3+ neighbors) and endpoints (1 neighbor) in a more robust way.
    Handles maps with unusual formats.
    """
    # Ensure the map is 0 and 1
    binary_map = binary_map.astype(np.uint8)
    
    # Use a kernel for neighbor counting
    kernel = np.array([[1, 1, 1], 
                       [1, 0, 1], 
                       [1, 1, 1]], dtype=np.uint8)
    
    # Count neighbors for each pixel
    neighbors = cv2.filter2D(binary_map, -1, kernel)
    
    # Only consider pixels that are part of the map
    mask = binary_map > 0
    
    # Display neighbor count distribution
    unique_counts, count_freq = np.unique(neighbors[mask], return_counts=True)
    print("Neighbor count distribution:")
    for count, freq in zip(unique_counts, count_freq):
        print(f"  {count} neighbors: {freq} pixels")
    
    # Junction points have 3 or more neighbors
    junctions = (binary_map > 0) & (neighbors >= 3)
    
    # Endpoints have exactly 1 neighbor
    endpoints = (binary_map > 0) & (neighbors == 1)
    
    print(f"Found {np.sum(junctions)} junction points and {np.sum(endpoints)} endpoints")
    
    return junctions, endpoints

def find_stubs_and_t_junctions(binary_map, max_stub_length=10):
    """
    Find stubs and T-junctions in the map using a more robust approach.
    """
    print(f"Finding stubs and T-junctions with max length of {max_stub_length}...")
    
    # Create output maps
    pruned_map = binary_map.copy()
    detection_map = np.zeros_like(binary_map)
    detection_map[binary_map > 0] = 1  # Mark all path pixels as white
    
    # Find junctions and endpoints
    junctions, endpoints = identify_junctions_and_endpoints(binary_map)
    
    # If we found no endpoints, the map might be inverted or have other issues
    if np.sum(endpoints) == 0:
        print("WARNING: No endpoints found. The map might be inverted or have pixel connectivity issues.")
        print("Trying to invert the map...")
        inverted_map = 1 - binary_map
        junctions_inv, endpoints_inv = identify_junctions_and_endpoints(inverted_map)
        
        if np.sum(endpoints_inv) > 0:
            print(f"Inverted map has {np.sum(endpoints_inv)} endpoints. Using inverted map.")
            binary_map = inverted_map
            pruned_map = binary_map.copy()
            detection_map = np.zeros_like(binary_map)
            detection_map[binary_map > 0] = 1
            junctions, endpoints = junctions_inv, endpoints_inv
    
    # Mark junction points and endpoints
    detection_map[junctions] = 4  # Blue - junctions
    detection_map[endpoints] = 5  # Yellow - endpoints
    
    # Find stubs (paths from endpoints to junctions)
    stubs_found = 0
    
    # Use connected component analysis to identify isolated and connected components
    # This is more robust than the previous pixel-by-pixel tracing
    print("Analyzing map components...")
    
    # Create a skeleton to ensure 1-pixel wide paths
    # Check if OpenCV contrib modules are available
    try:
        skeleton = cv2.ximgproc.thinning(binary_map.astype(np.uint8) * 255)
        skeleton = skeleton // 255  # Convert back to 0/1
    except:
        print("WARNING: OpenCV contrib modules not available. Using original map.")
        skeleton = binary_map
    
    # For endpoints, check if they're close to junctions
    endpoint_coords = np.where(endpoints)
    for i in tqdm(range(len(endpoint_coords[0])), desc="Analyzing endpoints"):
        y, x = endpoint_coords[0][i], endpoint_coords[1][i]
        
        # Use breadth-first search to trace path until junction or max length
        queue = [(y, x, 0)]  # (y, x, distance)
        visited = np.zeros_like(binary_map, dtype=bool)
        visited[y, x] = True
        path = [(y, x)]
        junction_found = False
        
        while queue and queue[0][2] <= max_stub_length:
            cy, cx, dist = queue.pop(0)
            
            # Check neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    
                    ny, nx = cy + dy, cx + dx
                    
                    # Skip if out of bounds
                    if ny < 0 or ny >= binary_map.shape[0] or nx < 0 or nx >= binary_map.shape[1]:
                        continue
                    
                    # Skip if already visited or not part of the map
                    if visited[ny, nx] or binary_map[ny, nx] == 0:
                        continue
                    
                    # Check if it's a junction
                    if junctions[ny, nx]:
                        path.append((ny, nx))
                        junction_found = True
                        break
                    
                    # Add to queue and path
                    queue.append((ny, nx, dist + 1))
                    visited[ny, nx] = True
                    path.append((ny, nx))
                
                if junction_found:
                    break
            
            if junction_found:
                break
        
        # If we found a path to a junction and it's short enough
        if junction_found and len(path) <= max_stub_length:
            # Mark as stub in detection map (except the junction point)
            for py, px in path[:-1]:  # Skip the last one (junction)
                detection_map[py, px] = 2  # Red - stub
                pruned_map[py, px] = 0  # Remove from pruned map
            
            stubs_found += 1
    
    # Find T-junctions
    t_junctions_found = 0
    
    # For each junction point
    junction_coords = np.where(junctions)
    for i in tqdm(range(len(junction_coords[0])), desc="Analyzing junctions"):
        y, x = junction_coords[0][i], junction_coords[1][i]
        
        # Skip if already processed
        if pruned_map[y, x] == 0:
            continue
        
        # For each connected branch, trace until another junction, endpoint, or max length
        # First, find the directions of connected branches
        branches = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                
                ny, nx = y + dy, x + dx
                
                # Skip if out of bounds
                if ny < 0 or ny >= binary_map.shape[0] or nx < 0 or nx >= binary_map.shape[1]:
                    continue
                
                # Skip if not part of the map
                if binary_map[ny, nx] == 0:
                    continue
                
                branches.append((ny, nx))
        
        # If this is a T-junction (3+ branches)
        if len(branches) >= 3:
            # For each branch, check if it's a stub
            for start_y, start_x in branches:
                # Use BFS to trace the branch
                queue = [(start_y, start_x, 1)]  # (y, x, distance)
                visited = np.zeros_like(binary_map, dtype=bool)
                visited[y, x] = True  # Mark the original junction
                visited[start_y, start_x] = True
                path = [(start_y, start_x)]
                connected_to_other_junction = False
                connected_to_endpoint = False
                
                while queue and queue[0][2] <= max_stub_length:
                    cy, cx, dist = queue.pop(0)
                    
                    # Check neighbors
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            
                            ny, nx = cy + dy, cx + dx
                            
                            # Skip if out of bounds
                            if ny < 0 or ny >= binary_map.shape[0] or nx < 0 or nx >= binary_map.shape[1]:
                                continue
                            
                            # Skip if already visited or not part of the map
                            if visited[ny, nx] or binary_map[ny, nx] == 0:
                                continue
                            
                            # Check if it's another junction
                            if junctions[ny, nx] and (ny != y or nx != x):
                                connected_to_other_junction = True
                                break
                            
                            # Check if it's an endpoint
                            if endpoints[ny, nx]:
                                connected_to_endpoint = True
                                break
                            
                            # Add to queue and path
                            queue.append((ny, nx, dist + 1))
                            visited[ny, nx] = True
                            path.append((ny, nx))
                        
                        if connected_to_other_junction or connected_to_endpoint:
                            break
                    
                    if connected_to_other_junction or connected_to_endpoint:
                        break
                
                # If this branch is a stub (short and not connected to another junction)
                if not connected_to_other_junction and len(path) <= max_stub_length:
                    # Mark as T-junction in detection map
                    for py, px in path:
                        detection_map[py, px] = 3  # Green - T-junction
                        pruned_map[py, px] = 0  # Remove from pruned map
                    
                    t_junctions_found += 1
    
    print(f"Found {stubs_found} stubs and {t_junctions_found} T-junctions")
    
    return pruned_map, detection_map

def create_colored_overlay(detection_map, binary_map):
    """Create a colored overlay to visualize detections."""
    # Create an RGB image
    h, w = detection_map.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Set base map as grayscale
    overlay[binary_map == 1] = [100, 100, 100]  # Gray for original map
    
    # Add colored markers
    overlay[detection_map == 2] = [255, 0, 0]    # Red for stubs
    overlay[detection_map == 3] = [0, 255, 0]    # Green for T-junctions
    overlay[detection_map == 4] = [0, 0, 255]    # Blue for junction points
    overlay[detection_map == 5] = [255, 255, 0]  # Yellow for endpoints
    
    return overlay

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Prune map by removing stubs and T-junctions')
    parser.add_argument('input', help='Path to input map image')
    parser.add_argument('output', help='Path to output pruned map image')
    parser.add_argument('--stub-length', type=int, default=20, help='Maximum stub length to remove')
    args = parser.parse_args()
    
    start_time = time.time()
    print("Starting map analysis process...")
    
    # Use the specified file paths
    input_path = args.input
    output_path = args.output
    overlay_path = os.path.join(os.path.dirname(output_path), "detection_overlay.png")
    max_stub_length = args.stub_length
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if overlay_path:
        os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
    
    try:
        print(f"Loading map from {input_path}...")
        binary_map = load_map(input_path)
    except Exception as e:
        print(f"Error loading map: {e}")
        return
    
    # If the map is all white or all black, something's wrong
    if np.all(binary_map == 0) or np.all(binary_map == 1):
        print("WARNING: The map appears to be all black or all white.")
        print("This may cause the analysis to fail.")
    
    # Find stubs and T-junctions
    pruned_map, detection_map = find_stubs_and_t_junctions(binary_map, max_stub_length=max_stub_length)
    
    # Create and save the colored overlay
    if overlay_path:
        overlay = create_colored_overlay(detection_map, binary_map)
        cv2.imwrite(overlay_path, overlay)
        print(f"Detection overlay saved as '{overlay_path}'")
    
    # Save the pruned map - FIXED: Invert the binary image before saving
    # This ensures white background (255) with black boundaries (0)
    pruned_map_output = (1 - pruned_map) * 255
    cv2.imwrite(output_path, pruned_map_output)
    print(f"Pruned map saved as '{output_path}'")
    
    # Count detections
    num_stubs = np.sum(detection_map == 2)
    num_t_junctions = np.sum(detection_map == 3)
    num_junctions = np.sum(detection_map == 4)
    num_endpoints = np.sum(detection_map == 5)
    
    print(f"\nDetection Results:")
    print(f"- {num_junctions} junction points (blue)")
    print(f"- {num_endpoints} endpoints (yellow)")
    print(f"- {num_stubs} stub pixels detected (red)")
    print(f"- {num_t_junctions} T-junction pixels detected (green)")
    
    print(f"\nTotal processing time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()