import json
import os
import argparse
import sys

def rgb_to_hex(color):
    """
    Convert RGB color (BGR in OpenCV format) to hex format without the # prefix.
    
    Args:
        color (list): [B, G, R] color values from 0-255
        
    Returns:
        str: Hex color string (uppercase)
    """
    # OpenCV uses BGR order, so we need to reverse to RGB for hex conversion
    r, g, b = color[2], color[1], color[0]
    return f"{r:02X}{g:02X}{b:02X}"

def hex_with_hash(hex_color):
    """Ensure hex color has # prefix"""
    if hex_color.startswith('#'):
        return hex_color
    return f"#{hex_color}"

def hex_without_hash(hex_color):
    """Ensure hex color does not have # prefix"""
    if hex_color.startswith('#'):
        return hex_color[1:]
    return hex_color

def convert_to_provinces_json(colors_json_path, centroids_json_path, adjacencies_json_path, output_path):
    """
    Convert color information, centroid data, and adjacency data to a provinces JSON format.
    
    Args:
        colors_json_path (str): Path to the JSON file with province colors
        centroids_json_path (str): Path to the JSON file with province centroids
        adjacencies_json_path (str): Path to the JSON file with province adjacencies
        output_path (str): Path to save the output provinces JSON
    """
    print(f"\nLoading color data from {colors_json_path}...")
    with open(colors_json_path, 'r') as f:
        colors = json.load(f)
    
    print(f"Loading centroid data from {centroids_json_path}...")
    with open(centroids_json_path, 'r') as f:
        centroids = json.load(f)
    
    print(f"Loading adjacency data from {adjacencies_json_path}...")
    with open(adjacencies_json_path, 'r') as f:
        adjacencies = json.load(f)
    
    # Create provinces list
    provinces = []
    
    # Create several mappings to handle different formats of colors
    province_id_by_color_hex = {}  # Maps hex color (with and without #) to province id
    province_by_id = {}            # Maps province id to province object
    
    print(f"\nFound {len(colors)} colors in colors.json")
    print(f"Found {len(centroids)} colors in centroids.json")
    print(f"Found {len(adjacencies)} colors in adjacencies.json")
    
    # First pass: Create provinces with basic data and build mappings
    for i, color in enumerate(colors, 1):
        province_id = i
        province_name = f"P{i}"
        
        # Convert color to different formats for robustness
        color_hex_no_hash = rgb_to_hex(color)
        color_hex_with_hash = f"#{color_hex_no_hash}"
        color_tuple_str = str(tuple(color))
        
        # Debug output
        print(f"\nProcessing province {province_name} (ID: {province_id}):")
        print(f"  BGR Color: {color}")
        print(f"  Hex Color (no #): {color_hex_no_hash}")
        print(f"  Hex Color (with #): {color_hex_with_hash}")
        print(f"  Color Tuple Str: {color_tuple_str}")
        
        # Add to mappings (both with and without hash for robustness)
        province_id_by_color_hex[color_hex_no_hash.lower()] = province_id
        province_id_by_color_hex[color_hex_with_hash.lower()] = province_id
        
        # Get centroid for this province, default to [0, 0, 0] if not found
        center_point = centroids.get(color_tuple_str, [0, 0, 0])
        if color_tuple_str in centroids:
            print(f"  Found centroid: {center_point}")
        else:
            print(f"  Warning: No centroid found for color {color_tuple_str}")
        
        # Create province entry in the required format (adjacencies will be added later)
        province = {
            "provinceName": province_name,
            "owner": "DemoCountry",
            "center": {
                "x": center_point[0],
                "y": center_point[2],
                "z": center_point[1]
            },
            "baseTax": 0,
            "baseProduction": 0,
            "baseManpower": 0,
            "color": color_hex_no_hash,
            "adjacencies": []
        }
        
        provinces.append(province)
        province_by_id[province_id] = province
    
    # Debug: Print out a few adjacency entries to understand their format
    print("\nSample adjacency entries:")
    adjacency_count = 0
    for hex_color, adjacent_provinces in list(adjacencies.items())[:3]:
        print(f"  Adjacencies for {hex_color}:")
        for adj_hex, adj_data in adjacent_provinces.items():
            print(f"    - {adj_hex}: {adj_data}")
            adjacency_count += 1
    print(f"Total adjacency relationships: {adjacency_count}")
    
    # Second pass: Add adjacency information
    for hex_color, adjacent_provinces in adjacencies.items():
        # Ensure consistent format for lookup (lowercase and try both with/without #)
        hex_color_lower = hex_color.lower()
        hex_color_no_hash = hex_without_hash(hex_color_lower)
        hex_color_with_hash = hex_with_hash(hex_color_lower)
        
        # Find the province ID for this color
        province_id = None
        if hex_color_no_hash in province_id_by_color_hex:
            province_id = province_id_by_color_hex[hex_color_no_hash]
        elif hex_color_with_hash in province_id_by_color_hex:
            province_id = province_id_by_color_hex[hex_color_with_hash]
        
        if province_id is None:
            print(f"Warning: No province found for color {hex_color}")
            continue
        
        # Get the province object
        province = province_by_id[province_id]
        province_name = province["provinceName"]
        
        print(f"\nProcessing adjacencies for {province_name} (color: {hex_color}):")
        
        # Add each adjacent province
        for adj_hex_color, adj_data in adjacent_provinces.items():
            # Ensure consistent format for lookup
            adj_hex_lower = adj_hex_color.lower()
            adj_hex_no_hash = hex_without_hash(adj_hex_lower)
            adj_hex_with_hash = hex_with_hash(adj_hex_lower)
            
            # Find the adjacent province ID
            adj_province_id = None
            if adj_hex_no_hash in province_id_by_color_hex:
                adj_province_id = province_id_by_color_hex[adj_hex_no_hash]
            elif adj_hex_with_hash in province_id_by_color_hex:
                adj_province_id = province_id_by_color_hex[adj_hex_with_hash]
            
            if adj_province_id is None:
                print(f"  Warning: No province found for adjacent color {adj_hex_color}")
                continue
            
            # Get the adjacent province name
            adj_province_name = f"P{adj_province_id}"
            
            # Add adjacency with distance
            distance = round(adj_data["centroid_distance"], 1)
            province["adjacencies"].append({
                "provinceName": adj_province_name,
                "distance": distance
            })
            
            print(f"  Added adjacency to {adj_province_name} with distance {distance}")
    
    # Format provinces for JSON output
    provinces_formatted = {
        "provinces": provinces
    }

    # Count provinces with adjacencies
    provinces_with_adjacencies = sum(1 for p in provinces if p["adjacencies"])
    print(f"\nProvinces with adjacencies: {provinces_with_adjacencies}/{len(provinces)}")
    
    # Save provinces to JSON
    print(f"Creating provinces JSON with {len(provinces)} provinces...")
    with open(output_path, 'w') as f:
        json.dump(provinces_formatted, f, indent=4)

    print(f"Provinces JSON saved to {output_path}")
    
    # Quick validation - print first few provinces with their adjacencies
    print("\nSample of the generated provinces JSON:")
    for i, province in enumerate(provinces[:3], 1):
        adjacency_count = len(province["adjacencies"])
        print(f"  Province {province['provinceName']} has {adjacency_count} adjacencies:")
        for adj in province["adjacencies"][:3]:  # Show only first 3 adjacencies
            print(f"    - {adj['provinceName']} (distance: {adj['distance']})")
        if adjacency_count > 3:
            print(f"    - ... and {adjacency_count - 3} more")

    return provinces_formatted

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Convert colors, centroids, and adjacencies to provinces JSON')
    parser.add_argument('colors_json', help='Path to the JSON file with province colors')
    parser.add_argument('output_path', help='Path to save the output provinces JSON')
    parser.add_argument('--centroids', help='Path to the JSON file with province centroids')
    parser.add_argument('--adjacencies', help='Path to the JSON file with province adjacencies')
    
    args = parser.parse_args()
    
    # If centroids path is not provided, try to infer it
    centroids_path = args.centroids
    if not centroids_path:
        base_dir = os.path.dirname(args.colors_json)
        base_name = os.path.splitext(os.path.basename(args.colors_json))[0]
        centroids_path = os.path.join(base_dir, "province_centroids.json")
        if not os.path.exists(centroids_path):
            print(f"Error: Centroids JSON file not found at {centroids_path}")
            print("Please provide the path to the centroids JSON file using --centroids option")
            sys.exit(1)
    
    # If adjacencies path is not provided, try to infer it
    adjacencies_path = args.adjacencies
    if not adjacencies_path:
        base_dir = os.path.dirname(args.colors_json)
        adjacencies_path = os.path.join(base_dir, "province_adjacencies.json")
        if not os.path.exists(adjacencies_path):
            print(f"Error: Adjacencies JSON file not found at {adjacencies_path}")
            print("Please provide the path to the adjacencies JSON file using --adjacencies option")
            sys.exit(1)
    
    # Quick test for RGB to Hex conversion
    test_color = [220, 47, 128]  # [B, G, R] -> should be 802FDC
    test_hex = rgb_to_hex(test_color)
    print(f"Color conversion test: RGB(BGR) {test_color} -> HEX {test_hex}")
    
    # Convert to provinces JSON
    convert_to_provinces_json(args.colors_json, centroids_path, adjacencies_path, args.output_path)

if __name__ == "__main__":
    main()