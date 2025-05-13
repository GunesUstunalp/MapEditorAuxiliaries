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
        str: Hex color string
    """
    # OpenCV uses BGR order, so we need to reverse to RGB for hex conversion
    r, g, b = color[2], color[1], color[0]
    return f"{r:02X}{g:02X}{b:02X}"

def convert_to_provinces_json(colors_json_path, centroids_json_path, output_path):
    """
    Convert color information and centroid data to a provinces JSON format.
    
    Args:
        colors_json_path (str): Path to the JSON file with province colors
        centroids_json_path (str): Path to the JSON file with province centroids
        output_path (str): Path to save the output provinces JSON
    """
    print(f"Loading color data from {colors_json_path}...")
    with open(colors_json_path, 'r') as f:
        colors = json.load(f)
    
    print(f"Loading centroid data from {centroids_json_path}...")
    with open(centroids_json_path, 'r') as f:
        centroids = json.load(f)
    
    # Create provinces list
    provinces = []
    
    for i, color in enumerate(colors, 1):
        # Convert color from list to tuple string for lookup in centroids
        color_tuple_str = str(tuple(color))
        
        # Get centroid for this province, default to [0, 0, 0] if not found
        center_point = centroids.get(color_tuple_str, [0, 0, 0])
        
        # Create province entry in the required format
        province = {
            "provinceName": f"P{i}",
            "owner": "DemoCountry",
            "center": {
                "x": center_point[0],
                "y": center_point[1],
                "z": center_point[2]
            },
            "baseTax": 0,
            "baseProduction": 0,
            "baseManpower": 0,
            "color": rgb_to_hex(color),
            "adjacencies": []
        }
        
        provinces.append(province)
    
    # Save provinces to JSON
    print(f"Creating provinces JSON with {len(provinces)} provinces...")
    with open(output_path, 'w') as f:
        json.dump(provinces, f, indent=4)
    
    print(f"Provinces JSON saved to {output_path}")
    return provinces

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Convert colors and centroids to provinces JSON')
    parser.add_argument('colors_json', help='Path to the JSON file with province colors')
    parser.add_argument('output_path', help='Path to save the output provinces JSON')
    parser.add_argument('--centroids', help='Path to the JSON file with province centroids')
    
    args = parser.parse_args()
    
    # If centroids path is not provided, try to infer it
    centroids_path = args.centroids
    if not centroids_path:
        base_dir = os.path.dirname(args.colors_json)
        base_name = os.path.splitext(os.path.basename(args.colors_json))[0]
        centroids_path = os.path.join(base_dir, f"{base_name}_centroids.json")
        if not os.path.exists(centroids_path):
            print(f"Error: Centroids JSON file not found at {centroids_path}")
            print("Please provide the path to the centroids JSON file using --centroids option")
            sys.exit(1)
    
    # Quick test for RGB to Hex conversion
    test_color = [220, 47, 128]  # [B, G, R] -> should be 802FDC
    test_hex = rgb_to_hex(test_color)
    print(f"Color conversion test: RGB(BGR) {test_color} -> HEX {test_hex}")
    
    # Convert to provinces JSON
    convert_to_provinces_json(args.colors_json, centroids_path, args.output_path)

if __name__ == "__main__":
    main()