#!/usr/bin/env python3
"""
Province JSON Converter

This script converts the JSON color file produced by color_provinces.py 
into a structured province data format required by the game engine.

Usage:
    python province_json_converter.py input_colors.json output_provinces.json
"""

import json
import os
import sys
import argparse

def convert_color_to_hex(bgr_color):
    """
    Convert BGR color (OpenCV format) to hex string (RGB format)
    
    Args:
        bgr_color (list): Color in BGR format [B, G, R]
        
    Returns:
        str: Hex color string in RGB format
    """
    # BGR to RGB
    rgb_color = [bgr_color[2], bgr_color[1], bgr_color[0]]
    # Convert to hex
    hex_color = ''.join([f'{c:02X}' for c in rgb_color])
    return hex_color

def create_province_data(colors):
    """
    Create province data in the required format
    
    Args:
        colors (list): List of colors in BGR format
        
    Returns:
        dict: Province data in the required format
    """
    province_data = {
        "provinces": []
    }
    
    for i, bgr_color in enumerate(colors):
        province_name = f"P{i+1}"
        hex_color = convert_color_to_hex(bgr_color)
        
        province = {
            "provinceName": province_name,
            "owner": "DemoCountry",
            "center": {
                "x": 0,
                "y": 0,
                "z": 0
            },
            "baseTax": 0,
            "baseProduction": 0,
            "baseManpower": 0,
            "color": hex_color,
            "adjacencies": []
        }
        
        province_data["provinces"].append(province)
    
    return province_data

def convert_provinces_json(input_json_path, output_json_path):
    """
    Convert the color JSON file to the province data format
    
    Args:
        input_json_path (str): Path to the input color JSON file
        output_json_path (str): Path to save the output province JSON file
    """
    try:
        # Read the input JSON file
        with open(input_json_path, 'r') as f:
            colors = json.load(f)
        
        # Create the province data
        province_data = create_province_data(colors)
        
        # Save the province data
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(province_data, f, indent=4)
        
        print(f"Converted {len(colors)} colors to province data")
        print(f"Province data saved to: {output_json_path}")
        
        return True
    except Exception as e:
        print(f"Error converting province data: {e}")
        return False

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Convert color JSON to province data JSON')
    parser.add_argument('input', help='Path to the input color JSON file')
    parser.add_argument('output', help='Path to save the output province JSON file')
    
    args = parser.parse_args()
    
    # Convert the JSON file
    success = convert_provinces_json(args.input, args.output)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()