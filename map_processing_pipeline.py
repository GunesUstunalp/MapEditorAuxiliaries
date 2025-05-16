#!/usr/bin/env python3
"""
Map Processing Pipeline

This script orchestrates the execution of multiple map processing scripts in sequence:
1. extract_black_boundaries.py - Extracts boundaries from a hand-drawn map
2. boundary_thinner.py - Thins the boundaries
3. map_pruner.py - Prunes/cleans up the map
4. color_provinces.py - Colors the provinces/regions
5. boundary_remover.py - Removes the boundaries and assigns each pixel to a province
6. province_centroid_finder.py - Finds the center point of each province
7. province_adjacency_finder.py - Finds which provinces are adjacent to each other
8. province_json_converter.py - Converts color data and centroid data to province data format

Intermediate images are saved for debugging purposes.
"""

import os
import sys
import subprocess
import shutil
import datetime
from PIL import Image

def ensure_dir(directory):
    """Ensure that a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_script(script_name, input_file, output_file, args=None):
    """Run a Python script with the specified input and output files."""
    print(f"Running {script_name}...")
    
    cmd = [sys.executable, script_name, input_file, output_file]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  {script_name} completed successfully")
        print(f"  Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error running {script_name}: {e}")
        print(f"  Error output: {e.stderr.strip()}")
        return False

def copy_debug_image(src_file, debug_dir, stage_name):
    """Copy an image to the debug directory with an appropriate name."""
    _, ext = os.path.splitext(src_file)
    debug_file = os.path.join(debug_dir, f"{stage_name}{ext}")
    shutil.copy2(src_file, debug_file)
    print(f"  Debug image saved to {debug_file}")

def main():
    # Check if input file was provided
    if len(sys.argv) < 2:
        print("Usage: python map_processing_pipeline.py <input_image> [output_directory]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Set output directory (default is current directory)
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    
    # Create timestamp for debug directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = os.path.join(output_dir, f"debug_{timestamp}")
    
    # Ensure directories exist
    ensure_dir(output_dir)
    ensure_dir(debug_dir)
    
    # Copy original input to debug directory
    copy_debug_image(input_file, debug_dir, "00_original")
    
    # Define intermediate and final filenames
    stage1_output = os.path.join(output_dir, "temp_boundaries.png")
    stage2_output = os.path.join(output_dir, "temp_thinned.png")
    stage3_output = os.path.join(output_dir, "temp_pruned.png")
    stage4_output = os.path.join(output_dir, "temp_colored_map.png")
    colors_json = os.path.join(output_dir, "temp_colored_map_colors.json")
    stage5_output = os.path.join(output_dir, "final_map_no_boundaries.png")
    centroids_json = os.path.join(output_dir, "province_centroids.json")
    adjacencies_json = os.path.join(output_dir, "province_adjacencies.json")
    provinces_json = os.path.join(output_dir, "provinces.json")
    centroid_vis = os.path.join(output_dir, "province_centroids_visualization.png")
    adjacency_vis = os.path.join(output_dir, "province_adjacencies_visualization.png")
    
    # Pipeline stages
    stages = [
        {
            "script": "extract_black_boundaries.py",
            "input": input_file,
            "output": stage1_output,
            "debug_name": "01_boundaries",
            "args": []
        },
        {
            "script": "boundary_thinner.py",
            "input": stage1_output,
            "output": stage2_output,
            "debug_name": "02_thinned",
            "args": []
        },
        {
            "script": "map_pruner.py",
            "input": stage2_output,
            "output": stage3_output,
            "debug_name": "03_pruned",
            "args": []
        },
        {
            "script": "color_provinces.py",
            "input": stage3_output,
            "output": stage4_output,
            "debug_name": "04_colored",
            "args": []
        },
        {
            "script": "boundary_remover.py",
            "input": stage4_output,
            "output": stage5_output,
            "debug_name": "05_no_boundaries",
            "args": ["--colors-json", colors_json]
        },
        {
            "script": "province_centroid_finder.py",
            "input": stage5_output,
            "output": centroids_json,
            "debug_name": "06_centroids",
            "args": ["--colors-json", colors_json, "--visualize", "--display-width", "8000", "--display-height", "4000"]
        },
        {
            "script": "province_adjacency_finder.py",
            "input": stage5_output,
            "output": adjacencies_json,
            "debug_name": "07_adjacencies",
            "args": ["--centroids", centroids_json]
        },
        {
            "script": "province_json_converter.py",
            "input": colors_json,
            "output": provinces_json,
            "debug_name": "08_json_conversion",
            "args": ["--centroids", centroids_json]
        }
    ]
    
    # Process each stage
    for stage in stages:
        success = run_script(
            stage["script"], 
            stage["input"], 
            stage["output"], 
            stage["args"]
        )
        
        if not success:
            print(f"Pipeline failed at {stage['script']}. Exiting.")
            sys.exit(1)
        
        # Handle different types of output files for debug
        if stage["debug_name"] in ["06_centroids", "07_adjacencies", "08_json_conversion"]:
            # For JSON files, copy directly
            if stage["output"].endswith(".json"):
                debug_file = os.path.join(debug_dir, f"{stage['debug_name']}.json")
                shutil.copy2(stage["output"], debug_file)
                print(f"  Debug JSON saved to {debug_file}")
            
            # For visualizations
            if stage["debug_name"] == "06_centroids" and os.path.exists(centroid_vis):
                copy_debug_image(centroid_vis, debug_dir, "06_centroids_visualization")
        else:
            # For image files
            copy_debug_image(stage["output"], debug_dir, stage["debug_name"])
    
    # Copy provinces.json to debug directory
    debug_provinces_json = os.path.join(debug_dir, "provinces.json")
    shutil.copy2(provinces_json, debug_provinces_json)
    
    # Clean up temporary files
    for temp_file in [stage1_output, stage2_output, stage3_output, stage4_output]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print("\nPipeline completed successfully!")
    print(f"Final map (no boundaries) saved to: {stage5_output}")
    print(f"Province centroids saved to: {centroids_json}")
    print(f"Province adjacencies saved to: {adjacencies_json}")
    print(f"Province data with centroids saved to: {provinces_json}")
    print(f"Centroid visualization saved to: {centroid_vis}")
    print(f"Adjacency visualization saved to: {adjacency_vis}")
    print(f"Debug images and files saved to: {debug_dir}")
    
    # Display summary of the processing stages
    try:
        with open(os.path.join(debug_dir, "processing_summary.txt"), "w") as f:
            f.write("Map Processing Pipeline Summary\n")
            f.write("==============================\n")
            f.write(f"Original input: {input_file}\n")
            f.write(f"Final map (no boundaries): {stage5_output}\n")
            f.write(f"Province centroids: {centroids_json}\n")
            f.write(f"Province adjacencies: {adjacencies_json}\n")
            f.write(f"Province data: {provinces_json}\n")
            f.write(f"Processing time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Processing Stages:\n")
            for i, stage in enumerate(stages, 1):
                f.write(f"{i}. {stage['script']}\n")
            
            print(f"Processing summary saved to {os.path.join(debug_dir, 'processing_summary.txt')}")
    except Exception as e:
        print(f"Error creating processing summary: {e}")

if __name__ == "__main__":
    main()