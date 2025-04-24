#!/usr/bin/env python3
"""
Map Processing Pipeline

This script orchestrates the execution of multiple map processing scripts in sequence:
1. extract_black_boundaries.py - Extracts boundaries from a hand-drawn map
2. boundary_thinner.py - Thins the boundaries
3. map_pruner.py - Prunes/cleans up the map
4. color_provinces.py - Colors the provinces/regions

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
    final_output = os.path.join(output_dir, "final_colored_map.png")
    
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
            "output": final_output,
            "debug_name": "04_colored",
            "args": []
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
        
        copy_debug_image(stage["output"], debug_dir, stage["debug_name"])
    
    # Clean up temporary files
    for temp_file in [stage1_output, stage2_output, stage3_output]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print("\nPipeline completed successfully!")
    print(f"Final output saved to: {final_output}")
    print(f"Debug images saved to: {debug_dir}")
    
    # Display summary of the processing stages
    try:
        with open(os.path.join(debug_dir, "processing_summary.txt"), "w") as f:
            f.write("Map Processing Pipeline Summary\n")
            f.write("==============================\n")
            f.write(f"Original input: {input_file}\n")
            f.write(f"Final output: {final_output}\n")
            f.write(f"Processing time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Processing Stages:\n")
            for i, stage in enumerate(stages, 1):
                f.write(f"{i}. {stage['script']}\n")
            
            print(f"Processing summary saved to {os.path.join(debug_dir, 'processing_summary.txt')}")
    except Exception as e:
        print(f"Error creating processing summary: {e}")

if __name__ == "__main__":
    main()