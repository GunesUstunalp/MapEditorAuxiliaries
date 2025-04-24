import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import img_as_bool
import os
import argparse

def thin_boundaries(input_path, output_path=None, show_result=False):
    """
    Process an image to thin its boundaries using skeletonization.
    
    Args:
        input_path (str): Path to the input image
        output_path (str, optional): Path to save the output image. If None, derived from input path.
        show_result (bool): Whether to display the result
    
    Returns:
        numpy.ndarray: The processed image with thinned boundaries
    """
    # Load the image
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image from {input_path}")
    
    # Ensure the image is binary (black and white)
    # Invert if needed so lines are white (255) on black background (0)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Convert to boolean image as required by skeletonize
    bool_image = img_as_bool(binary)
    
    # Perform skeletonization
    skeleton = skeletonize(bool_image)
    
    # Convert back to uint8 format for saving
    skeleton_image = np.uint8(skeleton * 255)
    
    # Invert back if needed
    result = cv2.bitwise_not(skeleton_image)
    
    # Determine output path if not provided
    if output_path is None:
        directory = os.path.dirname(input_path)
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(directory, f"{name}_thinned.png")
    
    # Save the result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result)
    print(f"Saved thinned image to {output_path}")
    
    # Display the results if requested
    if show_result:
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(result, cmap='gray')
        plt.title('Thinned Image')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return result

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Thin boundaries in an image using skeletonization')
    parser.add_argument('input', help='Path to the input image')
    parser.add_argument('-o', '--output', help='Path to save the output image')
    parser.add_argument('-s', '--show', action='store_true', help='Display the result')
    
    args = parser.parse_args()
    
    # Process the image
    thin_boundaries(args.input, args.output, args.show)

if __name__ == "__main__":
    main()