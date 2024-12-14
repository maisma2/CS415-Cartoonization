import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.cluster.vq import kmeans, vq
import os


def quantize_image(image, n_colors=8):
    """
    Reduces the number of colors in the image using K-means clustering.

    Args:
    image (numpy.ndarray): Input RGB image.
    n_colors (int): Number of colors to quantize to.

    Returns:
    numpy.ndarray: Color-quantized image.
    """
    h, w, c = image.shape
    reshaped_image = image.reshape(-1, c)
    centroids, _ = kmeans(reshaped_image.astype(float), n_colors)
    quantized, _ = vq(reshaped_image, centroids)
    clustered_image = centroids[quantized].reshape(h, w, c).astype(np.uint8)
    return clustered_image


def cartoonize_image(input_path, output_path):
    """
    Cartoonizes an image by applying median filtering, bilateral filtering,
    edge detection, color quantization, and combining layers.

    Args:
    input_path (str): Path to the input image.
    output_path (str): Path to save the cartoonized image.
    """
    # Read the input image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Cannot load image {input_path}")
        return

    # Resize for consistency (optional, based on your image resolution)
    img = cv2.resize(img, (800, 800))

    # Step 1: Apply Gaussian Filter for noise reduction (SciPy)
    smoothed = gaussian_filter(img, sigma=1)

    # Step 2: Convert to Grayscale
    gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)

    # Step 3: Detect edges using Canny edge detection
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Step 4: Quantize colors in the image using K-means (SciPy + NumPy)
    quantized_colors = quantize_image(smoothed)

    # Step 5: Combine edge layer with quantized color layer
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(quantized_colors, edges_colored)

    # Save the cartoonized image
    cv2.imwrite(output_path, cartoon)
    print(f"Cartoonized image saved at: {output_path}")


def process_directory(input_dir, output_dir):
    """
    Processes all images in the input directory and saves cartoonized images
    to the output directory.

    Args:
    input_dir (str): Directory containing the input images.
    output_dir (str): Directory to save the cartoonized images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        # Ensure we are processing only image files
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            print(f"Processing {file_name}...")
            cartoonize_image(input_path, output_path)
        else:
            print(f"Skipping non-image file: {file_name}")


if __name__ == "__main__":
    # Directories for input and output images
    input_directory = "base-images"
    output_directory = "cartooned-images\bw"

    # Process all images in the input directory
    process_directory(input_directory, output_directory)
