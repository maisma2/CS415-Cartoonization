import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.cluster.vq import kmeans, vq
import os


def quantize_image(image, n_colors=16):
    #Convert BGR to RGB first
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = rgb_image.shape
    reshaped_image = rgb_image.reshape(-1, c).astype(float) / 255.0

    #Use K-means to find color clusters
    centroids, _ = kmeans(reshaped_image, n_colors, iter=20)

    #Assign each pixel to the nearest cluster center
    quantized, _ = vq(reshaped_image, centroids)

    #Rescale the quantized image back to [0, 255] and convert back to BGR
    clustered_image = (centroids[quantized] * 255).reshape(h, w, c).astype(np.uint8)
    return cv2.cvtColor(clustered_image, cv2.COLOR_RGB2BGR)
def cartoonize_image(input_path, output_path):
    """
    "Cartoonifies" an image by applying advanced filtering, balanced color quantization,
    edge detection, and blending.
    """
    #Read each image in the input path
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not load image {input_path}")
        return

    #Resize for consistency
    img = cv2.resize(img, (1200, 1200))

    #Apply Gaussian Filter for noise reduction, this is part of Scipy
    smoothed = gaussian_filter(img, sigma=.5)

    #Convert to Grayscale, for better edge detection
    gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)

    #Detect edges using Canny edge detection
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    #Calling the Quantize function to smoothen out the image
    quantized_colors = quantize_image(smoothed, n_colors=12)  #For testing can be changed to 16

    #Invert edges to white, and background to black for edge detection
    inverted_edges = cv2.bitwise_not(edges)

    #Use the edge mask to overlap with the quantized colors
    mask = cv2.cvtColor(inverted_edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(quantized_colors, mask)

    #Save the "cartoony" image
    cv2.imwrite(output_path, cartoon)
    print(f"Filtered image saved at: {output_path}")


def process_directory(input_dir, output_dir):
    #Loops through each image in a directory and sends it for processing, and saves it
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
    #Hardcoded input and output directories
    input_directory = "base-images"
    output_directory = "cartooned-images\enhanced"

    #Send it over for the remainder of the program
    process_directory(input_directory, output_directory)
