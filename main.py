import cv2
import os


def cartoonize_image(input_path, output_path):
    """
    "Cartoonifies" an image by applying median filtering then bilateral filtering,
    edge detection, and overlapping edge and color layers.
    """
    #Read the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Cannot load image {input_path}")
        return

    #Resize for consistency
    img = cv2.resize(img, (800, 800))

    #Apply Median Filter to reduce noise
    median_filtered = cv2.medianBlur(img, 5)

    #Convert to Grayscale
    gray = cv2.cvtColor(median_filtered, cv2.COLOR_BGR2GRAY)

    #Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )

    #Apply Bilateral Filter for color smoothing while preserving edges
    color_smooth = cv2.bilateralFilter(median_filtered, 9, 300, 300)

    #Combine edge layer with smoothed color layer
    cartoon = cv2.bitwise_and(color_smooth, color_smooth, mask=edges)

    #Save the image
    cv2.imwrite(output_path, cartoon)
    print(f"Cartoonized image saved at: {output_path}")


def process_directory(input_dir, output_dir):
    #Loops through the directory, sending each image for filtering
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
    #Hardcoded directories
    input_directory = "base-images"
    output_directory = "cartooned-images\basic"

    #Start the program
    process_directory(input_directory, output_directory)
