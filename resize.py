import os
import cv2

# Define paths
input_folder = 'Men-Face'
output_folder = 'Men-Face-Processed'

# Create output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get list of all image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]


# Loop through all image files
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    
    # Read the image
    image = cv2.imread(image_path)
    
    # Resize the image to 128x128
    resized_image = cv2.resize(image, (128, 128))
    
    # Define the path to save the resized image
    output_path = os.path.join(output_folder, f'{os.path.splitext(image_file)[0]}.jpg')
    
    # Save the resized image
    cv2.imwrite(output_path, resized_image)

print("Resizing completed.")
