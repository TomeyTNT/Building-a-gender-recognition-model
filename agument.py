import os
import cv2
import albumentations as A

# Define paths
input_folder = 'Men-Face'
output_folder = 'Men-Face-Processed'

# Create output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get list of all image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Define augmentation transformations
transform = A.Compose([
    A.Resize(128, 128),
    A.SomeOf([
        A.ToGray(p=1),
        A.ColorJitter(p=1),
        A.GaussNoise(p=1),
        A.MotionBlur(p=1),
        A.RandomBrightnessContrast(p=1),
    ], n=2)  # Apply 2 random transformations from the list
])

# Loop through all image files
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    
    # Read the image
    image = cv2.imread(image_path)
    
    # Apply augmentation
    augmented = transform(image=image)
    augmented_image = augmented['image']
    
    # Define the path to save the augmented image
    augmented_output_path = os.path.join(output_folder, f'{os.path.splitext(image_file)[0]}_aug.jpg')
    
    # Save the augmented image
    cv2.imwrite(augmented_output_path, augmented_image)

print("augmentation completed.")
