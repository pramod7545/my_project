from PIL import Image
import numpy as np
import os

def preprocess_images_in_folder(folder_path, output_folder, kernel_size=3):
  """
  Preprocesses images in a given folder using median filtering.

  Args:
    folder_path: Path to the folder containing images.
    output_folder: Path to the folder where preprocessed images will be saved.
    kernel_size: Kernel size for median filtering (odd integer).
  """

  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
      image_path = os.path.join(folder_path, filename)
      preprocessed_image = preprocess_image(image_path, kernel_size)

      if preprocessed_image is not None:
        output_path = os.path.join(output_folder, filename)
        preprocessed_image = Image.fromarray(preprocessed_image.astype(np.uint8))
        preprocessed_image.save(output_path)
        print(f"Preprocessed image saved as: {output_path}")

def preprocess_image(image_path, kernel_size=3):
  """
  Preprocesses a single image using median filtering.

  Args:
    image_path: Path to the image file.
    kernel_size: Kernel size for median filtering (odd integer).

  Returns:
    Preprocessed image as a NumPy array, or None if an error occurs.
  """

  try:
    image = Image.open(image_path).convert('L')  # Load and convert to grayscale
    image_array = np.array(image)

    # Median filtering implementation
    filtered_image = np.zeros_like(image_array)
    for i in range(kernel_size // 2, image_array.shape[0] - kernel_size // 2):
      for j in range(kernel_size // 2, image_array.shape[1] - kernel_size // 2):
        kernel = image_array[i - kernel_size // 2:i + kernel_size // 2 + 1, 
                            j - kernel_size // 2:j + kernel_size // 2 + 1]
        filtered_image[i, j] = np.median(kernel)

    # Normalize pixel values to [0, 1]
    preprocessed_image = filtered_image / 255.0
    return preprocessed_image

  except Exception as e:
    print(f"Error preprocessing image: {e}")
    return None

# Example usage
input_folder = r'C:\Users\pilli\OneDrive\Pictures\Camera Roll'
output_folder =r'C:\Users\pilli\OneDrive\Desktop\project\photos'
preprocess_images_in_folder(input_folder, output_folder, kernel_size=5)