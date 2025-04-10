import os
import requests
import shutil
from PIL import Image, ImageFile


# Creates required folder structure
def prepare_folders(city):
    folder_names = ["points",
                    "roads",
                    "temp_sv_images",
                    "sv_images",
                    "temp_seg_facades",
                    "seg_facades",
                    "temp_seg_windows",
                    "seg_windows"
                    ]

    for folder_name in folder_names:
      prepare_folder(city, folder_name)


# Creates a folder if it doesn't exist yet
def prepare_folder(city, folder_name):
  dir_path = os.path.join("results", city, folder_name)
  if not os.path.exists(dir_path):
      os.makedirs(dir_path)


# Function adapted from iabrilvzqz
# GitHub: https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/StreetView-NatureVisibility-GSV
# Processes street view imagery (SVI) based on an image URL.
# Returns a list of processed images (1 for normal SVI, 2 for panoramic SVI)
def process_image(image_url, is_panoramic, road_angle):
    # Fetch and process the image
    image = Image.open(requests.get(image_url, stream=True).raw)

    if is_panoramic:
        width, height = image.size

        # Crop out the bottom 20%
        cropped_height = int(height*0.8)
        image = image.crop((0, 0, width, cropped_height))

        # Generate two images looking perpendicular from the road
        left_face, right_face = get_perpendicular_images(image, road_angle)
        images = [left_face, right_face]
    else:
        images = [image]

    return images

# Takes a panoramic SVI and returns two images looking perpendicular from the road
def get_perpendicular_images(image, road_angle):
    width, height = image.size
    eighth_width = int(0.125 * width)

    # We want left and right facing images. Wrap around in case values are out of bounds
    wanted_angles = ((road_angle - 90) % 360, (road_angle + 90) % 360)

    faces = []
    original_image = image.copy()

    # We want 1/8th of the image before and after the wanted angle within the shot (1/4th total)
    for wanted_angle in wanted_angles:
        image = original_image.copy()

        # E.g. if wanted_angle is 10, the wanted shift is to fraction 0.0278  of the image on a 0-1 range
        wanted_fractional_axis = float(wanted_angle)/360.0
        wanted_axis= int(width * wanted_fractional_axis)

        left_max = max(wanted_axis - eighth_width, 0)
        right_max = min(wanted_axis + eighth_width, width)
        perpendicular_face = image.crop((left_max, 0, right_max, height))

        faces.append(perpendicular_face)

    # Return the left and right perpendicular face
    return faces


# Takes an input path and deletes all files inside of that path
def delete_files(path):
  files = os.listdir(path)

  for file_name in files:
    file_path = os.path.join(path, file_name)
    os.truncate(file_path, 0)
    os.remove(file_path)


# Takes a TIF PIL image and returns the number of white pixels in it
def count_white_pixels(image):
  return sum(1 for pixel in image.getdata() if pixel == 255)


# Calculates weighted average ratio (WAR) of two perpendicular looking images
def calculate_WAR(width_A, height_A, ratio_A, width_B, height_B, ratio_B):
  size_A = width_A * height_A
  size_B = width_B * height_B

  # Calculate weighted ratio (WR) for each image
  WR_A = ratio_A * size_A
  WR_B = ratio_B * size_B

  # Calculate weighted average ratio (WAR)
  WR_total = WR_A + WR_B
  total_size = size_A + size_B

  WAR = WR_total / total_size

  return WAR


# Finds all images (files) inside two directories and returns them as lists
def load_images(dir1, dir2):
  images1 = [f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]
  images2 = [f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))]

  return images1, images2


# Moves all files from one directory to another directory
def move_files(source_dir, destination_dir):
  files = os.listdir(source_dir)

  for f in files:
    source_path = os.path.join(source_dir, f)
    destination_path = os.path.join(destination_dir, f)
    shutil.move(source_path, destination_path)


# Takes a list of images and creates segmentation masks in the output folders
def segment_images(sam, images, city, index, save_streetview):
  # Temporarily save images per batch, segment them, then remove original image
  # Also save a non-temporary copy for analysis purposes
  temp_path = os.path.join("results", city, "temp_sv_images")
  sv_path = os.path.join("results", city, "sv_images")

  for i, image in enumerate(images):
    temp_output_path = os.path.join(temp_path, f"{index}_streetview_{i}.tif")
    image.save(temp_output_path)

    # Save streetview images permanently if needed
    if save_streetview:
      output_path = os.path.join(sv_path, f"{index}_streetview_{i}.tif")
      image.save(output_path)

  # Get a list of files inside the temp directory
  files = [f for f in os.listdir(temp_path) if os.path.isfile(os.path.join(temp_path, f))]

  # Create segmentation masks based on the text prompts
  # Note: Should match with folder names defined in prepare_folders()
  text_prompts = ["facades", "windows"]

  for prompt in text_prompts:
    out_dir = os.path.join("results", city, f"temp_seg_{prompt}")

    sam.predict_batch(images=temp_path,
                      out_dir=out_dir,
                      text_prompt=prompt,
                      box_threshold=0.24,
                      text_threshold=0.24,
                      merge=False
                      )

  # Remove SV images to prevent duplicate predictions
  delete_files(temp_path)
