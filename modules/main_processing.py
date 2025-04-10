from modules.road_network import *
from modules.process_data import *

import random
import math
from geojson import Feature, FeatureCollection
import fiona

# Function adapted from iabrilvzqz
# GitHub: https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/StreetView-NatureVisibility-GSV
# Prepare facade folders, create road network, and create features. 
# Returns the point features
def create_features(city, access_token, distance, num_sample_images, begin, end, save_roads_points, bbox=None, i=0):
    print(f"Creating features for city {city}")  # Debug statement
    # Create file paths for roads and points
    file_path_features = os.path.join("results", city, "points", f"points_{i}.gpkg")
    file_path_road = os.path.join("results", city, "roads", f"roads_{i}.gpkg")

    if not os.path.exists(file_path_features):
        # Get the sample points and the features assigned to each point
        # Use city name only (no bbox)
        road = get_road_network(city)

        # Debug prints to check if roads are fetched
        print(f"Total roads found: {len(road)}")  # Check the length of roads fetched

        if len(road) == 0:
            print(f"No roads found for city {city}. Continuing...")
            return gpd.GeoDataFrame()  # Return an empty GeoDataFrame

        # If there are no roads, there are no features too
        if road.empty:
            return gpd.GeoDataFrame()

        # Save road in gpkg file
        road["index"] = road.index
        road["index"] = road["index"].astype(str)
        road["highway"] = road["highway"].astype(str)
        road["length"] = road["length"].astype(float)
        road["road_angle"] = road["road_angle"].astype(float)

        # Create features from the sample points
        points = select_points_on_road_network(road, distance)
        features = get_features_on_points(points, road, access_token, distance)

        # Save the roads and features if necessary
        if save_roads_points:
            road[["index", "geometry", "length", "highway", "road_angle"]].to_file(file_path_road, driver="GPKG", crs=road.crs)
            features.to_file(file_path_features, driver="GPKG")
    else:
        # If the points file already exists, then we use it to continue with the analysis
        features = gpd.read_file(file_path_features, layer=f"points_{i}")

    # Set True for n randomly selected rows to analyze their images later
    sample_indices = random.sample(range(len(features)), min(num_sample_images, len(features)))
    features["save_sample"] = False
    features.loc[sample_indices, "save_sample"] = True

    features = features.sort_values(by='id')

    # If we include a begin and end value, then the dataframe is split and we are going to analyse just those points
    if begin != None and end != None:
        features = features.iloc[begin:end]
    
    return features


# For each feature, calculates the facade greening potential score (GPS).
# Returns a list of GeoJSON features.
def calculate_usable_wall_ratios(features, city, sam, access_token, save_streetview, bbox_i="0"):
  # Get Mapillary header
  header = {'Authorization': 'OAuth {}'.format(access_token)}

  # Initialize output list
  usable_ratio = []

  # Initialize output directories of segmentations
  facades_dir = os.path.join("results", city, "temp_seg_facades")
  windows_dir = os.path.join("results", city, "temp_seg_windows")

  # Calculate the ratio of usable wall surface for each street view image (SVI)
  for index, row in features.iterrows():
      if row["save_sample"] == True:

          # Get info from SVI URL by concatenating parameter fields
          # Note: does not prevent duplicate image
          image_id = row["image_id"]
          url = f'https://graph.mapillary.com/{image_id}?fields=\
                  camera_parameters,\
                  camera_type,\
                  captured_at,\
                  compass_angle,\
                  computed_geometry,\
                  computed_rotation,\
                  thumb_original_url'

          # Send a GET request to the Mapillary API to obtain SVI information
          try:
            response = requests.get(url, headers=header)
            data = response.json()
          except:
            print("Error retrieving image data. Skipping sample.")
            continue

          try:
            image_url = data["thumb_original_url"]
            camera_angle = data["compass_angle"] if "compass_angle" in data else None  # Handle missing compass_angle
            # Location is GeoJSON Point
            location = data["computed_geometry"]

            # Get yaw (up/down) angle. Range: -pi to +pi radians
            computed_rotation = data["computed_rotation"]
            yaw = computed_rotation[1]
          except KeyError:
            print(f"Image invalid: missing attributes")
            continue

          # Makes sure the location is skipped if image angle > 45 degrees
          if yaw < -0.25 * math.pi or yaw > 0.25 * math.pi:
            print(f"Image invalid: yaw too high ({yaw})")
            continue

          # Extract road angle and panoramic property from dataframe
          road_angle = float(row["road_angle"] if not math.isnan(row["road_angle"]) else 0)
          is_panoramic = bool(row["is_panoramic"])

          # Return list of processed image(s)
          images = process_image(image_url, is_panoramic, road_angle)

          # Create segmentation masks of facades and windows using SAM
          segment_images(sam, images, city, index, save_streetview)

          # Get pixel ratio of usable facade compared to total facade
          # We use file names to find matching segmentations in the temp folders
          facade_segs, windows_segs = load_images(facades_dir, windows_dir)

          widths = []
          heights = []
          ratios = []
          
          # Check if there are facade and window segmentations
          # if at least one is missing, we can determine the WAR (GPS) directly
          ratio_left, ratio_right = None, None

          if (not facade_segs and not windows_segs) or (not facade_segs):
            print("Invalid: No facades (or windows) detected.")
            ratio = 0
          elif not windows_segs:
            print("No windows detected")
            ratio = 1
          # There is at least one segmentation, so continue
          else:
            if not is_panoramic:
              # Non-panoramic images always have 1 facade and 1 window seg by now
              for image_name in facade_segs:
                if image_name in windows_segs:

                  with Image.open(f"{facades_dir}/{image_name}") as facades_seg:
                    facades_count_total = count_white_pixels(facades_seg)
                    width, height = facades_seg.size

                  with Image.open(f"{windows_dir}/{image_name}") as windows_seg:
                    windows_count_total = count_white_pixels(windows_seg)

                  # Get pixel ratio of usable facade compared to total facade
                  ratio = (facades_count_total - windows_count_total) / facades_count_total \
                          if (facades_count_total - windows_count_total) > 0 else 0

                  # Get ratio, width, height for left and right image
                  widths.append(width)
                  heights.append(height)
                  ratios.append(ratio)
            # Panoramic images can have 1,1, 1,2, 2,1, 2,2 segs by this point
            else:
              n_facade_segs = len(facade_segs)
              n_windows_segs = len(windows_segs)

              # Loop over the longest (or equal size) list so we don't miss any segmentations when name-matching
              if n_facade_segs >= n_windows_segs:
                # Check if there is a matching name for this seg
                for image_name in facade_segs:
                  if image_name in windows_segs:

                    with Image.open(f"{facades_dir}/{image_name}") as facades_seg:
                      facades_count_total = count_white_pixels(facades_seg)
                      width, height = facades_seg.size

                    with Image.open(f"{windows_dir}/{image_name}") as windows_seg:
                      windows_count_total = count_white_pixels(windows_seg)

                    # Get pixel ratio of usable facade compared to total facade
                    ratio = (facades_count_total - windows_count_total) / facades_count_total \
                            if (facades_count_total - windows_count_total) > 0 else 0

                    # Get ratio, width, height for left and right image
                    widths.append(width)
                    heights.append(height)
                    ratios.append(ratio)
                  # If there is no matching name, set the count of the missing window seg to 0
                  else:
                    with Image.open(f"{facades_dir}/{image_name}") as facades_seg:
                      facades_count_total = count_white_pixels(facades_seg)
                      width, height = facades_seg.size

                    windows_count_total = 0
                    print("Missing window segmentation. Window area: 0")

                    # Get pixel ratio of usable facade compared to total facade
                    ratio = (facades_count_total - windows_count_total) / facades_count_total \
                            if (facades_count_total - windows_count_total) > 0 else 0

                    # Ratio, width, height for left and right image
                    widths.append(width)
                    heights.append(height)
                    ratios.append(ratio)
              else:
                # Switch the name-matching order so we don't miss matches
                for image_name in windows_segs:
                  if image_name in facade_segs:

                    with Image.open(f"{windows_dir}/{image_name}") as windows_seg:
                      windows_count_total = count_white_pixels(windows_seg)
                      width, height = windows_seg.size

                    with Image.open(f"{facades_dir}/{image_name}") as facades_seg:
                      facades_count_total = count_white_pixels(windows_seg)

                    # Get pixel ratio of usable facade compared to total facade
                    ratio = (facades_count_total - windows_count_total) / facades_count_total \
                            if (facades_count_total - windows_count_total) > 0 else 0

                    # Ratio, width, height for left and right image
                    widths.append(width)
                    heights.append(height)
                    ratios.append(ratio)
                  # If there is no matching name, set the count of the missing facade seg to 0
                  else:
                    with Image.open(f"{windows_dir}/{image_name}") as windows_seg:
                      windows_count_total = count_white_pixels(windows_seg)
                      width, height = windows_seg.size

                    facades_count_total = 0
                    print("Missing facade segmentation. Facade area: 0")

                    # Get pixel ratio of usable facade compared to total facade
                    ratio = (facades_count_total - windows_count_total) / facades_count_total \
                            if (facades_count_total - windows_count_total) > 0 else 0

                    # Get ratio, width, height for left and right image
                    widths.append(width)
                    heights.append(height)
                    ratios.append(ratio)

          # Save usable ratios
          try:
            ratio_left, ratio_right = ratios[0], ratios[1]
            width_left, width_right = widths[0], widths[1]
            height_left, height_right = heights[0], heights[1]

            # Calculate the Weighted Average Ratio (GPS) of this sample point
            WAR = calculate_WAR(width_left, height_left, ratio_left, width_right, height_right, ratio_right)
          except IndexError:
            print("Index error (non-panoramic or no segmentation).")
            ratio_left, ratio_right = None, None        
            # If the image is panoramic (or no segs), the WAR is equal to the ratio of that image
            WAR = ratio

          # Move all masks to their non-temporary folders after analysis so the analysis path is clear
          prepare_folder(city, os.path.join("seg_facades", bbox_i))
          facades_destination = os.path.join("results", city, "seg_facades", bbox_i)
          move_files(facades_dir, facades_destination)

          prepare_folder(city, os.path.join("seg_windows", bbox_i))
          windows_destination = os.path.join("results", city, "seg_windows", bbox_i)
          move_files(windows_dir, windows_destination)

          # Save the usable ratio as a feature
          usable_ratio.append(Feature(geometry = location, properties = {"ratio_left": ratio_left,
                                                                        "ratio_right": ratio_right,
                                                                        "GPS": round(WAR, 2),
                                                                        "image_url": image_url,
                                                                        "camera_angle": camera_angle,
                                                                        "road_angle": road_angle,
                                                                        "idx": index}))

  return usable_ratio



# Save the usable ratios as a geopackage file
# Save the usable ratios as a geopackage file
def save_usable_wall_ratios(city, usable_ratios):
    # Check if usable_ratios is empty
    print(len(usable_ratios))  # Check if it's empty
    if len(usable_ratios) == 0:
        print("No usable ratios found. Nothing to save.")
        return  # Exit the function if no usable ratios

    # If usable_ratios is not empty, continue with saving the file
    feature_collection = FeatureCollection(usable_ratios)

    # Convert the feature collection to a GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(feature_collection["features"])

    # Check the GeoDataFrame to ensure it's correct
    print(gdf.head())  # Check the first few rows
    print(gdf.crs)  # Check the CRS

    # Set the CRS to EPSG:4326
    gdf.set_crs('EPSG:4326', allow_override=True, inplace=True)

    # Check if CRS has been set correctly
    print(gdf.crs)  # Double-check the CRS

    # Proceed with saving the GeoDataFrame to a geopackage file
    features_file = f'{city}_features.gpkg'
    features_path = os.path.join("results", city)

    # Check if the features directory exists, if not, create it
    if not os.path.exists(features_path):
        os.makedirs(features_path)

    # Save the GeoDataFrame as a geopackage file
    gdf.to_file(f'{features_path}/{features_file}', driver="GPKG", engine="fiona")

    print(f"Saved features to {features_file}")

