from modules.road_network import *
from modules.process_data import *

import random
import math
from geojson import Feature, FeatureCollection
import fiona
from PIL import Image
import requests
import geopandas as gpd

# Use Fiona for writes so layer= is supported and no pyogrio 'crs' issue
gpd.options.io_engine = "fiona"

# Function adapted from iabrilvzqz
# GitHub: https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/StreetView-NatureVisibility-GSV
# Prepare facade folders, create road network, and create features. 
# Returns the point features
# Prepare facade folders, create road network, and create features.
# Returns the point features

import math
try:
    import numpy as np
except Exception:
    np = None

def _is_inf(x):
    try:
        if isinstance(x, float) and math.isinf(x):
            return True
    except Exception:
        pass
    try:
        if np is not None and bool(np.isinf(x)):
            return True
    except Exception:
        pass
    # strings like "inf"
    try:
        if str(x).strip().lower() in ("inf", "+inf", "infinity"):
            return True
    except Exception:
        pass
    return False

def _to_int_or_default(x, default=None):
    if x is None or _is_inf(x):
        return default
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default

# ---- sanitize numerics ----
distance = _to_int_or_default(distance, 50)

if _is_inf(num_sample_images) or num_sample_images is None:
    num_sample_images = 10**9
else:
    num_sample_images = _to_int_or_default(num_sample_images, 10**6)

begin = _to_int_or_default(begin, 0)
end   = _to_int_or_default(end, None)
i     = _to_int_or_default(i, 0)

def create_features(
    city,
    access_token,
    distance,
    num_sample_images,
    begin,
    end,
    save_roads_points,
    bbox=None,
    i=0,
):
    print(f"Creating features for city {city}")

    # ---- sanitize numerics ----
    try:
        distance = int(distance)
    except Exception:
        distance = 50

    if num_sample_images in (None, float("inf")):
        num_sample_images = 10**9
    else:
        num_sample_images = int(num_sample_images)

    # robust to None, inf, NaN, strings
    begin = 0 if (begin is None or (isinstance(begin, float) and math.isinf(begin))) else int(begin)
    end   = None if (end   is None or (isinstance(end,   float) and math.isinf(end))) else int(end)

    try:
        i = int(i)
    except Exception:
        i = 0

    # Explicit layer names for GPKG
    features_layer = f"points_{i}"
    roads_layer = f"roads_{i}"

    # File paths
    file_path_features = os.path.join("results", city, "points", f"points_{i}.gpkg")
    file_path_road = os.path.join("results", city, "roads", f"roads_{i}.gpkg")

    if not os.path.exists(file_path_features):
        # ---- get roads (supports repos with or without bbox arg) ----
        try:
            road = get_road_network(city, bbox=bbox)
        except TypeError:
            # fall back to city-only
            road = get_road_network(city)

        # Debug
        try:
            print(f"Total roads found: {len(road)}")
        except Exception:
            print("Total roads found: <unknown>")

        # If there are no roads, return empty GeoDataFrame
        if (road is None) or (hasattr(road, "empty") and road.empty) or (hasattr(road, "__len__") and len(road) == 0):
            print(f"No roads found for city {city}. Continuing...")
            return gpd.GeoDataFrame()

        # Normalize dtypes (safe)
        if "index" not in road.columns:
            road["index"] = road.index
        road["index"] = road["index"].astype(str)
        if "highway" in road.columns:
            road["highway"] = road["highway"].astype(str)
        if "length" in road.columns:
            road["length"] = road["length"].astype(float)
        if "road_angle" in road.columns:
            road["road_angle"] = road["road_angle"].astype(float)

        # Create features from the sample points
        points = select_points_on_road_network(road, distance)
        features = get_features_on_points(points, road, access_token, distance)

        # Save roads & features if requested
        if save_roads_points:
            # IMPORTANT: do not pass crs= here; let GeoDataFrame carry it
            road.to_file(file_path_road, driver="GPKG", layer=roads_layer)
            features.to_file(file_path_features, driver="GPKG", layer=features_layer)

    else:
        # If already saved, read that layer
        features = gpd.read_file(file_path_features, layer=features_layer)

    # Sample a subset to analyze
    k = int(min(num_sample_images, len(features)))
    if k > 0:
        sample_indices = random.sample(range(len(features)), k)
    else:
        sample_indices = []

    features["save_sample"] = False
    if sample_indices:
        features.loc[sample_indices, "save_sample"] = True

    if "id" in features.columns:
        features = features.sort_values(by="id")

    # Optional subrange
    if (begin is not None) and (end is not None):
        features = features.iloc[begin : (end if end is not None else None)]

    return features


# For each feature, calculates the facade greening potential score (GPS).
# Returns a list of GeoJSON features.
def calculate_usable_wall_ratios(features, city, sam, access_token, save_streetview, bbox_i=0):
    # Keep bbox_i as int internally; use string only for folder suffix
    try:
        bbox_i_int = int(bbox_i)
    except Exception:
        bbox_i_int = 0
    suffix = str(bbox_i_int)

    # Mapillary header
    header = {"Authorization": f"OAuth {access_token}"}

    usable_ratio = []

    # temp seg output dirs
    facades_dir = os.path.join("results", city, "temp_seg_facades")
    windows_dir = os.path.join("results", city, "temp_seg_windows")

    for index, row in features.iterrows():
        if row.get("save_sample", False):

            image_id = row["image_id"]
            url = (
                f"https://graph.mapillary.com/{image_id}?fields="
                "camera_parameters,camera_type,captured_at,compass_angle,"
                "computed_geometry,computed_rotation,thumb_original_url"
            )

            # Fetch metadata
            try:
                response = requests.get(url, headers=header, timeout=20)
                response.raise_for_status()
                data = response.json()
            except Exception:
                print("Error retrieving image data. Skipping sample.")
                continue

            try:
                image_url = data["thumb_original_url"]
                camera_angle = data.get("compass_angle", None)
                location = data["computed_geometry"]
                computed_rotation = data["computed_rotation"]
                yaw = computed_rotation[1]
            except KeyError:
                print("Image invalid: missing attributes")
                continue

            # filter on yaw
            if yaw < -0.25 * math.pi or yaw > 0.25 * math.pi:
                print(f"Image invalid: yaw too high ({yaw})")
                continue

            road_angle = float(row["road_angle"]) if not math.isnan(row["road_angle"]) else 0.0
            is_panoramic = bool(row["is_panoramic"])

            # Get one or two perp-view images
            images = process_image(image_url, is_panoramic, road_angle)

            # Segment
            segment_images(sam, images, city, index, save_streetview)

            # Gather seg masks
            facades_segs, windows_segs = load_images(facades_dir, windows_dir)

            widths, heights, ratios = [], [], []

            # default ratio if nothing detected
            ratio = 0

            if (not facades_segs and not windows_segs) or (not facades_segs):
                print("Invalid: No facades (or windows) detected.")
                ratio = 0

            elif not windows_segs:
                print("No windows detected")
                ratio = 1

            else:
                if not is_panoramic:
                    # expect one matching pair
                    for image_name in facades_segs:
                        if image_name in windows_segs:
                            with Image.open(f"{facades_dir}/{image_name}") as facades_seg:
                                facades_count_total = count_white_pixels(facades_seg)
                                width, height = facades_seg.size
                            with Image.open(f"{windows_dir}/{image_name}") as windows_seg:
                                windows_count_total = count_white_pixels(windows_seg)

                            ratio = (
                                (facades_count_total - windows_count_total) / facades_count_total
                                if (facades_count_total - windows_count_total) > 0
                                else 0
                            )
                            widths.append(width); heights.append(height); ratios.append(ratio)

                else:
                    n_f = len(facades_segs); n_w = len(windows_segs)

                    if n_f >= n_w:
                        for image_name in facades_segs:
                            if image_name in windows_segs:
                                with Image.open(f"{facades_dir}/{image_name}") as facades_seg:
                                    facades_count_total = count_white_pixels(facades_seg)
                                    width, height = facades_seg.size
                                with Image.open(f"{windows_dir}/{image_name}") as windows_seg:
                                    windows_count_total = count_white_pixels(windows_seg)

                                ratio = (
                                    (facades_count_total - windows_count_total) / facades_count_total
                                    if (facades_count_total - windows_count_total) > 0
                                    else 0
                                )
                                widths.append(width); heights.append(height); ratios.append(ratio)
                            else:
                                with Image.open(f"{facades_dir}/{image_name}") as facades_seg:
                                    facades_count_total = count_white_pixels(facades_seg)
                                    width, height = facades_seg.size
                                windows_count_total = 0
                                print("Missing window segmentation. Window area: 0")
                                ratio = (
                                    (facades_count_total - windows_count_total) / facades_count_total
                                    if (facades_count_total - windows_count_total) > 0
                                    else 0
                                )
                                widths.append(width); heights.append(height); ratios.append(ratio)
                    else:
                        for image_name in windows_segs:
                            if image_name in facades_segs:
                                with Image.open(f"{windows_dir}/{image_name}") as windows_seg:
                                    windows_count_total = count_white_pixels(windows_seg)
                                    width, height = windows_seg.size
                                with Image.open(f"{facades_dir}/{image_name}") as facades_seg:
                                    facades_count_total = count_white_pixels(facades_seg)  # <-- fixed typo

                                ratio = (
                                    (facades_count_total - windows_count_total) / facades_count_total
                                    if (facades_count_total - windows_count_total) > 0
                                    else 0
                                )
                                widths.append(width); heights.append(height); ratios.append(ratio)
                            else:
                                with Image.open(f"{windows_dir}/{image_name}") as windows_seg:
                                    windows_count_total = count_white_pixels(windows_seg)
                                    width, height = windows_seg.size
                                facades_count_total = 0
                                print("Missing facade segmentation. Facade area: 0")
                                ratio = (
                                    (facades_count_total - windows_count_total) / facades_count_total
                                    if (facades_count_total - windows_count_total) > 0
                                    else 0
                                )
                                widths.append(width); heights.append(height); ratios.append(ratio)

            # Compute WAR
            try:
                ratio_left, ratio_right = ratios[0], ratios[1]
                width_left, width_right = widths[0], widths[1]
                height_left, height_right = heights[0], heights[1]
                WAR = calculate_WAR(width_left, height_left, ratio_left, width_right, height_right, ratio_right)
            except Exception:
                # non-pano or only one seg → fall back to single-image ratio
                ratio_left, ratio_right = (ratios[0], ratios[1]) if len(ratios) >= 2 else (None, None)
                WAR = ratios[0] if ratios else ratio

            # Move masks to per-bbox folders
            prepare_folder(city, os.path.join("seg_facades", suffix))
            facades_destination = os.path.join("results", city, "seg_facades", suffix)
            move_files(facades_dir, facades_destination)

            prepare_folder(city, os.path.join("seg_windows", suffix))
            windows_destination = os.path.join("results", city, "seg_windows", suffix)
            move_files(windows_dir, windows_destination)

            # Save one feature
            usable_ratio.append(
                Feature(
                    geometry=location,
                    properties={
                        "ratio_left": ratio_left,
                        "ratio_right": ratio_right,
                        "GPS": round(WAR, 2),
                        "image_url": image_url,
                        "camera_angle": camera_angle,
                        "road_angle": road_angle,
                        "idx": int(index),
                    },
                )
            )

    return usable_ratio


def save_usable_wall_ratios(city, usable_ratios):
    print(len(usable_ratios))
    if len(usable_ratios) == 0:
        print("No usable ratios found. Nothing to save.")
        return

    feature_collection = FeatureCollection(usable_ratios)
    gdf = gpd.GeoDataFrame.from_features(feature_collection["features"])

    # Ensure CRS on the GeoDataFrame, then write (no crs= in to_file)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)

    features_file = f"{city}_features.gpkg"
    features_path = os.path.join("results", city)
    os.makedirs(features_path, exist_ok=True)

    # Write with explicit layer name for clarity
    gdf.to_file(os.path.join(features_path, features_file), driver="GPKG", layer="features")

    print(f"Saved features to {features_file}")


