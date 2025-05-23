from vt2geojson.tools import vt_bytes_to_geojson
from shapely.geometry import Point
from scipy.spatial import cKDTree
import geopandas as gpd
import osmnx as ox
import mercantile

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import numpy as np
import requests
import math
import networkx as nx


# Function adapted from iabrilvzqz
# GitHub: https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/StreetView-NatureVisibility-GSV
# Generates a road network from either a placename or bounding box using OpenStreetMap data
# Also saves the road angle of each road segment
def get_road_network(city, bbox=None):
    print(f"Fetching road network for city: {city}")  # Debug print to check the city
    # Use a custom filter to only get car-driveable roads that are not motorways or trunk roads
    cf = '["highway"~"primary|secondary|tertiary|residential|primary_link|secondary_link|tertiary_link|living_street|service|unclassified"]'

    try:
        if bbox:
            G = ox.graph_from_bbox(bbox[3], bbox[1], bbox[2], bbox[0], simplify=True, custom_filter=cf)
        else:
            G = ox.graph_from_place(city, simplify=True, custom_filter=cf)
    except:
        # If there are no roads, return an empty gdf
        return gpd.GeoDataFrame()

    # Create a set to store unique road identifiers
    unique_roads = set()

    # Create a new graph to store the simplified road network
    G_simplified = G.copy()

    # Iterate over each road segment
    for u, v, key in G.edges(keys=True):
        # Check if the road segment is a duplicate
        if (v, u) in unique_roads:
            # Remove the duplicate road segment
            G_simplified.remove_edge(u, v, key)
        else:
            # Add the road segment to the set of unique roads
            unique_roads.add((u, v))

            y0, x0 = G.nodes[u]['y'], G.nodes[u]['x']
            y1, x1 = G.nodes[v]['y'], G.nodes[v]['x']

            # Calculate the angle from North (in radians)
            angle_from_north = math.atan2(x1 - x0, y1 - y0)

            # Convert the angle to degrees
            angle_from_north_degrees = math.degrees(angle_from_north)

            if angle_from_north_degrees < 0:
                angle_from_north_degrees += 360.0

            # Add the road angle as a new attribute to the edge
            G_simplified.edges[u, v, key]['road_angle'] = angle_from_north_degrees

    # Update the graph with the simplified road network
    G = G_simplified
    
    # Project the graph from latitude-longitude coordinates to a local projection (in meters)
    G_proj = ox.project_graph(G)

    # Convert the projected graph to a GeoDataFrame
    _, edges = ox.graph_to_gdfs(G_proj) 

    return edges


# Function courtesy of iabrilvzqz
# GitHub: https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/StreetView-NatureVisibility-GSV
# Get a gdf of points over a road network with a N distance between them
def select_points_on_road_network(roads, N=50):
  points = []
  
  # Iterate over each road
  for row in roads.itertuples(index=True, name='Road'):
    # Get the LineString object from the geometry
    linestring = row.geometry
    index = row.Index

    # Calculate the distance along the linestring and create points every 50 meters
    for distance in range(0, int(linestring.length), N):
      # Get the point on the road at the current position
      point = linestring.interpolate(distance)

      # Add the curent point to the list of points
      points.append([point, index])
  
  # Convert the list of points to a GeoDataFrame
  gdf_points = gpd.GeoDataFrame(points, columns=["geometry", "road_index"], geometry="geometry")

  # Set the same CRS as the road dataframes for the points dataframe
  gdf_points.set_crs(roads.crs, inplace=True)

  # Drop duplicate rows based on the geometry column
  gdf_points = gdf_points.drop_duplicates(subset=['geometry'])
  gdf_points = gdf_points.reset_index(drop=True)

  return gdf_points


# Function courtesy of iabrilvzqz
# GitHub: https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/StreetView-NatureVisibility-GSV
# This function extracts the features for a given tile
def get_features_for_tile(tile, access_token):
    # This URL retrieves all the features within the tile. These features are then going to be assigned to each sample point depending on the distance.
    tile_url = f"https://tiles.mapillary.com/maps/vtp/mly1_public/2/{tile.z}/{tile.x}/{tile.y}?access_token={access_token}"
    response = requests.get(tile_url)
    result = vt_bytes_to_geojson(response.content, tile.x, tile.y, tile.z, layer="image")
    return [tile, result]


# Function adapted from iabrilvzqz
# GitHub: https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/StreetView-NatureVisibility-GSV
# Creates features from sample points
def get_features_on_points(points, road, access_token, max_distance=50, zoom=14):
  # Store the local crs in meters that was assigned by osmnx previously so we can use it to calculate the distances between features and points
  local_crs = points.crs

  # Set the CRS to 4326 because it is used by Mapillary
  points.to_crs(crs=4326, inplace=True)
  
  # Add a new column to gdf_points that contains the tile coordinates for each point
  points["tile"] = [mercantile.tile(x, y, zoom) for x, y in zip(points.geometry.x, points.geometry.y)]

  # Group the points by their corresponding tiles
  groups = points.groupby("tile")

  # Download the tiles and extract the features for each group
  features = []
  
  # To make the process faster the tiles are downloaded using threads
  with ThreadPoolExecutor(max_workers=10) as executor:
    futures = []

    for tile, _ in groups:
      futures.append(executor.submit(get_features_for_tile, tile, access_token))
    
    for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading tiles"):
      result = future.result()
      features.append(result)

  pd_features = pd.DataFrame(features, columns=["tile", "features"])

  # Compute distances between each feature and all the points in gdf_points
  feature_points = gpd.GeoDataFrame(
    [(Point(f["geometry"]["coordinates"]), f) for row in pd_features["features"] for f in row["features"]],
    columns=["geometry", "feature"],
    geometry="geometry",
    crs=4326)

  # Transform from EPSG:4326 (world °) to the local crs in meters that we got when we projected the roads graph in the previous step
  feature_points.to_crs(local_crs, inplace=True)
  points.to_crs(local_crs, inplace=True)

  # Create a KDTree (k-dimensional tree) from the "geometry" coordinates of feature_points
  feature_tree = cKDTree(feature_points["geometry"].apply(lambda p: [p.x, p.y]).tolist())
  # Use the KDTree to query the nearest neighbors of the points in the "geometry" column of points DataFrame
  # The query returns the distances and indices of the nearest neighbors
  # The parameter "k=1" specifies that we want to find the nearest neighbor
  # The parameter "distance_upper_bound=max_distance" sets a maximum distance for the nearest neighbors
  distances, indices = feature_tree.query(points["geometry"].apply(lambda p: [p.x, p.y]).tolist(), k=1, distance_upper_bound=max_distance/2)

  # Create a list to store the closest features and distances to each point. If there are no images close then set the value of both to None
  closest_features = [feature_points.loc[i, "feature"] if np.isfinite(distances[idx]) else None for idx, i in enumerate(indices)]
  closest_distances = [distances[idx] if np.isfinite(distances[idx]) else None for idx in range(len(distances))]

  # Store the closest feature for each point in the "feature" column of the points DataFrame
  points["feature"] = closest_features

  # Store the distances as a new column in points
  points["distance"] = closest_distances

  # Store image id and is_panoramic information as part of the dataframe
  points["image_id"] = points.apply(lambda row: str(row["feature"]["properties"]["id"]) if row["feature"] else "", axis=1)
  points["image_id"] = points["image_id"].astype(str)
  
  points["is_panoramic"] = points.apply(lambda row: bool(row["feature"]["properties"]["is_pano"]) if row["feature"] else None, axis=1)
  points["is_panoramic"] = points["is_panoramic"].astype(bool)

  # Add camera angle
  header = {'Authorization': 'OAuth {}'.format(access_token)}

  def get_compass_angle(image_id):
    try:
        # Request to Mapillary API
        response = requests.get(f'https://graph.mapillary.com/{image_id}?fields=compass_angle',
                                headers=header)
        # Check if the response was successful and contains the desired field
        if response.status_code == 200:
            data = response.json()
            # Return the compass angle if it exists, else None
            return data.get("compass_angle", None)  # Using .get() ensures no KeyError
        else:
            print(f"Error fetching data for image_id {image_id}, Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching compass angle for image {image_id}: {str(e)}")
        return None

# Safely apply the function to each row in the DataFrame
  points["camera_angle"] = points.apply(lambda row: get_compass_angle(row["image_id"]) if row["feature"] else None, axis=1)
  
  # Add road angle
  points = gpd.sjoin_nearest(points, road, how='left', max_distance = 0.2)

  columns_to_drop = ["index_right0", "index_right1", "index_right2", "osmid", "name", "highway", "oneway", "reversed", "length",
                     "maxspeed", "lanes", "width", "bridge", "access", "tunnel", "index", "ref", "junction", "service"]
  
  for column in columns_to_drop:
    if column in points.columns:
      points = points.drop(columns=[column])

  # Convert results to geodataframe
  points["road_index"] = points["road_index"].astype(str)
  points["tile"] = points["tile"].astype(str)

  # Save the current index as a column
  points["id"] = points.index
  points = points[~points.index.duplicated(keep="first")]
  points = points.reset_index(drop=True)

  # Transform the coordinate reference system to EPSG 4326
  points.to_crs(epsg=4326, inplace=True)

  return points
