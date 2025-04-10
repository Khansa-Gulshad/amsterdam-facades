from area import area
from shapely.geometry import Polygon


# Creates Shapely polygon from coordinates
def coords_to_shapely_polygon(x_neg, y_neg, x_pos, y_pos):
  coords = [[x_neg,y_neg], [x_neg,y_pos], [x_pos,y_pos], [x_pos,y_neg], [x_neg,y_neg]]
  polygon = Polygon(coords)

  return polygon


# Creates GeoJSON polygon from coordinates
def coords_to_polygon(x_neg, y_neg, x_pos, y_pos):
  rect = {'type':'Polygon','coordinates':[[[x_neg,y_neg], [x_neg,y_pos], [x_pos,y_pos], [x_pos,y_neg], [x_neg,y_neg]]]}

  return rect


# Returns area of rectangle with bbox (-x,-y,x,y) in hectares
def rect_area(x_neg, y_neg, x_pos, y_pos):
  rect = coords_to_polygon(x_neg, y_neg, x_pos, y_pos)
  rect_area_ha = area(rect) / 10000

  return rect_area_ha


# Recursively splits up rectangle until they are each smaller than max_area_ha
# Returns the list of all rectangles
def split_rectangle(max_area_ha, x_neg, y_neg, x_pos, y_pos):
  # Calculate the area of the original rectangle
  area_ha = rect_area(x_neg, y_neg, x_pos, y_pos)

  # If the area is larger than max_area_ha, split the rectangle into four equal-sized rectangles
  # This is the max selection size we allow at one time for processing reasons
  if area_ha > max_area_ha:
    x_mid = (x_neg + x_pos) / 2
    y_mid = (y_neg + y_pos) / 2

    # Recursively split each smaller rectangle
    rectangles = [
      split_rectangle(max_area_ha, x_neg, y_neg, x_mid, y_mid),
      split_rectangle(max_area_ha, x_mid, y_neg, x_pos, y_mid),
      split_rectangle(max_area_ha, x_neg, y_mid, x_mid, y_pos),
      split_rectangle(max_area_ha, x_mid, y_mid, x_pos, y_pos)]

    # Combine the coordinate lists of all smaller rectangles
    result = [rect for sublist in rectangles for rect in sublist]
  else:
    # If the area is not larger than 400, return the coordinates of the original rectangle
    result = [[x_neg, y_neg, x_pos, y_pos]]

  return result
