# Copyright 2024 omasakun <omasakun@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# %%

import math

import torch
from torch import Tensor

def haversine_distance(lat1, lon1, lat2, lon2):
  # Equatorial radius (https://en.wikipedia.org/wiki/World_Geodetic_System#WGS_84)
  radius = 6378137
  lat1 = lat1 * math.pi / 180
  lon1 = lon1 * math.pi / 180
  lat2 = lat2 * math.pi / 180
  lon2 = lon2 * math.pi / 180
  d_lat = lat2 - lat1
  d_lon = lon2 - lon1

  if type(lat1) == torch.Tensor:
    inner = torch.sin(d_lat / 2)**2 + torch.sin(d_lon / 2)**2 * torch.cos(lat1) * torch.cos(lat2)
    return 2 * radius * torch.asin(torch.sqrt(inner))
  else:
    inner = math.sin(d_lat / 2)**2 + math.sin(d_lon / 2)**2 * math.cos(lat1) * math.cos(lat2)
    return 2 * radius * math.asin(math.sqrt(inner))

def geoguesser_score(distance, map_size=14916862):
  """
  https://www.reddit.com/r/geoguessr/comments/zqwgnr/comment/j12rjkq/ by MiraMattie on 2022-12-21 (edited on 2023-06-10)

  There have been lots of discussions. One of the older ones is:
  https://www.reddit.com/r/geoguessr/comments/7ekj80/for_all_my_geoguessing_math_nerds/

  You can look at the code to Chatguessr on github for one of the most impactful estimates, which uses score 5000 * 0.99866017 ^ ( distance in meters / scale), where the scale is determined by the distance between the top left and bottom right corners of the map, divided by 7.458421.

  I hate odd magic numbers, so I did some math. With:
  - s = score
  - d = distance, meters
  - z = map size, meters (provided as maxErrorDistance by the map API that indicates the map's size; chatguessr recalculates it from the bounds; but gets the same number - for the world map, 14916862 if you're using distance in meters, 14916.862 if in KM)
  - m = max score (5000)
  - k1 = exponent base - (0.99866017 in chatguessr)
  - k2 = power factor (7458.421 in chatguessr - well actually they divide the error by this, and then multiply the distance by 1000)

  ... jugging around the code, the general equation they use is:
  s = m * k1 ^ ( k2 * d / z )
  For simplicity, let's divide both sides by m:
  s / m = k1 ^ ( k2 * d / z )
  ... then we can say:
  - Ps = s / m (Percent score)
  - Pd = d / z (Percent distance)
  ... and the equation gets real simple:
  Ps = k1 ^ ( k2 * Pd )
  Take the log:
  ln(Ps) = ln (k1 ^ ( k2 * Pd ) )
  ... so we can pull out the expontant:
  ln(Ps) = k2 * Pd * ln (k1)
  ... Rearrange ever so slightly:
  ln(Ps) = Pd * k2 * ln (k1)
  Now we can observe that since k1 and k2 are constants, k2 * ln(k1) is itself a different constant. So let's call it k, and define it as:
  k = k2 * ln (k1)
  ... making the calculation:
  ln(Ps) = Pd * k
  un-log both sides, and that's:
  Ps = e ^ ( Pd * k )
  Using the values of k1 and k2 in chatguessr, k = -10.0040256448936.

  The value of that constant is just too close to -10 to be a coincidence. So I believe Geoguessr's score calculation uses k = -10; plugging that in and backing out the substitutions, then:
  s = 5000 * e ^ ( -10 * d / z )
  ... and now we have a nice, simple formula with no weird high-precision constants needed to calculate the score.

  There are lots of smart geoguessers and it wouldn't surprise me if someone has derived it before, but it's the first time I've seen a precise calculation with a single whole number as a constant.
  """
  if type(distance) == torch.Tensor:
    return 5000 * torch.exp((-10 * distance) / map_size)
  else:
    return 5000 * math.exp((-10 * distance) / map_size)

def latlon_to_xyz(lat: float, lon: float):
  lat = math.radians(lat)
  lon = math.radians(lon)
  x = math.cos(lat) * math.cos(lon)
  y = math.cos(lat) * math.sin(lon)
  z = math.sin(lat)
  return x, y, z

def xyz_to_latlon_torch(x: Tensor, y: Tensor, z: Tensor):
  length = torch.sqrt(x**2 + y**2 + z**2)
  lat = torch.asin(z / length)
  lon = torch.atan2(y, x)
  lat = lat * 180 / math.pi
  lon = lon * 180 / math.pi
  lat = lat.nan_to_num(nan=0)
  lon = lon.nan_to_num(nan=0)
  return lat, lon
