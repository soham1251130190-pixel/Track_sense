import numpy as np
from pyproj import Transformer

# Real first GPS point of IO-VNBD S1 dataset
ORIGIN_LAT = 52.40166
ORIGIN_LON = -1.50529


def latlon_to_local_xy(
    latitude,
    longitude,
    origin_latitude=ORIGIN_LAT,
    origin_longitude=ORIGIN_LON
):
    """
    Convert GPS latitude/longitude to local X/Y coordinates in metres.
    X = Easting difference
    Y = Northing difference
    The default origin is the agreed first GPS point
    of the IO-VNBD S1 dataset.
    """
    latitude = np.asarray(latitude, dtype=float)
    longitude = np.asarray(longitude, dtype=float)
    # Choose UTM zone from the origin longitude.
    utm_zone = int((origin_longitude + 180) // 6) + 1
    epsg = 32600 + utm_zone
    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg}",
        always_xy=True
    )
    # Project GPS points.
    east, north = transformer.transform(
        longitude,
        latitude
    )
    # Project the origin.
    origin_east, origin_north = transformer.transform(
        origin_longitude,
        origin_latitude
    )
    # Shift origin to (0, 0).
    x = east - origin_east
    y = north - origin_north
    return x, y


def local_xy_to_latlon(
    x,
    y,
    origin_latitude=ORIGIN_LAT,
    origin_longitude=ORIGIN_LON,
):
    """
    Convert local X/Y coordinates in metres back to
    GPS latitude/longitude.
    Uses the same UTM zone and origin as
    latlon_to_local_xy().
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Use the same UTM zone as the forward projection.
    utm_zone = int((origin_longitude + 180) // 6) + 1
    epsg = 32600 + utm_zone
    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg}",
        always_xy=True
    )
    # Convert the origin to UTM coordinates.
    origin_east, origin_north = transformer.transform(
        origin_longitude,
        origin_latitude
    )
    # Restore absolute UTM coordinates.
    east = x + origin_east
    north = y + origin_north
    # Convert UTM back to GPS.
    inverse_transformer = Transformer.from_crs(
        f"EPSG:{epsg}",
        "EPSG:4326",
        always_xy=True
    )
    longitude, latitude = inverse_transformer.transform(
        east,
        north
    )
    return latitude, longitude
