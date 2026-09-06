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
    """

    latitude = np.asarray(latitude, dtype=float)
    longitude = np.asarray(longitude, dtype=float)

    utm_zone = int((origin_longitude + 180) // 6) + 1
    epsg = 32600 + utm_zone

    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg}",
        always_xy=True
    )

    east, north = transformer.transform(
        longitude,
        latitude
    )

    origin_east, origin_north = transformer.transform(
        origin_longitude,
        origin_latitude
    )

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

    utm_zone = int((origin_longitude + 180) // 6) + 1
    epsg = 32600 + utm_zone

    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg}",
        always_xy=True
    )

    origin_east, origin_north = transformer.transform(
        origin_longitude,
        origin_latitude
    )

    east = x + origin_east
    north = y + origin_north

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
