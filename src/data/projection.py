import numpy as np
from pyproj import Transformer


def latlon_to_local_xy(
    latitude,
    longitude,
    origin_latitude=None,
    origin_longitude=None
):
    """
    Convert GPS latitude/longitude to local X/Y coordinates in metres.

    The first GPS point can be used as the local origin:
        X = 0 m
        Y = 0 m

    Returns:
        x, y: local coordinates in metres
    """

    latitude = np.asarray(latitude, dtype=float)
    longitude = np.asarray(longitude, dtype=float)

    if origin_latitude is None:
        origin_latitude = latitude[0]

    if origin_longitude is None:
        origin_longitude = longitude[0]

    # Choose UTM zone from the origin longitude.
    utm_zone = int((origin_longitude + 180) // 6) + 1

    # Northern hemisphere EPSG codes: 32601–32660
    epsg = 32600 + utm_zone

    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg}",
        always_xy=True
    )

    # Project all GPS points.
    east, north = transformer.transform(longitude, latitude)

    # Project the origin.
    origin_east, origin_north = transformer.transform(
        origin_longitude,
        origin_latitude
    )

    # Shift origin to (0, 0).
    x = east - origin_east
    y = north - origin_north

    return x, y
