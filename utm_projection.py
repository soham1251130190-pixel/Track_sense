import numpy as np
from pyproj import Transformer


def latlon_to_local_xy(
    latitude,
    longitude,
    origin_latitude=None,
    origin_longitude=None
):
    latitude = np.asarray(latitude, dtype=float)
    longitude = np.asarray(longitude, dtype=float)

    if origin_latitude is None:
        origin_latitude = latitude[0]
    if origin_longitude is None:
        origin_longitude = longitude[0]

    utm_zone = int((origin_longitude + 180) // 6) + 1
    epsg = 32600 + utm_zone

    transformer = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}", always_xy=True
    )

    east, north = transformer.transform(longitude, latitude)

    origin_east, origin_north = transformer.transform(
        origin_longitude, origin_latitude
    )

    x = east - origin_east
    y = north - origin_north

    return x, y


def local_xy_to_latlon(
    x,
    y,
    origin_latitude,
    origin_longitude,
):
    """
    Inverse of latlon_to_local_xy above. UNLIKE the forward function,
    origin_latitude/origin_longitude are REQUIRED here (not optional) --
    there's no "first point" to default to when going xy -> latlon, so
    the caller must always supply the same origin that was used going
    the other direction.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Must compute the SAME utm_zone/epsg the same way the forward
    # function does, from the same origin -- otherwise this would
    # silently decode into the wrong coordinate system.
    utm_zone = int((origin_longitude + 180) // 6) + 1
    epsg = 32600 + utm_zone

    fwd_transformer = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}", always_xy=True
    )
    origin_east, origin_north = fwd_transformer.transform(
        origin_longitude, origin_latitude
    )

    east = x + origin_east
    north = y + origin_north

    inv_transformer = Transformer.from_crs(
        f"EPSG:{epsg}", "EPSG:4326", always_xy=True
    )
    lon, lat = inv_transformer.transform(east, north)

    return lat, lon


# ======================================================================
# ROUND-TRIP TEST -- proves the inverse actually matches P1's forward fn
# ======================================================================
if __name__ == "__main__":
    origin_lat, origin_lon = 52.40166, -1.50529  # Real IO-VNBD S1 dataset origin, confirmed by P1

    test_points = [
        (52.40166, -1.50529),    # the origin itself -- should round-trip to (0, 0)
        (52.40200, -1.50480),    # a small nearby offset, simulating a real driven point
        (52.40350, -1.50250),    # a slightly larger offset
    ]

    print("=== Round-trip test: P1's forward fn vs. P3's inverse fn ===\n")
    max_error_m = 0.0

    for lat, lon in test_points:
        x, y = latlon_to_local_xy(lat, lon, origin_lat, origin_lon)
        lat_back, lon_back = local_xy_to_latlon(x, y, origin_lat, origin_lon)

        # Convert the lat/lon discrepancy back to an approximate meters
        # error, for an intuitive number rather than tiny degree deltas
        lat_error_m = abs(lat - lat_back) * 111000
        lon_error_m = abs(lon - lon_back) * 111000 * np.cos(np.radians(lat))
        error_m = np.sqrt(lat_error_m**2 + lon_error_m**2)
        max_error_m = max(max_error_m, error_m)

        print(f"Original:    ({lat:.6f}, {lon:.6f})")
        print(f"  -> xy:     ({float(x):.3f}, {float(y):.3f})")
        print(f"  -> back:   ({float(lat_back):.6f}, {float(lon_back):.6f})")
        print(f"  round-trip error: {error_m*1000:.4f} mm\n")

    assert max_error_m < 0.001, "Round-trip error too large -- inverse doesn't match forward"
    print("PASS -- inverse function correctly reverses P1's forward projection.")
