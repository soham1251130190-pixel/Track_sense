import numpy as np


def calculate_gnss_quality(
    x,
    y,
    speed_kmh,
    timestamps,
    jump_threshold_m=10.0,
    motion_ratio_threshold=10.0
):
    """
    Detect suspicious GNSS position measurements using
    motion consistency.

    A GNSS measurement is flagged when the observed position
    jump is much larger than the movement expected from
    the reported GPS speed and elapsed time.

    Returns:
        gps_jump_m
        expected_motion_m
        motion_ratio
        gnss_outlier
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    speed_kmh = np.asarray(speed_kmh, dtype=float)

    timestamps = np.asarray(timestamps)

    # Position displacement between consecutive samples
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])

    gps_jump_m = np.sqrt(dx**2 + dy**2)

    # Time difference in seconds
    dt = np.zeros(len(timestamps))

    if len(timestamps) > 1:
        dt[1:] = (
            timestamps[1:] - timestamps[:-1]
        ) / np.timedelta64(1, "s")

    # Expected movement from GPS speed
    speed_ms = speed_kmh / 3.6
    expected_motion_m = speed_ms * dt

    # Avoid division by zero
    motion_ratio = np.full(len(x), np.nan)

    valid = expected_motion_m > 0

    motion_ratio[valid] = (
        gps_jump_m[valid] / expected_motion_m[valid]
    )

    # Flag suspicious GNSS measurements
    gnss_outlier = (
        (gps_jump_m > jump_threshold_m)
        & (
            (motion_ratio > motion_ratio_threshold)
            | ~valid
        )
    )

    return (
        gps_jump_m,
        expected_motion_m,
        motion_ratio,
        gnss_outlier
    )