"""
test_map_matching.py
---------------------
Hr 6-10 synthetic test: prove the map-matching plumbing works using a REAL
road (so snapping actually means something) plus FAKE GPS noise on top of
it (so there's something worth correcting).

This mirrors the same pattern as your EKF's synthetic test: fake noise,
real underlying structure, checking the plumbing before real data exists.
"""

import numpy as np
from map_matching import (
    NaiveNearestPointMatcher,
    latlon_to_local_xy,
    local_xy_to_latlon,
)


def generate_noisy_gps_along_real_road(seed=0):
    """
    A real short road stretch: Mumbai-Agra Highway (NH 60) near Nashik,
    Maharashtra -- a fairly straight arterial stretch, picked because
    straight roads make it easy to visually confirm snapping worked
    (noisy zigzag points -> a clean straight line).

    Coordinates are approximate real-world lat/lon along that stretch,
    NOT survey-precise -- good enough to demonstrate the matching logic.
    """
    # Approximate real waypoints along a straight highway stretch (lat, lon)
    real_road_latlon = [
        (19.9975, 73.7898),
        (19.9990, 73.7920),
        (20.0005, 73.7942),
        (20.0020, 73.7964),
        (20.0035, 73.7986),
    ]

    origin_lat, origin_lon = real_road_latlon[0]

    # Convert the real road into the local x/y frame your EKF uses
    road_xy = [
        latlon_to_local_xy(lat, lon, origin_lat, origin_lon)
        for lat, lon in real_road_latlon
    ]
    road_xy = np.array(road_xy)

    # Densify: interpolate points ALONG the road (simulating a car driving
    # it), then add fake GPS noise -- same idea as your EKF's synthetic GPS
    rng = np.random.default_rng(seed)
    n_points = 40
    t_values = np.linspace(0, len(road_xy) - 1, n_points)

    true_xy = np.array([
        [
            np.interp(t, np.arange(len(road_xy)), road_xy[:, 0]),
            np.interp(t, np.arange(len(road_xy)), road_xy[:, 1]),
        ]
        for t in t_values
    ])

    noisy_xy = true_xy + rng.normal(0, 4.0, true_xy.shape)  # ~4m GPS-like noise

    return road_xy, true_xy, noisy_xy, origin_lat, origin_lon


def run_test():
    road_xy, true_xy, noisy_xy, origin_lat, origin_lon = \
        generate_noisy_gps_along_real_road()

    # --- Test the naive stand-in matcher (works today, no server needed) ---
    matcher = NaiveNearestPointMatcher(road_polyline_xy=road_xy)
    matched_xy = matcher.match_points(noisy_xy)

    error_before = np.mean(np.linalg.norm(noisy_xy - true_xy, axis=1))
    error_after = np.mean(np.linalg.norm(matched_xy - true_xy, axis=1))

    print("=== Map-Matching Plumbing Test (Hr 6-10) ===")
    print(f"Mean error BEFORE matching (raw noisy GPS): {error_before:.2f} m")
    print(f"Mean error AFTER matching (snapped to road):  {error_after:.2f} m")

    assert error_after < error_before, (
        "Matching made things WORSE, not better -- check the polyline "
        "projection logic."
    )
    print("PASS -- matching reduced error, plumbing works.\n")

    # --- Demonstrate the lat/lon conversion round-trip (needed for the
    #     REAL GraphHopperClient path, which expects lat/lon, not x/y) ---
    sample_lat, sample_lon = local_xy_to_latlon(
        matched_xy[0, 0], matched_xy[0, 1], origin_lat, origin_lon
    )
    print(f"Sample matched point converted back to lat/lon: "
          f"({sample_lat:.6f}, {sample_lon:.6f})")
    print("This is the (lat, lon) format the REAL GraphHopperClient.match_points()")
    print("would send to an actual GraphHopper server, once one is running.")

    return road_xy, true_xy, noisy_xy, matched_xy


if __name__ == "__main__":
    road_xy, true_xy, noisy_xy, matched_xy = run_test()

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 6))
        plt.plot(road_xy[:, 0], road_xy[:, 1], "g-", linewidth=3,
                 label="Real road", alpha=0.5)
        plt.scatter(noisy_xy[:, 0], noisy_xy[:, 1], color="red", s=20,
                    label="Noisy 'GPS' input")
        plt.scatter(matched_xy[:, 0], matched_xy[:, 1], color="blue", s=20,
                    label="Matched (snapped) output", marker="x")
        plt.legend()
        plt.title("Map-Matching: Noisy Points Snapped Onto Real Road")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.savefig("/home/claude/map_matching_test.png", dpi=150)
        print("\nPlot saved to map_matching_test.png")
    except ImportError:
        print("\n(matplotlib not installed -- skipping plot)")
