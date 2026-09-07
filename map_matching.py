"""
map_matching.py
----------------
Person 3 (Fusion Engineer) — Hr 6-10 deliverable.

Takes the raw (x, y) trajectory your EKF produces and snaps it onto a real
road network, so the position shown to the user always sits on an actual
road instead of drifting into a field or building.

This file has TWO matcher implementations:

1. GraphHopperClient  -- the REAL integration, talks to an actual GraphHopper
   server's /match endpoint over HTTP. This is what ships in the final
   project. Requires a running GraphHopper instance with an OpenStreetMap
   extract loaded for your test region (see setup notes at the bottom).

2. NaiveNearestPointMatcher -- a simple stand-in used ONLY for testing the
   plumbing today, before a real GraphHopper server exists. It just snaps
   each point to the closest point on a known road polyline. This is
   explicitly NOT what your project spec calls for (the spec wants
   Hidden-Markov-based matching, which handles ambiguity at junctions and
   parallel roads -- this naive version does not). Its only job is proving
   your pipeline's plumbing (EKF output -> matcher -> corrected output)
   works end to end, the same "walking skeleton" idea as Sync 1.

COORDINATE NOTE: your EKF works in local x/y meters. GraphHopper (and real
road data) works in lat/lon. lat/lon <-> local xy conversion now lives in
the SHARED projection.py (Person 1 + Person 3 both import from it), using
UTM via pyproj -- NOT the flat-earth approximation this file used to have
inline. This guarantees your x/y numbers agree with Person 1's GPS
pipeline and your EKF's expected input, since all three now go through
the exact same origin + projection math.
"""

import numpy as np
from projection import latlon_to_local_xy, local_xy_to_latlon  # noqa: F401 -- re-exported for convenience


# ======================================================================
# 1. REAL INTEGRATION: GraphHopper /match client
# ======================================================================
class GraphHopperClient:
    """
    Talks to a real, running GraphHopper server's map-matching endpoint.

    Setup needed BEFORE this works (not done in this sandbox):
        1. Download GraphHopper (a .jar file) from graphhopper.com
        2. Download an OpenStreetMap extract (.osm.pbf) for your test
           region from download.geofabrik.de
        3. Run GraphHopper with the map-matching module enabled, pointed
           at that extract -- this starts a local server, typically at
           http://localhost:8989
        4. Point base_url below at that running server

    Usage once a server exists:
        client = GraphHopperClient(base_url="http://localhost:8989")
        matched = client.match_points(latlon_points, timestamps)
    """

    def __init__(self, base_url="http://localhost:8989", vehicle_profile="car"):
        self.base_url = base_url.rstrip("/")
        self.vehicle_profile = vehicle_profile

    def match_points(self, latlon_points, timestamps=None):
        """
        latlon_points: list of (lat, lon) tuples -- your EKF's trajectory,
                       already converted from local x/y via local_xy_to_latlon().
        timestamps:    optional list of unix-epoch-ms timestamps, same length
                       as latlon_points. GraphHopper uses these to help
                       disambiguate speed-implausible jumps between points.

        Returns: list of (lat, lon) tuples -- the same trajectory, corrected
        to lie on real road geometry.

        Raises: requests.exceptions.ConnectionError if no server is running
        at base_url -- this is expected until Step 3 above is done by you
        or a teammate.
        """
        import requests

        # Build a GPX-like point list, which is what GraphHopper's /match
        # endpoint expects as its "points" payload.
        points_payload = []
        for i, (lat, lon) in enumerate(latlon_points):
            entry = {"lat": lat, "lon": lon}
            if timestamps is not None:
                entry["time"] = timestamps[i]
            points_payload.append(entry)

        response = requests.post(
            f"{self.base_url}/match",
            params={"vehicle": self.vehicle_profile},
            json={"points": points_payload},
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()

        # GraphHopper returns matched points nested inside the response's
        # path geometry -- extract just the (lat, lon) pairs.
        matched_coords = result["paths"][0]["points"]["coordinates"]
        # GeoJSON order is [lon, lat], NOT [lat, lon] -- easy bug to miss.
        matched_latlon = [(lat, lon) for lon, lat in matched_coords]
        return matched_latlon


# ======================================================================
# 2. STAND-IN MATCHER: for testing plumbing today, without a real server
# ======================================================================
class NaiveNearestPointMatcher:
    """
    NOT a real map-matcher. Snaps each point independently to the closest
    point on a single known road polyline. Does not use a Hidden Markov
    Model, has no memory between points, and would break at junctions or
    parallel roads -- exactly the failure modes your project spec calls
    out as reasons to use a real HMM-based tool instead.

    Use this ONLY to prove your pipeline's plumbing (EKF output -> matcher
    -> corrected trajectory) runs end to end today. Swap for
    GraphHopperClient the moment a real server exists.
    """

    def __init__(self, road_polyline_xy):
        """road_polyline_xy: list of (x, y) points defining one road, in
        the same local meters frame as your EKF's output."""
        self.road = np.array(road_polyline_xy)

    def match_points(self, trajectory_xy):
        """
        trajectory_xy: list/array of (x, y) points -- your EKF's raw output.
        Returns: array of (x, y) points, each snapped to the nearest point
        on the road polyline.
        """
        snapped = []
        for point in trajectory_xy:
            snapped.append(self._snap_to_polyline(np.array(point)))
        return np.array(snapped)

    def _snap_to_polyline(self, point):
        """Finds the closest point to `point` across every segment of the
        road polyline, by projecting onto each segment and clamping to
        the segment's endpoints."""
        best_point = None
        best_dist = float("inf")

        for i in range(len(self.road) - 1):
            a, b = self.road[i], self.road[i + 1]
            ab = b - a
            ab_len_sq = np.dot(ab, ab)
            if ab_len_sq == 0:
                continue
            t = np.dot(point - a, ab) / ab_len_sq
            t = np.clip(t, 0.0, 1.0)  # stay within this segment, don't overshoot
            projected = a + t * ab
            dist = np.linalg.norm(point - projected)
            if dist < best_dist:
                best_dist = dist
                best_point = projected

        return best_point
