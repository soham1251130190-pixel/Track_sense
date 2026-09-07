from loader import load_io_vnbd
from projection import latlon_to_local_xy


# Load the IO-VNBD dataset
df = load_io_vnbd()

# Convert GPS coordinates to local metres
x, y = latlon_to_local_xy(
    df["GPS LATITUDE (degrees)"],
    df["GPS LONGITUDE (degrees)"]
)

# Add local coordinates to the dataframe
df["x_m"] = x
df["y_m"] = y

print("\n===== LOCAL COORDINATES =====")
print(df[[
    "GPS LATITUDE (degrees)",
    "GPS LONGITUDE (degrees)",
    "x_m",
    "y_m"
]].head(10).to_string(index=False))

print("\n===== TRAJECTORY EXTENT =====")
print(f"X range: {df['x_m'].min():.2f} m → {df['x_m'].max():.2f} m")
print(f"Y range: {df['y_m'].min():.2f} m → {df['y_m'].max():.2f} m")