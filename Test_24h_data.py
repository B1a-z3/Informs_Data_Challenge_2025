import xarray as xr
import pandas as pd

# Open the NetCDF file
ds = xr.open_dataset("test_24h_demo.nc")

# Convert to dataframe
df = ds.to_dataframe().reset_index()

# Save to CSV
df.to_csv("test_24h_demo.csv", index=False)

print("✅ Conversion complete: output.csv")
