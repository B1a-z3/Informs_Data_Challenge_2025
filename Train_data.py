import xarray as xr
import pandas as pd

#Opening the netcdf file
ds = xr.open_dataset("train.nc")
print(ds)

#Converting .nc file to dataframe
df = ds.to_dataframe().reset_index()        
print(df.head())

#Copy of the dataframe
df1 = df.copy()

#Pivoting the dataframe to wide format
train_wide = df1.pivot_table(index=["timestamp", "location", "out","tracked"], columns="feature", values="weather").reset_index()
print(train_wide.head())
print(train_wide.tail())

