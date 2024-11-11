import xarray as xr
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

# Reading the data
exp_name = "NNR52DmZ"                     # wroo3fiQ O6eZ55rh LZZkuGAZ   ibZ5x3E5  RGBS_p00     RGBSLANSHEGLA_p00_cleaned  t7Hiwh6A(2000+ boxes)   kzHjKthT (200+ boxes) ME0X3ynk aE4pm07D 8FU3FYuL 8asQEA6P (best so far) n8djAKPf
# train_scores.nc path:
train_scores_path = f"/lustre_scratch/shaerdan/scores/train_scores_{exp_name}.nc"
# test_scores.nc path:
test_scores_path = f"/lustre_scratch/shaerdan/scores/test_scores_{exp_name}.nc"


ds_train_toy = xr.open_dataset(train_scores_path)
ds_test_toy = xr.open_dataset(test_scores_path)

print(ds_train_toy)
print(ds_test_toy)

save_plots_test = True
save_plots_train = True

# Assuming your dataset is loaded as 'ds'
# Extract the land cover and ST_slices variables
land_cover = ds_test_toy['land_cover']  # Shape: (box, land_cover_chan, y, x)
st_slices = ds_test_toy['ST_slices']    # Shape: (box, channel, y, x)

# Initialize a dictionary to store the average temperature for each land cover class
land_cover_classes = np.unique(land_cover)  # Unique land cover classes
average_temps = {}

# Loop through each land cover class and compute the average temperature
for land_class in land_cover_classes:
    # Create a mask for the current land cover class
    mask = (land_cover == land_class)
    
    # Mask the ST_slices with this land cover class mask
    masked_temps = st_slices.where(mask, drop=False)
    
    # Compute the average temperature, ignoring NaN values
    avg_temp = masked_temps.mean().item()
    
    # Store the result in the dictionary
    average_temps[land_class] = avg_temp

# Print the average temperature for each land cover class
for land_class, avg_temp in average_temps.items():
    print(f"Land Cover Class {land_class}: Average Temperature = {avg_temp}")
