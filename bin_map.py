import argparse
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely.geometry as sgeom
import os
import xarray as xr
import numpy as np
import pandas as pd

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--grid-path", default="/lustre_scratch/shaerdan/maps/EOCIS-CHUK-GRID-100M-v1.0.nc")
parser.add_argument("--exp-name", default="O6eZ55rh", help="Experiment name")
parser.add_argument("--cases", nargs='+', choices=["train", "test", "both"], default="both", help="Cases to plot: train, test, or both")
args = parser.parse_args()

path = args.grid_path
exp_name = args.exp_name
cases = args.cases if args.cases != "both" else ["train", "test"]

# Open the target grid, just to get the UK bounding box
ds = xr.open_dataset(path)

# Work out the area of interest - the UK
x_min = 0
x_max = int(ds.x.max())
y_min = 0
y_max = int(ds.y.max())

print(x_min, x_max, y_min, y_max)

# Function to compute RMSE and prepare data for plotting
def prepare_rmse_data(case, exp_name, sz=10000):
    test_scores_path = f"/lustre_scratch/shaerdan/scores/{case}_scores_{exp_name}.nc"
    ds_test = xr.open_dataset(test_scores_path)
    
    # Extract the prediction and ground truth
    prediction = ds_test.hires_estimate.isel(model_output_channel=0)
    ground_truth = ds_test.ST_slices.isel(channel=0)
    
    # Drop coordinates from ground_truth that are not in prediction
    ground_truth = ground_truth.drop_vars([coord for coord in ground_truth.coords if coord not in prediction.dims])
    ground_truth.attrs.pop('units', None)
    ground_truth.attrs.pop('standard_name', None)
    prediction = prediction.rename({'model_output_y': 'y', 'model_output_x': 'x'})
    
    # Ensure that prediction and ground_truth have the same shape
    assert prediction.shape == ground_truth.shape, "Shapes of prediction and ground_truth do not match"
    
    # Compute RMSE for each box
    rmse = np.sqrt(((prediction - ground_truth) ** 2).mean(dim=['y', 'x']))
    
    # Assuming x_eastings and y_northings are coordinates in ds_test and are 2D arrays with shape (4630, 100)
    x_eastings = ds_test.x_eastings.values
    y_northings = ds_test.y_northings.values
        
    # Calculate the center coordinates for each box
    x_centers = np.round((x_eastings[:, 0] + 5000) / sz) * sz
    y_centers = np.round((y_northings[:, -1] + 5000) / sz) * sz
    
    # Combine RMSE with center coordinates into a DataFrame
    rmse_df = pd.DataFrame({
        'x_centers': x_centers,
        'y_centers': y_centers,
        'rmse': rmse.values
    })
    
    # Group by x_centers and y_centers and calculate the mean RMSE
    mean_rmse_df = rmse_df.groupby(['x_centers', 'y_centers'])['rmse'].mean().reset_index()
    
    return mean_rmse_df

rmse_data = {case: prepare_rmse_data(case, exp_name) for case in cases}

#  maximum RMSE value for consistent color mapping
max_rmse = max(data['rmse'].max() for data in rmse_data.values())
min_rmse = 0  # You can adjust this if you have a different minimum RMSE value to consider

# Plot the selected cases on the same map
fig, axes = plt.subplots(len(cases), 1, figsize=(12, 24), subplot_kw={'projection': ccrs.OSGB()})

if len(cases) == 1:
    axes = [axes]

cmap = plt.cm.jet
sz = 10000

for ax, case in zip(axes, cases):
    rmse_df = rmse_data[case]
    ax.set_extent([x_min, x_max, y_min, y_max], crs=ccrs.OSGB())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    
    # Plot RMSE on the map
    for i in range(len(rmse_df)):
        x_center = rmse_df.loc[i, 'x_centers']
        y_center = rmse_df.loc[i, 'y_centers']
        rmse_value = rmse_df.loc[i, 'rmse']
        
        norm_rmse = (min(max(rmse_value, min_rmse), max_rmse) - min_rmse) / (max_rmse - min_rmse)
        col = cmap(norm_rmse)
        
        # Create a polygon for the bin
        polygon = sgeom.Polygon(shell=[
            (x_center - sz / 2, y_center - sz / 2),
            (x_center - sz / 2, y_center + sz / 2),
            (x_center + sz / 2, y_center + sz / 2),
            (x_center + sz / 2, y_center - sz / 2)
        ])
        
        # Add the polygon to the plot
        ax.add_geometries([polygon], ccrs.OSGB(), facecolor=col)
    
    ax.set_title(f"{case.capitalize()} RMSE")

# Add color bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_rmse, vmax=max_rmse))
sm._A = []  # Dummy array for the scalar mappable
cbar = plt.colorbar(sm, ax=axes, orientation='horizontal', pad=0.05, aspect=50)
cbar.set_label('RMSE')

plt.savefig(f"plot_rmse_{cases}.png")
plt.show()
