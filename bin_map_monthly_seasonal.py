import argparse
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely.geometry as sgeom
import os
import xarray as xr
import numpy as np
import pandas as pd

# Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--grid-path", default="/lustre_scratch/shaerdan/maps/EOCIS-CHUK-GRID-100M-v1.0.nc")
parser.add_argument("--exp-name", default="NNR52DmZ", help="Experiment name") #0m5TAkjO
parser.add_argument("--cases", nargs='+', choices=["train", "test", "both"], default="both", help="Cases to plot: train, test, or both")
args = parser.parse_args()

path = args.grid_path
exp_name = args.exp_name
cases = args.cases if args.cases != "both" else ["train", "test"]

# Get the bounding box
ds = xr.open_dataset(path)

# Work out the area of interest - the UK
x_min = 0
x_max = int(ds.x.max())
y_min = 0
y_max = int(ds.y.max())

print(x_min, x_max, y_min, y_max)

def prepare_rmse_data(case, exp_name, sz=10000):      # Compute RMSE and prepare data for plotting

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
    
    # Combine RMSE with center coordinates and time into a DataFrame
    rmse_df = pd.DataFrame({
        'x_centers': x_centers,
        'y_centers': y_centers,
        'rmse': rmse.values,
        'box_time': ds_test.box_time.values
    })
    
    # Convert box_time to datetime if it's not already in that format
    if not pd.api.types.is_datetime64_any_dtype(rmse_df['box_time']):
        rmse_df['box_time'] = pd.to_datetime(rmse_df['box_time'])
    
    # Group by x_centers, y_centers, and time to calculate the mean RMSE
    rmse_df['month'] = rmse_df['box_time'].dt.month
    rmse_df['season'] = ((rmse_df['box_time'].dt.month - 1) // 3) + 1
    
    mean_rmse_month_df = rmse_df.groupby(['x_centers', 'y_centers', 'month'])['rmse'].mean().reset_index()
    mean_rmse_season_df = rmse_df.groupby(['x_centers', 'y_centers', 'season'])['rmse'].mean().reset_index()
    
    return mean_rmse_month_df, mean_rmse_season_df

rmse_data = {case: prepare_rmse_data(case, exp_name) for case in cases}

# Get the maximum RMSE value for color map
max_rmse_month = max(data[0]['rmse'].max() for data in rmse_data.values())
max_rmse_season = max(data[1]['rmse'].max() for data in rmse_data.values())
min_rmse = 0  # You can adjust this if you have a different minimum RMSE value to consider

def plot_rmse(rmse_df, title, filename, max_rmse): # To plot the RMSE data
    fig, axes = plt.subplots(4, 3, figsize=(18, 24), subplot_kw={'projection': ccrs.OSGB()})
    cmap = plt.cm.jet
    sz = 10000
    
    for month in range(1, 13):
        ax = axes[(month - 1) // 3, (month - 1) % 3]
        ax.set_extent([x_min, x_max, y_min, y_max], crs=ccrs.OSGB())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        
        rmse_month_df = rmse_df[rmse_df['month'] == month]
        
        if rmse_month_df.empty:
            print(f"No data for month {month}")
            continue
        
        # Plot rmse on the map
        for i in range(len(rmse_month_df)):
            x_center = rmse_month_df.iloc[i]['x_centers']
            y_center = rmse_month_df.iloc[i]['y_centers']
            rmse_value = rmse_month_df.iloc[i]['rmse']
            
            norm_rmse = (min(max(rmse_value, min_rmse), max_rmse) - min_rmse) / (max_rmse - min_rmse)
            col = cmap(norm_rmse)
            
            # Make a polygon for the bin
            polygon = sgeom.Polygon(shell=[
                (x_center - sz / 2, y_center - sz / 2),
                (x_center - sz / 2, y_center + sz / 2),
                (x_center + sz / 2, y_center + sz / 2),
                (x_center + sz / 2, y_center - sz / 2)
            ])
            
            # Add the polygon to the plot
            ax.add_geometries([polygon], ccrs.OSGB(), facecolor=col)
        
        # Calculate and add the mean RMSE value to the subplot
        mean_rmse_value = rmse_month_df['rmse'].mean()
        ax.text(0.5, -0.1, f"Mean RMSE: {mean_rmse_value:.2f}", ha="center", transform=ax.transAxes, fontsize=10)
        
        ax.set_title(f"Month {month}")
    
    # Color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_rmse, vmax=max_rmse))
    sm._A = []  # Dummy array for the scalar mappable
    cbar = plt.colorbar(sm, ax=axes, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label('RMSE')
    
    plt.suptitle(title)
    plt.savefig(filename)
    plt.show()

def plot_rmse_season(rmse_df, title, filename, max_rmse):     # Plot RMSE data by season:
    fig, axes = plt.subplots(2, 2, figsize=(18, 24), subplot_kw={'projection': ccrs.OSGB()})
    cmap = plt.cm.jet
    sz = 10000
    
    for season in range(1, 5):
        ax = axes[(season - 1) // 2, (season - 1) % 2]
        ax.set_extent([x_min, x_max, y_min, y_max], crs=ccrs.OSGB())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        
        rmse_season_df = rmse_df[rmse_df['season'] == season]
        
        if rmse_season_df.empty:
            print(f"No data for season {season}")
            continue
        
        for i in range(len(rmse_season_df)):
            x_center = rmse_season_df.iloc[i]['x_centers']
            y_center = rmse_season_df.iloc[i]['y_centers']
            rmse_value = rmse_season_df.iloc[i]['rmse']
            
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
        
        # Calculate and add the mean RMSE value to the subplot
        mean_rmse_value = rmse_season_df['rmse'].mean()
        ax.text(0.5, -0.1, f"Mean RMSE: {mean_rmse_value:.2f}", ha="center", transform=ax.transAxes, fontsize=10)
        
        ax.set_title(f"Season {season}")
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_rmse, vmax=max_rmse))
    sm._A = []  # Dummy array for the scalar mappable
    cbar = plt.colorbar(sm, ax=axes, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label('RMSE')
    
    plt.suptitle(title)
    plt.savefig(filename)
    plt.show()

# Plot RMSE for each month
for case in cases:
    mean_rmse_month_df, mean_rmse_season_df = rmse_data[case]
    plot_rmse(mean_rmse_month_df, f"{case.capitalize()} RMSE by Month", f"plot_rmse_{case}_monthly_{exp_name}.png", max_rmse_month)

# Plot RMSE for each season
for case in cases:
    mean_rmse_month_df, mean_rmse_season_df = rmse_data[case]
    plot_rmse_season(mean_rmse_season_df, f"{case.capitalize()} RMSE by Season", f"plot_rmse_{case}_seasonal_{exp_name}.png", max_rmse_season)
