import numpy as np
import folium
import csv
import os

def save_region_data_to_csv(region_data, region_name, latitude_list, longitude_list, surface_vars):
    """
    Saves region data to CSV files.

    Parameters:
        region_data (numpy.ndarray): Region forecast data.
        region_name (str): Name of the region.
        latitude_list (list): List of latitude values.
        longitude_list (list): List of longitude values.
        surface_vars (list): List of variable names.

    Returns:
    """
    # Create directory for sub-region if it doesn't exist
    sub_region_dir = os.path.join("Sub-region", region_name)
    os.makedirs(sub_region_dir, exist_ok=True)

    for i, var_name in enumerate(surface_vars):
        # Create CSV filename based on variable name
        csv_filename =  os.path.join(sub_region_dir, f"{var_name}.csv")

        # Open CSV file for writing
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Write header
            writer.writerow(['Lat', 'Lon', var_name])

            # Iterate over latitude and longitude indices
            for lat_index, lat in enumerate(latitude_list):
                for lon_index, lon in enumerate(longitude_list):
                    # Write data row (latitude, longitude, variable value)
                    writer.writerow([lat, lon, region_data[i, lat_index, lon_index]])


def map_bounding_box(result_dir,region_name,lat_min, lat_max, lon_min, lon_max):
    # Bounding box 
    print("Bounding Box for the Region:")
    print("Latitude Range:", [lat_min, lat_max])
    print("Longitude Range:", [lon_min, lon_max])

    # Center of the bounding box
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    # Folium map centered at the bounding box
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
    # Add a rectangle representing the bounding box
    folium.Rectangle(bounds=[[lat_min, lon_min], [lat_max, lon_max]], color='blue', fill=True, fill_opacity=0.2).add_to(m)
    file_name = "bounding_box_{}.html".format(region_name)
    m.save(os.path.join(result_dir, file_name))

def extract_region(output_surface, output_upper, lat_min, lat_max, lon_min, lon_max):
    """
    Extracts the surface forecast data for a specified region and saves it.

    Parameters:
        output_surface (numpy.ndarray): Global surface variable forecast array.
        output_upper (numpy.ndarray): Global upper variable forecast array.
        region_name (str): Name of the region for file naming.
        lat_min (float): Minimum latitude of the region.
        lat_max (float): Maximum latitude of the region.
        lon_min (float): Minimum longitude of the region.
        lon_max (float): Maximum longitude of the region.
    
    Returns:
        numpy.ndarray: Forecast data for the specified region.
    """

    # Convert latitude and longitude bounds to indices
    lat_values = np.linspace(90, -90, 721)  # Latitude values from 90 to -90 with 721 values
    lon_values = np.linspace(0.0000e+00, 3.5975e+02, 1440)  # Longitude values from 0 to 360 with 1440 values

    lat_indices = np.where((lat_values >= lat_min) & (lat_values <= lat_max))[0]
    lon_indices = np.where((lon_values >= lon_min) & (lon_values <= lon_max))[0]
    lat_mesh, lon_mesh = np.meshgrid(lat_indices, lon_indices, indexing='ij')
    lat_lon_combinations = np.stack((lat_mesh, lon_mesh), axis=-1)

    # Extract region forecast data
    region_surface_data = output_surface[:, lat_lon_combinations[:, :, 0], lat_lon_combinations[:, :, 1]]
    region_upper_data = output_upper[:, :, lat_lon_combinations[:, :, 0], lat_lon_combinations[:, :, 1]]

    # Extract latitude and longitude values for the specified region
    latitude_list = lat_values[lat_indices]
    longitude_list = lon_values[lon_indices]

    return region_surface_data, region_upper_data, latitude_list, longitude_list
