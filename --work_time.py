import numpy as np

def calculate_work_times(forecast_time):
    work_times = np.zeros(4, dtype=np.int32)
    work_times[0] = forecast_time//24
    work_times[1] = (forecast_time%24)//6
    work_times[2] = (forecast_time%24%6)//3
    work_times[3] = forecast_time%24%6%3

    return work_times




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



def extract_region_surface(output_surface,region_name, lat_min, lat_max, lon_min, lon_max):
    """
    Extracts the surface forecast data for a specified region and saves a map with a bounding box.

    Parameters:
        output_surface (numpy.ndarray): Global surface variable forecast array.
        region_name (str): Name of the region for file naming.
        lat_min (float): Minimum latitude of the region.
        lat_max (float): Maximum latitude of the region.
        lon_min (float): Minimum longitude of the region.
        lon_max (float): Maximum longitude of the region.
    
    Returns:
        numpy.ndarray: Forecast data for the specified region.

    Example usage:
        north_sea_forecast_data = extract_region_surface(output_surface,region_name, lat_min, lat_max, lon_min, lon_max)

    """
    # Create directory for sub-region if it doesn't exist
    sub_region_dir = os.path.join("Sub-region", region_name)
    os.makedirs(sub_region_dir, exist_ok=True)

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
    folium.Rectangle(bounds=[[lat_min, lon_min], [lat_max, lon_max]], color='blue', fill_opacity=0.1).add_to(m)
    file_name = "bounding_box_{}.html".format(region_name)
    m.save(os.path.join(sub_region_dir, file_name))

    # Define the grid resolution
    grid_resolution = 0.25  # in degrees

    # Define the starting longitude and latitude
    start_longitude = -180
    start_latitude = -90

    # Convert latitude and longitude bounds to indices
    lat_indices = np.arange(int((lat_max - start_latitude) / grid_resolution) + 1, int((lat_min - start_latitude) / grid_resolution) - 1, -1)
    lon_indices = np.arange(int((lon_min - start_longitude) / grid_resolution), int((lon_max - start_longitude) / grid_resolution) + 1)

    # Convert indices to coordinates for the specified region
    region_coordinates = []
    Latitude = set()  # List to store  unique latitudes
    Longitude = set()  # List to store unique longitude
    for lat_index in lat_indices:
        for lon_index in lon_indices:
            longitude = start_longitude + (lon_index * grid_resolution)
            latitude = start_latitude + (lat_index * grid_resolution)
            region_coordinates.append((latitude, longitude))
            Latitude.add(latitude)
            Longitude.add(longitude)

    print("Coordinates within the bounding box:")
    print(region_coordinates)

    # Extract separate lists of latitude and longitude
    latitude_list = sorted(list(Latitude))
    longitude_list = sorted(list(Longitude))
    print("List of Latitudes:", latitude_list)
    print("List of Longitudes:", longitude_list)

    # Extract region forecast data
    region_surface_data = output_surface[:, lat_indices, :][:, :, lon_indices]

    # Save region forecast data to folder
    np.save(os.path.join(sub_region_dir, f"surface_data_{region_name}.npy"), region_surface_data)

    return region_surface_data, latitude_list, longitude_list

