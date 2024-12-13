import os
import numpy as np
import xarray as xr
import pandas as pd
#!pip install git+https://github.com/google-research/weatherbench2.git
#!pip install gcsfs
import apache_beam  
import weatherbench2


def kelvin_to_celsius(k):
    return k - 273.15

def convert_temperature_to_celsius(df):
    temperature_cols = [col for col in df.columns if 'temperature' in col]
    for col in temperature_cols:
        df[col] = kelvin_to_celsius(df[col])

    return df

def load_forecast_surface_variable_v2(start_date, end_date, hours, region_name, surface_variables, specified_latitudes=None, specified_longitudes=None):
    # improveed version of load_forecast_surface_variable
    datetime_range = pd.date_range(start=start_date, end=end_date, freq='D')
    dfs = []  # for each file

    for date in datetime_range:
        for hour in hours:
            forecast_array_path = f'./output_{region_name}/{date.strftime("%Y-%m-%d")}/output_surface_{date.strftime("%Y%m%d")}_{hour}.npy'
            latitude_file_path = f'./output_{region_name}/{date.strftime("%Y-%m-%d")}/latitude_list.txt'
            longitude_file_path = f'./output_{region_name}/{date.strftime("%Y-%m-%d")}/longitude_list.txt'
            
            if os.path.exists(forecast_array_path) and os.path.exists(latitude_file_path) and os.path.exists(longitude_file_path):
                forecast_array = np.load(forecast_array_path)
                forecast_latitudes = np.loadtxt(latitude_file_path)
                forecast_longitudes = np.loadtxt(longitude_file_path)
                variable_names = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']

                dfs_per_file = []  # each coordinate combination in the current file

                for lat_idx, lat in enumerate(forecast_latitudes):
                    for lon_idx, lon in enumerate(forecast_longitudes):
                        # Check if the current lat/lon should be included
                        if specified_latitudes and lat not in specified_latitudes:
                            continue
                        if specified_longitudes and lon not in specified_longitudes:
                            continue
                        
                        timestamp = pd.Timestamp(date) + pd.Timedelta(hours=int(hour)) 
                        forecast_values = []
                        for var_idx, var in enumerate(variable_names):
                            value = forecast_array[var_idx, lat_idx, lon_idx]
                            forecast_values.append(value)
                        
                        # for current coordinate combination
                        df_dict = {
                            'forecast_lat': lat,
                            'forecast_lon': lon,
                            'Timestamp': timestamp
                        }
                        for var, value in zip(variable_names, forecast_values):
                            df_dict[f'forecast_{var}'] = value
                        
                        df = pd.DataFrame(df_dict, index=[0])  # Pass index explicitly
                        dfs_per_file.append(df)

                # per file along the rows
                if dfs_per_file:
                    dfs.append(pd.concat(dfs_per_file, ignore_index=True))

    # for all files along the rows
    if dfs:
        forecast_df = pd.concat(dfs, ignore_index=True)
        selected_columns = ['forecast_lat', 'forecast_lon', 'Timestamp'] + [f'forecast_{var}' for var in surface_variables]
        forecast_df = forecast_df.filter(selected_columns, axis=1)

        return forecast_df
    else:
        print("No forecast data found.")
        return None


def load_forecast_surface_variable(start_date, end_date, hours, region_name, surface_variables):
    datetime_range = pd.date_range(start=start_date, end=end_date, freq='D')
    dfs = []  # for each file

    for date in datetime_range:
        for hour in hours:
            forecast_array_path = f'./output_{region_name}/{date.strftime("%Y-%m-%d")}/output_surface_{date.strftime("%Y%m%d")}_{hour}.npy'
            latitude_file_path = f'./output_{region_name}/{date.strftime("%Y-%m-%d")}/latitude_list.txt'
            longitude_file_path = f'./output_{region_name}/{date.strftime("%Y-%m-%d")}/longitude_list.txt'
            
            if os.path.exists(forecast_array_path) and os.path.exists(latitude_file_path) and os.path.exists(longitude_file_path):
                forecast_array = np.load(forecast_array_path)
                forecast_latitudes = np.loadtxt(latitude_file_path)
                forecast_longitudes = np.loadtxt(longitude_file_path)
                variable_names = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']

                dfs_per_file = []  # each coordinate combination in the current file

                for lat_idx, lat in enumerate(forecast_latitudes):
                    for lon_idx, lon in enumerate(forecast_longitudes):
                        timestamp = pd.Timestamp(date) + pd.Timedelta(hours=int(hour)) 
                        forecast_values = []
                        for var_idx, var in enumerate(variable_names):
                            value = forecast_array[var_idx, lat_idx, lon_idx]
                            forecast_values.append(value)
                        
                        # for current coordinate combination
                        df_dict = {
                            'forecast_lat': lat,
                            'forecast_lon': lon,
                            'Timestamp': timestamp
                        }
                        for var, value in zip(variable_names, forecast_values):
                            df_dict[f'forecast_{var}'] = value
                        
                        df = pd.DataFrame(df_dict, index=[0])  # Pass index explicitly
                        dfs_per_file.append(df)

                # per file along the rows
                if dfs_per_file:
                    dfs.append(pd.concat(dfs_per_file, ignore_index=True))

    # for all files along the rows
    if dfs:
        forecast_df = pd.concat(dfs, ignore_index=True)
        selected_columns = ['forecast_lat', 'forecast_lon', 'Timestamp'] + [f'forecast_{var}' for var in surface_variables]
        forecast_df = forecast_df.filter(selected_columns, axis=1)

        return forecast_df
    else:
        print("No forecast data found.")
        return None


def load_forecast_upper_variable_v2(start_date, end_date, hours, region_name, upper_variables, levels, specified_latitudes=None, specified_longitudes=None):
    # improveed version of load_forecast_upper_variable
    datetime_range = pd.date_range(start=start_date, end=end_date, freq='D')
    dfs = []  # for each file

    for date in datetime_range:
        for hour in hours:
            forecast_array_path = f'./output_{region_name}/{date.strftime("%Y-%m-%d")}/output_upper_{date.strftime("%Y%m%d")}_{hour}.npy'
            latitude_file_path = f'./output_{region_name}/{date.strftime("%Y-%m-%d")}/latitude_list.txt'
            longitude_file_path = f'./output_{region_name}/{date.strftime("%Y-%m-%d")}/longitude_list.txt'
            
            if os.path.exists(forecast_array_path) and os.path.exists(latitude_file_path) and os.path.exists(longitude_file_path):
                forecast_array = np.load(forecast_array_path)
                forecast_latitudes = np.loadtxt(latitude_file_path)
                forecast_longitudes = np.loadtxt(longitude_file_path)
                variable_names = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']
                pressure_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
                
                dfs_per_file = []  # for each coordinate combination in the current file

                for lat_idx, lat in enumerate(forecast_latitudes):
                    for lon_idx, lon in enumerate(forecast_longitudes):
                        # Check if the current lat/lon should be included
                        if specified_latitudes and lat not in specified_latitudes:
                            continue
                        if specified_longitudes and lon not in specified_longitudes:
                            continue

                        timestamp = pd.Timestamp(date) + pd.Timedelta(hours=int(hour)) 
                        forecast_values = []
                        for var_idx, var in enumerate(variable_names):
                            var_values = []
                            for level_idx, level in enumerate(pressure_levels):
                                value = forecast_array[var_idx, level_idx, lat_idx, lon_idx]
                                var_values.append(value)
                            forecast_values.append(var_values)
                        
                        # for current coordinate combination
                        df_dict = {
                            'forecast_lat': lat,
                            'forecast_lon': lon,
                            'Timestamp': timestamp
                        }
                        for var, var_values in zip(variable_names, forecast_values):
                            for level, value in zip(pressure_levels, var_values):
                                df_dict[f'forecast_{var}_{level}'] = value
                        
                        df = pd.DataFrame(df_dict, index=[0])  # Pass index explicitly
                        dfs_per_file.append(df)

                # concatenate per file along the rows
                if dfs_per_file:
                    dfs.append(pd.concat(dfs_per_file, ignore_index=True))

    # concatenate for all files along the rows
    if dfs:
        forecast_df = pd.concat(dfs, ignore_index=True)
        # select specified upper_variables and levels
        selected_columns = ['forecast_lat', 'forecast_lon', 'Timestamp'] + [f'forecast_{var}_{level}' for var in upper_variables for level in levels]
        forecast_df = forecast_df.filter(selected_columns, axis=1)

        return forecast_df
    else:
        print("No forecast data found.")
        return None


def load_forecast_upper_variable(start_date, end_date, hours, region_name, upper_variables, levels):
    datetime_range = pd.date_range(start=start_date, end=end_date, freq='D')
    dfs = []  #  for each file

    for date in datetime_range:
        for hour in hours:
            forecast_array_path = f'./output_{region_name}/{date.strftime("%Y-%m-%d")}/output_upper_{date.strftime("%Y%m%d")}_{hour}.npy'
            latitude_file_path = f'./output_{region_name}/{date.strftime("%Y-%m-%d")}/latitude_list.txt'
            longitude_file_path = f'./output_{region_name}/{date.strftime("%Y-%m-%d")}/longitude_list.txt'
            
            if os.path.exists(forecast_array_path) and os.path.exists(latitude_file_path) and os.path.exists(longitude_file_path):
                forecast_array = np.load(forecast_array_path)
                forecast_latitudes = np.loadtxt(latitude_file_path)
                forecast_longitudes = np.loadtxt(longitude_file_path)
                variable_names = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']
                pressure_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
                
                dfs_per_file = []  #  for each coordinate combination in the current file

                for lat_idx, lat in enumerate(forecast_latitudes):
                    for lon_idx, lon in enumerate(forecast_longitudes):
                        timestamp = pd.Timestamp(date) + pd.Timedelta(hours=int(hour)) 
                        forecast_values = []
                        for var_idx, var in enumerate(variable_names):
                            var_values = []
                            for level_idx, level in enumerate(pressure_levels):
                                value = forecast_array[var_idx, level_idx, lat_idx, lon_idx]
                                var_values.append(value)
                            forecast_values.append(var_values)
                        
                        #  for current coordinate combination
                        df_dict = {
                            'forecast_lat': lat,
                            'forecast_lon': lon,
                            'Timestamp': timestamp
                        }
                        for var, var_values in zip(variable_names, forecast_values):
                            for level, value in zip(pressure_levels, var_values):
                                df_dict[f'forecast_{var}_{level}'] = value
                        
                        df = pd.DataFrame(df_dict, index=[0])  # Pass index explicitly
                        dfs_per_file.append(df)

                # concatenate per file along the rows
                if dfs_per_file:
                    dfs.append(pd.concat(dfs_per_file, ignore_index=True))

    # concatenate for all files along the rows
    if dfs:
        forecast_df = pd.concat(dfs, ignore_index=True)
        # select specified upper_variables and levels
        selected_columns = ['forecast_lat', 'forecast_lon', 'Timestamp'] + [f'forecast_{var}_{level}' for var in upper_variables for level in levels]
        forecast_df = forecast_df.filter(selected_columns, axis=1)

        return forecast_df
    else:
        print("No forecast data found.")
        return None
        
def load_reanalysis_surface_variable(start_date, end_date, hours, region_name, surface_variables):
    dfs_dict = {}
    
    datetime_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    for time in datetime_range:
        for hour in hours:
            reanalysis_path = f'./reanalysis_{region_name}/{time.strftime("%Y-%m-%d")}/reanalysis_surface_{hour}.nc'
            if os.path.exists(reanalysis_path):
                ds = xr.open_dataset(reanalysis_path)
                for variable in surface_variables:
                    if isinstance(variable, str) and variable in ds.data_vars:
                        # Extract data for the current variable
                        data = ds[variable].to_dataframe()
                        data.reset_index(inplace=True)
                        data.rename(columns={'latitude': 'reanalysis_lat', 'longitude': 'reanalysis_lon','time': 'Timestamp'}, inplace=True)
                        column_name = f'reanalysis_{variable}'
                        data.rename(columns={variable: column_name}, inplace=True)
            
                        key = (variable,)
                        if key in dfs_dict:
                            dfs_dict[key].append(data)
                        else:
                            dfs_dict[key] = [data]
                    else:
                        print(f"Variable {variable} not found in dataset or not a string in {reanalysis_path}")
            else:
                print(f"Reanalysis file not found for {reanalysis_path}")

    if dfs_dict:
        combined_dfs = {}
        for key, dfs_list in dfs_dict.items():
            combined_df = pd.concat(dfs_list)
            combined_df.set_index(['Timestamp', 'reanalysis_lat', 'reanalysis_lon'], inplace=True)
            combined_dfs[key] = combined_df
        merged_df = merge_dataframes(combined_dfs)
        return merged_df.reset_index()  #### merged_df
    else:
        print("No reanalysis data found.")
        return None
    
def load_reanalysis_upper_variable(start_date, end_date, hours, region_name, upper_variables, levels):
    dfs_dict = {}
    
    datetime_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    for time in datetime_range:
        for hour in hours:
            reanalysis_path = f'./reanalysis_{region_name}/{time.strftime("%Y-%m-%d")}/reanalysis_upper_{hour}.nc'
            if os.path.exists(reanalysis_path):
                ds = xr.open_dataset(reanalysis_path)
                for variable in upper_variables:
                    if isinstance(variable, str) and variable in ds.data_vars:
                        for level in levels:
                            if level in ds[variable].level.values:
                                # load data for the current variable and level
                                data = ds[variable].sel(level=level).to_dataframe()
                                data.reset_index(inplace=True)
                                data.rename(columns={'latitude': 'reanalysis_lat', 'longitude': 'reanalysis_lon','time': 'Timestamp'}, inplace=True)
                                column_name = f'reanalysis_{variable}_{level}'
                                data.rename(columns={variable: column_name}, inplace=True)
                                data.drop(columns='level', inplace=True)
                            #not#if hour == 24:
                    #necessary#   data['time'] = pd.to_datetime(data['time']).dt.strftime('%Y-%m-%d 00:00:00')
                                key = (variable, level)
                                if key in dfs_dict:
                                    dfs_dict[key].append(data)
                                else:
                                    dfs_dict[key] = [data]
                            else:
                                print(f"Level {level} not found for variable {variable} in {reanalysis_path}")
                    else:
                        print(f"Variable {variable} not found in dataset or not a string in {reanalysis_path}")
            else:
                print(f"Reanalysis file not found for {reanalysis_path}")

    if dfs_dict:
        combined_dfs = {}
        for key, dfs_list in dfs_dict.items():
            combined_df = pd.concat(dfs_list)
            combined_df.set_index(['Timestamp', 'reanalysis_lat', 'reanalysis_lon'], inplace=True)
            combined_dfs[key] = combined_df
        merged_df = merge_dataframes(combined_dfs)
        return merged_df.reset_index() #### merged_df
    else:
        print("No reanalysis data found.")
        return None

def load_comparison_data(start_date, end_date, hours, region_name, upper_variables=None, surface_variables=None, levels=None):
    if upper_variables is None and surface_variables is None:
        print("Please specify either upper_variables, surface_variables, or both.")
        return None
    
    if levels is None:
        levels = [1000, 850]  # Default levels 
    
    comparison_data = pd.DataFrame()  
    
    if upper_variables:
        reanalysis_upper_data = load_reanalysis_upper_variable(start_date, end_date, hours, region_name, upper_variables, levels)
        forecast_upper_data = load_forecast_upper_variable(start_date, end_date, hours, region_name, upper_variables, levels)
        comparison_data_upper = pd.concat([reanalysis_upper_data.set_index('Timestamp'), forecast_upper_data.set_index('Timestamp')], axis=1)
        comparison_data = pd.concat([comparison_data, comparison_data_upper], axis=1)
        comparison_data = comparison_data.loc[:, ~comparison_data.columns.duplicated()]
    if surface_variables:
        reanalysis_surface_data = load_reanalysis_surface_variable(start_date, end_date, hours, region_name, surface_variables)
        forecast_surface_data = load_forecast_surface_variable(start_date, end_date, hours, region_name, surface_variables)
        comparison_data_surface = pd.concat([reanalysis_surface_data.set_index('Timestamp'), forecast_surface_data.set_index('Timestamp')], axis=1)
        comparison_data = pd.concat([comparison_data, comparison_data_surface], axis=1)
        comparison_data = comparison_data.loc[:, ~comparison_data.columns.duplicated()]
    return comparison_data

def merge_dataframes(combined_dfs):
    merged_df = None
    for key, df in combined_dfs.items():
        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.merge(df, how='outer', left_index=True, right_index=True)
    return merged_df

def extract_weatherbench_reanalysis_data_v2(start_date, end_date, hours, region_name, latitudes=None, longitudes=None): 
    #improved version
    obs_path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr'
    obs = xr.open_zarr(obs_path)
    surface_variables = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']
    upper_variables = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']
    pressure_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

    datetime_range = pd.date_range(start=start_date, end=end_date, freq='D')

    if latitudes is None or longitudes is None:
        lat_file_path = f'output_{region_name}/{datetime_range[0].strftime("%Y-%m-%d")}/latitude_list.txt'
        lon_file_path = f'output_{region_name}/{datetime_range[0].strftime("%Y-%m-%d")}/longitude_list.txt'
        with open(lat_file_path, 'r') as f:
            latitudes = list(map(float, f.read().splitlines()))
        with open(lon_file_path, 'r') as f:
            longitudes = list(map(float, f.read().splitlines()))

    for time in datetime_range:
        output_dir = f'reanalysis_{region_name}/{time.strftime("%Y-%m-%d")}'
        os.makedirs(output_dir, exist_ok=True)

        for hour in hours:
            output_file_surface = os.path.join(output_dir, f'reanalysis_surface_{hour}.nc')
            output_file_upper = os.path.join(output_dir, f'reanalysis_upper_{hour}.nc')

            if os.path.exists(output_file_surface) and os.path.exists(output_file_upper):
                print(f"Data for {time.strftime('%Y-%m-%d')} at hour {hour} already exists. Skipping...")
                continue

            # Ensure the data is selected for the specific time, latitudes, and longitudes
            selected_data = obs.sel(time=time + np.timedelta64(hour, 'h'), latitude=latitudes, longitude=longitudes, method='nearest')

            surface_data = selected_data[surface_variables]
            surface_data.to_netcdf(output_file_surface)

            upper_data = selected_data.sel(level=pressure_levels)[upper_variables]
            upper_data.to_netcdf(output_file_upper)

            print(f"Extracted reanalysis data of {region_name} for {time.strftime('%Y-%m-%d')} at hour {hour}")


def extract_weatherbench_reanalysis_data(start_date, end_date, hours, region_name): 
    obs_path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr'
    obs = xr.open_zarr(obs_path)
    surface_variables = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']
    upper_variables = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']
    pressure_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

    datetime_range = pd.date_range(start=start_date, end=end_date, freq='D')

    lat_file_path = f'output_{region_name}/{datetime_range[0].strftime("%Y-%m-%d")}/latitude_list.txt'
    lon_file_path = f'output_{region_name}/{datetime_range[0].strftime("%Y-%m-%d")}/longitude_list.txt'
    with open(lat_file_path, 'r') as f:
        latitudes = list(map(float, f.read().splitlines()))
    with open(lon_file_path, 'r') as f:
        longitudes = list(map(float, f.read().splitlines()))

    for time in datetime_range:
        output_dir = f'reanalysis_{region_name}/{time.strftime("%Y-%m-%d")}'
        os.makedirs(output_dir, exist_ok=True)

        for hour in hours:
            output_file_surface = os.path.join(output_dir, f'reanalysis_surface_{hour}.nc')
            output_file_upper = os.path.join(output_dir, f'reanalysis_upper_{hour}.nc')

            if os.path.exists(output_file_surface) and os.path.exists(output_file_upper):
                print(f"Data for {time.strftime('%Y-%m-%d')} at hour {hour} already exists. Skipping...")
                continue

            selected_data = obs.sel(time=time + np.timedelta64(hour, 'h'), latitude=latitudes, longitude=longitudes)

            surface_data = selected_data[surface_variables]
            surface_data.to_netcdf(output_file_surface)

            upper_data = selected_data.sel(level=pressure_levels)[upper_variables]
            upper_data.to_netcdf(output_file_upper)

            print(f"Extracted reanalysis data of {region_name} for {time.strftime('%Y-%m-%d')} at hour {hour}")
    
def extract_weatherbench_input_data(start_date,end_date):  # for input (global) 
    obs_path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr'
    obs = xr.open_zarr(obs_path)
    surface_variables = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']
    upper_variables = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']
    pressure_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50] 

    datetime_range = pd.date_range(start=start_date, end=end_date, freq='D')

    for time in datetime_range:
        # select specific variables, time
        selected_surface_data = obs.sel(time=time)[surface_variables]
        selected_upper_data = obs.sel(time=time,level=pressure_levels)[upper_variables]

        output_dir = f'input/{time}/'
        os.makedirs(output_dir, exist_ok=True)

        # .nc to .npy
        selected_surface_data.to_netcdf(os.path.join(output_dir, 'surface.nc'))
        print(f"Saved surface data: {os.path.join(output_dir, 'surface.nc')}")
        surface_data = np.zeros((len(surface_variables), 721, 1440), dtype=np.float32)
        with xr.open_dataset(os.path.join(output_dir, 'surface.nc')) as nc_file:
            for i, var in enumerate(surface_variables):
                surface_data[i] = nc_file[var.lower()].values.astype(np.float32)
        np.save(os.path.join(output_dir, 'input_surface.npy'), surface_data)
        print(f"Saved surface npy data: {os.path.join(output_dir, 'input_surface.npy')}")

    
        selected_upper_data.to_netcdf(os.path.join(output_dir, 'upper.nc'))
        print(f"Saved upper data: {os.path.join(output_dir, 'upper.nc')}")
        upper_data = np.zeros((len(upper_variables), len(pressure_levels), 721, 1440), dtype=np.float32)
        with xr.open_dataset(os.path.join(output_dir, 'upper.nc')) as nc_file:
            for i, var in enumerate(upper_variables):
                upper_data[i] = nc_file[var.lower()].values.astype(np.float32)
        np.save(os.path.join(output_dir, 'input_upper.npy'), upper_data)
        print(f"Saved upper npy data: {os.path.join(output_dir, 'input_upper.npy')}")
