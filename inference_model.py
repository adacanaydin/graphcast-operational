import os
import numpy as np
import onnxruntime as ort
import pandas as pd
import datetime
import psutil
from utulities.subregion import map_bounding_box, extract_region



def run_inference_region(ort_session, input_data, result_dir, region_name, lat_min, lat_max, lon_min, lon_max, interval):
    # Plotted bounding box on the map 
    #map_bounding_box(result_dir, region_name, lat_min, lat_max, lon_min, lon_max)

    # Extract input arrays from input_data
    input_upper = input_data['input_upper']
    input_surface = input_data['input_surface']
    latitude_list_saved = False  # Flag to indicate if latitude list is saved
    longitude_list_saved = False  

    for i in range(24 // interval):
        print(f'{input_data["time"].strftime("%Y-%m-%d")} {(i+1)*interval} hour')
        
        output_upper, output_surface = ort_session.run(None, {'input': input_upper, 'input_surface': input_surface})
        
        # Extract sub-region
        region_surface_data, region_upper_data, latitude_list, longitude_list = extract_region(output_surface, output_upper, lat_min, lat_max, lon_min, lon_max)

        # Save the sub-region results
        np.save(os.path.join(result_dir, f'output_upper_{input_data["time"].strftime("%Y%m%d")}_{(i+1)*interval}'), region_upper_data) ##np.save(os.path.join(result_dir, f'output_upper_{(i+1)*interval}'), region_upper_data)
        np.save(os.path.join(result_dir, f'output_surface_{input_data["time"].strftime("%Y%m%d")}_{(i+1)*interval}'), region_surface_data) ##np.save(os.path.join(result_dir, f'output_surface_{(i+1)*interval}'), region_surface_data)
        
        ## Save latitude and longitude lists only once
        if not latitude_list_saved:
            with open(os.path.join(result_dir, 'latitude_list.txt'), 'w') as f:
                f.write('\n'.join(map(str, latitude_list)))
            latitude_list_saved = True
        if not longitude_list_saved:
            with open(os.path.join(result_dir, 'longitude_list.txt'), 'w') as f:
                f.write('\n'.join(map(str, longitude_list)))
            longitude_list_saved = True

        input_upper = output_upper
        input_surface = output_surface

def predict_region_weather(input_base_dir, model_path, start_date, end_date, region_name, lat_min, lat_max, lon_min, lon_max, model_type='1hr'):
    # Check if model type is valid
    if model_type not in ['1hr', '6hr']:
        raise ValueError("Invalid model type. Please choose '1hr' or '6hr'.")
    
    # Set the behavier of onnxruntime
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena=False # CPU memory management, False= not optimizing memory allocation on the CPU.
    options.enable_mem_pattern = False  # controls memory pattern optimization, False=could be beneficial for performance but may consume more memory.
    options.enable_mem_reuse = False # controls memory reuse, False= potentially save memory but may affect performance.
    # Increase the number for faster inference and more memory consumption
    options.intra_op_num_threads = 2 # the number of threads for intra-op parallelism. It specifies how many threads should be used for parallel execution within each operator.
    # Set the behavier of cuda provider
    cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',} 

    # Define the input datetime range
    datetime_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Initialize ONNX runtime session based on model type
    ort_session = ort.InferenceSession(model_path,sess_options=options, providers=['CPUExecutionProvider']) #inference will be performed on the CPU.

    # Run inference for each day
    for time in datetime_range:
        print(f"Running inference for {time.strftime('%Y-%m-%d')}")
        input_dir = os.path.join(input_base_dir, f'{time}')
        result_dir = os.path.join(f'output_{region_name}', time.strftime('%Y-%m-%d'))
        os.makedirs(result_dir, exist_ok=True)
        input_data = load_input_data(input_dir, time)
        # Choose the appropriate inference function based on the model type
        interval = 6 if model_type == '6hr' else 1

        # Start measuring CPU time
        start_time = datetime.datetime.now()

        run_inference_region(ort_session, input_data, result_dir, region_name, lat_min, lat_max, lon_min, lon_max, interval)

        # End measuring CPU time
        end_time = datetime.datetime.now()
        cpu_time = end_time - start_time
        print("CPU Time:", int(cpu_time.total_seconds() // 60), "minutes", int(cpu_time.total_seconds() % 60), "seconds")

        # Memory usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss  # in bytes
        print("Memory Usage:", memory_usage / (1024 * 1024), "MB")  # convert to MB

def load_input_data(input_dir, time):
    input_data = {}
    input_data['input_upper'] = np.load(os.path.join(input_dir, 'input_upper.npy')).astype(np.float32)
    input_data['input_surface'] = np.load(os.path.join(input_dir, 'input_surface.npy')).astype(np.float32)
    input_data['time'] = time
    return input_data

