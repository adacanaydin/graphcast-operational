import os
import time
import netCDF4 as nc
import numpy as np
import onnxruntime as ort

from utulities.work_time import calculate_work_times
from utulities.npy_to_nc import npy2nc_surface, npy2nc_upper
from utulities.naming_files import name_output_upper_ncfile, name_output_surface_ncfile




def initialize_sessions(ort_options, forecast_hour):
    # Record the time to initialize
    start_Time = time.time()

    work_times = calculate_work_times(forecast_hour)
    print(f"All the models used for the {forecast_hour}hr prediction is: 24hr: {work_times[0]} || 6hr: {work_times[1]} || 3hr: {work_times[2]} || 1hr: {work_times[3]}")

    ort_session_24 = ort.InferenceSession('pangu_weather_24.onnx', sess_options=ort_options, providers=['CPUExecutionProvider']) if work_times[0] else None
    ort_session_6 = ort.InferenceSession('pangu_weather_6.onnx', sess_options=ort_options, providers=['CPUExecutionProvider']) if work_times[1] else None
    ort_session_3 = ort.InferenceSession('pangu_weather_3.onnx', sess_options=ort_options, providers=['CPUExecutionProvider']) if work_times[2] else None
    ort_session_1 = ort.InferenceSession('pangu_weather_1.onnx', sess_options=ort_options, providers=['CPUExecutionProvider']) if work_times[3] else None
    
    run_Time = int(time.time() - start_Time)
    print(f"Sessions have initalized. Time cost is {run_Time//60}min {run_Time%60}s.")

    return [ort_session_24, ort_session_6, ort_session_3, ort_session_1]