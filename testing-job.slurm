#!/bin/bash -l
#SBATCH --job-name= first_try
#SBATCH --account=
#SBATCH --cluster=wice
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=%x.out
#SBATCH --error=%x.error
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ada.canaydin@kuleuven.be


# Load necessary modules (if required)
# module load module_name

# Activate your conda environment (if you are using one)
# conda activate my_environment

pip install -r requirements.txt

# Define variables
input_base_dir='$VSC_DATA/pangu-weather-operational/input'
model_path='$VSC_DATA/pangu-weather-operational/input/models/pangu_weather_1.onnx'

start_date='2022-01-01T00:00:00'
end_date='2022-01-31T00:00:00'

lat_min=$(echo "54.0 - 0.125" | bc)  # Compute lat_min using bc command-line calculator
lat_max=$(echo "58.5 + 0.125" | bc)  # Compute lat_max using bc command-line calculator
lon_min=$(echo "3.0 - 0.125" | bc)   # Compute lon_min using bc command-line calculator
lon_max=$(echo "10.0 + 0.125" | bc)  # Compute lon_max using bc command-line calculator
region_name='North_Sea'

model_type='1hr'


# Run your Python script 
python run_forecast.py "$input_base_dir" "$model_path" "$start_date" "$end_date" "$lat_min" "$lat_max" "$lon_min" "$lon_max" "$region_name" "$model_type"