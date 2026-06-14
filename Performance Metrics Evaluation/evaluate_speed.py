import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error


ground_truth_speeds = np.array([85, 63, 35, 51, 41, 72, 101, 92, 57, 78, 48, 68]) 

folder_name = 'vehicle_speeds'

file_names = [os.path.join(folder_name, f'vehicle_speeds_{i}.csv') for i in range(1, 13)]

median_estimated_speeds = []


for file_path in file_names:
    try:
        df = pd.read_csv(file_path)
        
        # Filter 1: Only consider track_id == '1' (handle both string and int types)
        df_filtered = df[(df['track_id'] == 1) | (df['track_id'] == '1')]
        
        # Filter 2: Ignore entries where speed_kmh is exactly 0
        df_filtered = df_filtered[df_filtered['speed_kmh'] != 0]
        
        # Calculate the median speed for this video
        if not df_filtered.empty:
            median_speed = df_filtered['speed_kmh'].median()
        else:
            print(f"Warning: No valid speed data found in {file_path} after filtering.")
            median_speed = np.nan # Use NaN if filtering removed all rows
            
        median_estimated_speeds.append(median_speed)
        
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure the folder and file exist.")
        median_estimated_speeds.append(np.nan)

# Convert to numpy array for metric calculations
median_estimated_speeds = np.array(median_estimated_speeds)


# Create a mask to ignore any files that failed to load or had no valid data (NaNs)
valid_mask = ~np.isnan(median_estimated_speeds)
y_true = ground_truth_speeds[valid_mask]
y_pred = median_estimated_speeds[valid_mask]

if len(y_true) > 0:
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100 # Convert to percentage
    rmse = root_mean_squared_error(y_true, y_pred) # Using the updated scikit-learn function
else:
    mae = mape = rmse = 0
    print("No valid data available to calculate metrics.")


# Create a DataFrame to display the table nicely. 
# Extracting just the filename from the path for a cleaner table display
display_names = [os.path.basename(path) for path in file_names]

results_df = pd.DataFrame({
    'Video File': display_names,
    'Ground Truth Speed (km/h)': ground_truth_speeds,
    'Estimated Median Speed (km/h)': median_estimated_speeds
})

# Calculate the absolute difference for the table
results_df['Difference (Error)'] = abs(results_df['Ground Truth Speed (km/h)'] - results_df['Estimated Median Speed (km/h)'])

print("\n" + "="*50)
print("SPEED ESTIMATION RESULTS TABLE")
print("="*50)
print(results_df.to_string(index=False))
print("="*50)

print("\n" + "="*50)
print("PERFORMANCE METRICS")
print("="*50)
print(f"Mean Absolute Error (MAE):           {mae:.2f} km/h")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f} %")
print(f"Root Mean Squared Error (RMSE):        {rmse:.2f} km/h")
print("="*50 + "\n")


plt.figure(figsize=(10, 6))

# X-axis labels (Video 1, Video 2, etc.)
x_labels = [f"Video {i}" for i in range(1, 13)]
x_indices = np.arange(len(x_labels))

# Plot Ground Truth
plt.plot(x_indices, ground_truth_speeds, marker='o', linestyle='-', linewidth=2, color='blue', label='Ground Truth Speed')

# Plot Estimated Median Speed
plt.plot(x_indices, median_estimated_speeds, marker='s', linestyle='--', linewidth=2, color='red', label='Estimated Median Speed')

# Formatting the plot
plt.title('Vehicle Speed Estimation: Ground Truth vs YOLOv8 Estimations', fontsize=14, fontweight='bold')
plt.xlabel('VS13 Dataset Videos', fontsize=12)
plt.ylabel('Speed (km/h)', fontsize=12)
plt.xticks(x_indices, x_labels, rotation=45)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()

# Display the plot
plt.show()

