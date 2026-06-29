import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def validate_predictions(csv_path="frame_metrics.csv", prediction_horizon_sec=5.0):
    print(f"[INFO] Loading metrics from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {csv_path}. Please run the main script first.")
        return

    # Ensure data is sorted by time
    df = df.sort_values("timestamp_sec").reset_index(drop=True)

    # 1. Prepare the "Future" target data
    # We now need the timestamp, actual live vehicles, AND actual risk score for the target
    df_future = df[["timestamp_sec", "live_vehicles", "risk_score"]].copy()
    df_future.rename(columns={
        "timestamp_sec": "future_timestamp",
        "live_vehicles": "actual_future_vehicles",
        "risk_score":    "actual_future_risk"
    }, inplace=True)

    # 2. Calculate the exact time each row is trying to predict
    df["target_timestamp"] = df["timestamp_sec"] + prediction_horizon_sec

    # 3. Merge the datasets
    # Match the 'target_timestamp' of our prediction to the closest 'future_timestamp'
    validation_df = pd.merge_asof(
        df,
        df_future,
        left_on="target_timestamp",
        right_on="future_timestamp",
        direction="nearest",
        tolerance=0.5  # Max half-second gap allowed for a match
    )

    # Drop the final seconds of the video where no future data exists to validate against
    validation_df = validation_df.dropna(subset=["actual_future_vehicles", "actual_future_risk"])

    if validation_df.empty:
        print("[ERROR] Not enough data to validate. Video must be longer than the prediction horizon.")
        return

    # 4. Calculate Error Metrics
    # Vehicle Errors
    validation_df['error_vehicles'] = validation_df['pred_vehicles_5s'] - validation_df['actual_future_vehicles']
    mae_veh = np.mean(np.abs(validation_df['error_vehicles']))
    rmse_veh = np.sqrt(np.mean(validation_df['error_vehicles']**2))

    # Risk Score Errors
    validation_df['error_risk'] = validation_df['pred_risk_score_5s'] - validation_df['actual_future_risk']
    mae_risk = np.mean(np.abs(validation_df['error_risk']))
    rmse_risk = np.sqrt(np.mean(validation_df['error_risk']**2))

    # Print Results
    print("-" * 50)
    print("PREDICTION ACCURACY METRICS")
    print("-" * 50)
    print(f"VEHICLE DENSITY ({prediction_horizon_sec}s Horizon):")
    print(f"  Mean Absolute Error (MAE) : {mae_veh:.2f} vehicles")
    print(f"  Root Mean Sq Error (RMSE) : {rmse_veh:.2f} vehicles")
    print("-" * 50)
    print(f"RISK SCORE ({prediction_horizon_sec}s Horizon):")
    print(f"  Mean Absolute Error (MAE) : {mae_risk:.2f}")
    print(f"  Root Mean Sq Error (RMSE) : {rmse_risk:.2f}")
    print("-" * 50)

    # 5. Generate the Comparison Plots
    
    # --- PLOT 1: Vehicle Density Validation ---
    plt.figure(figsize=(12, 6))
    fig1 = plt.gcf()
    fig1.canvas.manager.set_window_title('Density Prediction Validation')

    plt.plot(validation_df['target_timestamp'], validation_df['actual_future_vehicles'], 
             label='Actual Vehicles', color='#00FF00', linewidth=2, alpha=0.8)
    
    plt.plot(validation_df['target_timestamp'], validation_df['pred_vehicles_5s'], 
             label=f'Predicted Vehicles (Forecasted {prediction_horizon_sec}s ago)', 
             color='#FF00FF', linestyle='--', linewidth=2, alpha=0.9)

    plt.title(f'{prediction_horizon_sec}-Second Forward Traffic Density Prediction vs Actual', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Video Time (Seconds)', fontsize=12)
    plt.ylabel('Number of Vehicles', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # --- PLOT 2: Risk Score Validation ---
    plt.figure(figsize=(12, 6))
    fig2 = plt.gcf()
    fig2.canvas.manager.set_window_title('Risk Score Prediction Validation')

    plt.plot(validation_df['target_timestamp'], validation_df['actual_future_risk'], 
             label='Actual Risk Score', color='darkorange', linewidth=2, alpha=0.8)
    
    plt.plot(validation_df['target_timestamp'], validation_df['pred_risk_score_5s'], 
             label=f'Predicted Risk (Forecasted {prediction_horizon_sec}s ago)', 
             color='blue', linestyle='--', linewidth=2, alpha=0.9)

    plt.title(f'{prediction_horizon_sec}-Second Forward Risk Score Prediction vs Actual', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Video Time (Seconds)', fontsize=12)
    plt.ylabel('Risk Score', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Display both windows simultaneously
    plt.show()

if __name__ == "__main__":
    validate_predictions()