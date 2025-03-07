
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=32, nhead=4, num_encoder_layers=4, num_decoder_layers=4,
                 dim_feedforward=32, dropout=0.2):
        super(TransformerRegressor, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.output_projection = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.input_projection(src)
        src = src.permute(1, 0, 2)
        transformer_output = self.transformer(src, src)
        transformer_output = transformer_output.permute(1, 0, 2)
        output = self.output_projection(transformer_output[:, -1, :])
        return output


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.2):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        out = self.fc(self.dropout(lstm_out[:, -1, :]))
        return out


x_data_2023 = pd.read_csv('Sensor_2023_daytime_1.csv')
y_data_2023 = pd.read_csv('Inverter_2023_daytime_1.csv')
x_data_2022 = pd.read_csv('Sensor_2022_daytime_1.csv')
y_data_2022 = pd.read_csv('Inverter_2022_daytime_1.csv')

import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler


x_data_combined = pd.concat([x_data_2022, x_data_2023], axis=0, ignore_index=True)
y_data_combined = pd.concat([y_data_2022, y_data_2023], axis=0, ignore_index=True)


y_data_combined['DC_Voltage_Lag'] = y_data_combined['DC_Voltage'].shift(1, fill_value=0)
y_data_combined['DC_Current_Lag'] = y_data_combined['DC_Current'].shift(1, fill_value=0)


X_transformer = np.stack([
    x_data_combined['Module_Temperature_degF'].values,
    x_data_combined['Ambient_Temperature_degF'].values,
    x_data_combined['Solar_Irradiation_Wpm2'].values,
    x_data_combined['Wind_Speed_mps'].values,
    y_data_combined['DC_Voltage_Lag'].values,
    y_data_combined['DC_Current_Lag'].values], axis=1)

X_lstm = np.stack([
    x_data_combined['Module_Temperature_degF'].values,
    x_data_combined['Ambient_Temperature_degF'].values,
    x_data_combined['Solar_Irradiation_Wpm2'].values,
    x_data_combined['Wind_Speed_mps'].values], axis=1)


y = np.stack([y_data_combined['DC_Voltage'].values, y_data_combined['DC_Current'].values], axis=1)


scaler_X_transformer = StandardScaler()
scaler_X_lstm = StandardScaler()
scaler_y = StandardScaler()

X_transformer_standardized = scaler_X_transformer.fit_transform(X_transformer)
X_lstm_standardized = scaler_X_lstm.fit_transform(X_lstm)
y_standardized = scaler_y.fit_transform(y)


n_samples = X_transformer_standardized.shape[0]
train_size = int(n_samples * 0.8)
valid_size = int(n_samples * 0.1)

X_valid_transformer = X_transformer_standardized[train_size:train_size + valid_size]
X_test_transformer = X_transformer_standardized[train_size + valid_size:]

X_test_lstm = X_lstm_standardized[train_size + valid_size:]
X_valid_lstm = X_lstm_standardized[train_size:train_size + valid_size]
y_valid = y_standardized[train_size:train_size + valid_size]
y_test = y_standardized[train_size + valid_size:]


X_valid_tensor_transformer = torch.tensor(X_valid_transformer, dtype=torch.float32).unsqueeze(1).to('cuda')
X_test_tensor_transformer = torch.tensor(X_test_transformer, dtype=torch.float32).unsqueeze(1).to('cuda')
X_valid_tensor_lstm=torch.tensor(X_valid_lstm, dtype=torch.float32).unsqueeze(1).to('cuda')
X_test_tensor_lstm = torch.tensor(X_test_lstm, dtype=torch.float32).unsqueeze(1).to('cuda')

y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to('cuda')
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to('cuda')


transformer_model = TransformerRegressor(input_dim=6, output_dim=2).to('cuda')
lstm_model = LSTMRegressor(input_dim=4, hidden_dim=128, output_dim=2).to('cuda')

transformer_model.load_state_dict(torch.load('Transformer_best_model_last_moment_1A.pth'))
lstm_model.load_state_dict(torch.load('LSTMRegressor_best_model_1A.pth'))


def predict(model, X_tensor, scaler_y):
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()
    return scaler_y.inverse_transform(predictions)


def detect_anomalies(y_true, y_pred, threshold):
    error = np.abs(y_true - y_pred)
    return error > threshold


y_valid_original_last = scaler_y.inverse_transform(y_valid_tensor.cpu().numpy())
predicted_values_valid_last = predict(transformer_model, X_valid_tensor_transformer, scaler_y)




voltage_threshold_last = 1.5*np.percentile(np.abs(y_valid_original_last[:, 0] - predicted_values_valid_last[:, 0]), 95)
current_threshold_last = 1.5*np.percentile(np.abs(y_valid_original_last[:, 1] - predicted_values_valid_last[:, 1]), 95)
print(voltage_threshold_last)
print(current_threshold_last)


y_valid_original_sensor = scaler_y.inverse_transform(y_valid_tensor.cpu().numpy())
predicted_values_valid_sensor = predict(lstm_model, X_valid_tensor_lstm, scaler_y)




voltage_threshold_sensor = 1.5*np.percentile(np.abs(y_valid_original_sensor[:, 0] - predicted_values_valid_sensor[:, 0]), 95)
current_threshold_sensor =1.5* np.percentile(np.abs(y_valid_original_sensor[:, 1] - predicted_values_valid_sensor[:, 1]), 95)

print(voltage_threshold_sensor)
print(current_threshold_sensor)
y_test_original = scaler_y.inverse_transform(y_test_tensor.cpu().numpy())
y_corrected = y_test_original.copy()
use_lstm = False
consecutive_anomalies = 0
anomaly_threshold = 3
lstm_trigger_count = 0


for t in range(len(y_test_original)):
    if not use_lstm:

        X_current_transformer = X_test_tensor_transformer[t:t + 1, :, :]
        y_pred_transformer = predict(transformer_model, X_current_transformer, scaler_y)[0]


        anomaly_voltage = detect_anomalies(y_test_original[t, 0], y_pred_transformer[0], voltage_threshold_last)
        anomaly_current = detect_anomalies(y_test_original[t, 1], y_pred_transformer[1], current_threshold_last)

        if anomaly_voltage or anomaly_current:
            consecutive_anomalies += 1
            if consecutive_anomalies >= anomaly_threshold:

                lstm_trigger_count += 1
                use_lstm = True
                print(f"Attack detected at step {t}, switching to LSTM.")
        else:
            consecutive_anomalies = 0
            y_corrected[t] = y_test_original[t]
    else:

        X_current_lstm = X_test_tensor_lstm[t:t + 1, :, :]
        y_pred_lstm = predict(lstm_model, X_current_lstm, scaler_y)[0]
        y_corrected[t] = y_pred_lstm

        # 检测是否恢复正常
        recovered_voltage = not detect_anomalies(y_test_original[t, 0], y_pred_lstm[0], voltage_threshold_sensor)
        recovered_current = not detect_anomalies(y_test_original[t, 1], y_pred_lstm[1], current_threshold_sensor)

        if recovered_voltage and recovered_current:
            use_lstm = False
            consecutive_anomalies = 0
            print(f"Attack ended at step {t}, switching back to Transformer.")

time_interval_hours = 5 / 60
true_power = y_test_original[:, 0] * y_test_original[:, 1]
predicted_power = y_corrected[:, 0] * y_corrected[:, 1]

true_energy_kwh = np.sum(true_power) * time_interval_hours / 1000
predicted_energy_kwh = np.sum(predicted_power) * time_interval_hours / 1000

energy_error_kwh = predicted_energy_kwh - true_energy_kwh
percentage_error = (energy_error_kwh / true_energy_kwh) * 100


print(f"True cumulative energy (kWh): {true_energy_kwh:.4f} kWh")
print(f"Predicted cumulative energy (kWh): {predicted_energy_kwh:.4f} kWh")
print(f"Energy error (kWh): {energy_error_kwh:.4f} kWh")
print(f"Percentage error: {percentage_error:.2f}%")
print(f"Total LSTM trigger events: {lstm_trigger_count} times")
