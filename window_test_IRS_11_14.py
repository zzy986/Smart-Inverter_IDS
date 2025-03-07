
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=32, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
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
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)  # 对LSTM输出进行Layer Normalization
        out = self.fc(self.dropout(lstm_out[:, -1, :]))  # 只取最后一个时间步的输出并应用Dropout
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



n_samples = X_transformer.shape[0]
train_size = int(n_samples * 0.8)


X_train_transformer = X_transformer[:train_size]
y_train = y[:train_size]

X_train=scaler_X_transformer.fit_transform(X_train_transformer)
scaler_y = StandardScaler()
y_train=scaler_y.fit_transform(y_train)

X_train_lstm = X_lstm[:train_size]
X_transformer_standardized = scaler_X_transformer.fit_transform(X_train_transformer)
X_lstm_standardized = scaler_X_lstm.fit_transform(X_train_lstm)

y_standardized = scaler_y.transform(y)


n_samples = X_transformer_standardized.shape[0]
train_size = int(n_samples * 0.8)
valid_size = int(n_samples * 0.1)



x_test_15= pd.read_csv('window_sensor_11_14.csv')
y_test_15= pd.read_csv('attack_14.csv')
y_test_15['DC_Voltage_Lag'] = y_test_15['DC_Voltage'].shift(1, fill_value=0)
y_test_15['DC_Current_Lag'] = y_test_15['DC_Current'].shift(1, fill_value=0)
x_transformer_15= np.stack([
    x_test_15['Module_Temperature_degF'].values,
    x_test_15['Ambient_Temperature_degF'].values,
    x_test_15['Solar_Irradiation_Wpm2'].values,
    x_test_15['Wind_Speed_mps'].values,
    y_test_15['DC_Voltage_Lag'].values,
    y_test_15['DC_Current_Lag'].values], axis=1)

x_lstm_15 = np.stack([
    x_test_15['Module_Temperature_degF'].values,
    x_test_15['Ambient_Temperature_degF'].values,
    x_test_15['Solar_Irradiation_Wpm2'].values,
    x_test_15['Wind_Speed_mps'].values], axis=1)

y_test_15_attack = np.stack([y_test_15['DC_Voltage'].values, y_test_15['DC_Current'].values], axis=1)
x_transformer_15_std = scaler_X_transformer.transform(x_transformer_15)
X_lstm_15_std = scaler_X_lstm.transform(x_lstm_15)
y_15_std = scaler_y.transform(y_test_15_attack)


X_valid_transformer = X_transformer_standardized[train_size:train_size + valid_size]
X_test_transformer = x_transformer_15_std

X_test_lstm =X_lstm_15_std
X_valid_lstm = X_lstm_standardized[train_size:train_size + valid_size]
y_valid = y_standardized[train_size:train_size + valid_size]

y_test=y_15_std

X_valid_tensor_transformer = torch.tensor(X_valid_transformer, dtype=torch.float32).unsqueeze(1).to('cuda')
X_test_tensor_transformer = torch.tensor(X_test_transformer, dtype=torch.float32).unsqueeze(1).to('cuda')
X_valid_tensor_lstm=torch.tensor(X_valid_lstm, dtype=torch.float32).unsqueeze(1).to('cuda')
X_test_tensor_lstm = torch.tensor(X_test_lstm, dtype=torch.float32).unsqueeze(1).to('cuda')

y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to('cuda')
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to('cuda')


transformer_model = TransformerRegressor(input_dim=6, output_dim=2).to('cuda')
lstm_model = LSTMRegressor(input_dim=4, hidden_dim=256, output_dim=2).to('cuda')

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
anomaly_threshold = 1
lstm_trigger_count = 0


# 遍历每个时间步
for t in range(len(y_test_original)):
    if not use_lstm:
        # 使用Transformer模型进行预测
        X_current_transformer = X_test_tensor_transformer[t:t + 1, :, :]
        y_pred_transformer = predict(transformer_model, X_current_transformer, scaler_y)[0]

        # 检测电压和电流是否异常
        anomaly_voltage = detect_anomalies(y_test_original[t, 0], y_pred_transformer[0], voltage_threshold_last)
        anomaly_current = detect_anomalies(y_test_original[t, 1], y_pred_transformer[1], current_threshold_last)

        if anomaly_voltage or anomaly_current:
            consecutive_anomalies += 1
            if consecutive_anomalies >= anomaly_threshold:
                # 如果连续异常达到阈值，则切换到LSTM预测
                lstm_trigger_count += 1
                use_lstm = True
                print(f"Attack detected at step {t}, switching to LSTM.")
        else:
            consecutive_anomalies = 0  # 未检测到异常时重置连续异常计数
    else:
        # 使用LSTM模型进行预测并将结果用于修正
        X_current_lstm = X_test_tensor_lstm[t:t + 1, :, :]
        y_pred_lstm = predict(lstm_model, X_current_lstm, scaler_y)[0]
        y_corrected[t] = y_pred_lstm  # 将预测值赋给y_corrected

        # 检测是否恢复正常
        recovered_voltage = not detect_anomalies(y_test_original[t, 0], y_pred_lstm[0], voltage_threshold_sensor)
        recovered_current = not detect_anomalies(y_test_original[t, 1], y_pred_lstm[1], current_threshold_sensor)

        if recovered_voltage and recovered_current:
            # 检测恢复正常后，切换回Transformer
            use_lstm = False
            consecutive_anomalies = 0
            print(f"Attack ended at step {t}, switching back to Transformer.")



y_true= pd.read_csv('window_inverter_11_14.csv')

y_true_15= np.stack([y_true['DC_Voltage'].values, y_true['DC_Current'].values], axis=1)



time_interval_hours = 5 / 60
true_power = y_true_15[:, 0] * y_true_15[:, 1]
predicted_power = y_corrected[:, 0] * y_corrected[:, 1]

true_energy_kwh = np.sum(true_power) * time_interval_hours / 1000
predicted_energy_kwh = np.sum(predicted_power) * time_interval_hours / 1000

energy_error_kwh = predicted_energy_kwh - true_energy_kwh
percentage_error = (energy_error_kwh / true_energy_kwh) * 100


y_true= pd.read_csv('window_inverter_11_13.csv')
y_true_15= np.stack([y_true['DC_Voltage'].values, y_true['DC_Current'].values], axis=1)
voltage_true = y_true_15[:, 0]
current_true = y_true_15[:, 1]
print(f"True cumulative energy (kWh): {true_energy_kwh:.4f} kWh")
print(f"Predicted cumulative energy (kWh): {predicted_energy_kwh:.4f} kWh")
print(f"Energy error (kWh): {energy_error_kwh:.4f} kWh")
print(f"Percentage error: {percentage_error:.2f}%")
print(f"Total LSTM trigger events: {lstm_trigger_count} times")
plt.figure(figsize=(12, 6))
plt.plot(current_true, label='True Current (A)', color='green', linestyle='--')
plt.plot(y_corrected[:, 1], label='Predicted Current (A)', color='orange')
plt.xlabel('Time Step (5min)')
highlight_regions = [(0, 36), (97, 157),(277,336),(397,456)]

# 循环为每个区间添加浅红色背景
for start, end in highlight_regions:
    plt.axvspan(start, end, color='red', alpha=0.1, label='Attack Period' if start == 0 else "")
plt.ylabel('Current (A)')
plt.title('Test Window Current')
plt.legend()

plt.savefig('Test_Current.pdf',dpi=300)

plt.show()

plt.figure(figsize=(12, 6))
plt.plot(voltage_true, label='True Voltage (V)', color='green', linestyle='--')
plt.plot(y_corrected[:, 0], label='Predicted Voltage (V)', color='orange')
plt.xlabel('Time Step (5min)')
highlight_regions = [(0, 36), (97, 157),(277,336),(397,456)]

# 循环为每个区间添加浅红色背景
for start, end in highlight_regions:
    plt.axvspan(start, end, color='red', alpha=0.1, label='Attack Period' if start == 0 else "")
plt.ylabel('Voltage (V)')
plt.title('Test Window Voltage')
plt.legend()

plt.savefig('Test_Voltage.pdf',dpi=300)
plt.show()