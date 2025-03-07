
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
scaler_y = StandardScaler()

X_transformer_standardized = scaler_X_transformer.fit_transform(X_transformer)
X_lstm_standardized = scaler_X_lstm.fit_transform(X_lstm)
y_standardized = scaler_y.fit_transform(y)


n_samples = X_transformer_standardized.shape[0]
train_size = int(n_samples * 0.8)
valid_size = int(n_samples * 0.1)



x_test_15= pd.read_csv('window_sensor_11_13.csv')
y_test_15= pd.read_csv('window_inverter_11_13.csv')
y_test_15['DC_Voltage_Lag'] = y_test_15['DC_Voltage'].shift(1, fill_value=444)
y_test_15['DC_Current_Lag'] = y_test_15['DC_Current'].shift(1, fill_value=1.08)


print(y_test_15['DC_Voltage_Lag'])
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


x_transformer_15_std = scaler_X_transformer.transform(x_transformer_15)
X_lstm_15_std = scaler_X_lstm.transform(x_lstm_15)



X_valid_transformer = X_transformer_standardized[train_size:train_size + valid_size]
X_test_transformer = x_transformer_15_std

X_test_lstm =X_lstm_15_std
X_valid_lstm = X_lstm_standardized[train_size:train_size + valid_size]
y_valid = y_standardized[train_size:train_size + valid_size]



X_valid_tensor_transformer = torch.tensor(X_valid_transformer, dtype=torch.float32).unsqueeze(1).to('cuda')
X_test_tensor_transformer = torch.tensor(X_test_transformer, dtype=torch.float32).unsqueeze(1).to('cuda')
X_valid_tensor_lstm=torch.tensor(X_valid_lstm, dtype=torch.float32).unsqueeze(1).to('cuda')
X_test_tensor_lstm = torch.tensor(X_test_lstm, dtype=torch.float32).unsqueeze(1).to('cuda')

y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to('cuda')



transformer_model = TransformerRegressor(input_dim=6, output_dim=2).to('cuda')
lstm_model = LSTMRegressor(input_dim=4,hidden_dim=256, output_dim=2).to('cuda')

transformer_model.load_state_dict(torch.load('Transformer_best_model_last_moment_1A.pth'))
lstm_model.load_state_dict(torch.load('LSTMRegressor_best_model_1A.pth'))


def predict(model, X_tensor, scaler_y):
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()
    return scaler_y.inverse_transform(predictions)



y_true= pd.read_csv('window_inverter_11_13.csv')

y_true_15= np.stack([y_true['DC_Voltage'].values, y_true['DC_Current'].values], axis=1)


# 使用 Transformer 模型预测
transformer_predictions = predict(transformer_model, X_test_tensor_transformer, scaler_y)

# 使用 LSTM 模型预测
lstm_predictions = predict(lstm_model, X_test_tensor_lstm, scaler_y)
y_transformer=transformer_predictions

y_lstm=lstm_predictions



time_interval_hours = 5 / 60
true_power = y_true_15[:, 0] * y_true_15[:, 1]
predicted_power_transformer = y_transformer[:, 0] * y_transformer[:, 1]

true_energy_kwh = np.sum(true_power) * time_interval_hours / 1000
predicted_energy_kwh_transformer = np.sum(predicted_power_transformer) * time_interval_hours / 1000

energy_error_kwh_transformer = predicted_energy_kwh_transformer - true_energy_kwh
percentage_error_transoformer = (energy_error_kwh_transformer / true_energy_kwh) * 100


print(f"True cumulative energy (kWh): {true_energy_kwh:.4f} kWh")
print(f"Predicted cumulative energy (kWh): {predicted_energy_kwh_transformer:.4f} kWh")
print(f"Energy error (kWh): {energy_error_kwh_transformer:.4f} kWh")
print(f"Percentage error: {percentage_error_transoformer:.2f}%")

true_power = y_true_15[:, 0] * y_true_15[:, 1]
predicted_power_lstm = y_lstm[:, 0] * y_lstm[:, 1]

true_energy_kwh = np.sum(true_power) * time_interval_hours / 1000
predicted_energy_kwh_lstm = np.sum(predicted_power_lstm) * time_interval_hours / 1000

energy_error_kwh_lstm = predicted_energy_kwh_lstm - true_energy_kwh
percentage_error_lstm = (energy_error_kwh_lstm/ true_energy_kwh) * 100
print(f"True cumulative energy (kWh): {true_energy_kwh:.4f} kWh")
print(f"Predicted cumulative energy (kWh): {energy_error_kwh_lstm:.4f} kWh")
print(f"Energy error (kWh): {energy_error_kwh_lstm:.4f} kWh")
print(f"Percentage error: {percentage_error_lstm:.2f}%")

from sklearn.metrics import mean_absolute_error

# 计算 MAE 的函数
def calculate_mae(y_true, y_pred):
    voltage_mae = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    current_mae = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    return voltage_mae, current_mae

# Transformer 模型的 MAE 计算
voltage_mae_transformer, current_mae_transformer = calculate_mae(y_true_15, y_transformer)

print(f"Transformer Model MAE:")
print(f"Voltage MAE: {voltage_mae_transformer:.4f}")
print(f"Current MAE: {current_mae_transformer:.4f}")

# LSTM 模型的 MAE 计算
voltage_mae_lstm, current_mae_lstm = calculate_mae(y_true_15, y_lstm)

print(f"\nLSTM Model MAE:")
print(f"Voltage MAE: {voltage_mae_lstm:.4f}")
print(f"Current MAE: {current_mae_lstm:.4f}")


import matplotlib.pyplot as plt

# 真实值和预测值
voltage_true = y_true_15[:, 0]
current_true = y_true_15[:, 1]

voltage_transformer = y_transformer[:, 0]
current_transformer = y_transformer[:, 1]

voltage_lstm = y_lstm[:, 0]
current_lstm = y_lstm[:, 1]

from window_test_recursive import recursive_predict
initial_time_index = 0
steps = len(X_test_tensor_transformer) - initial_time_index  # 从起始索引到测试集结束
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
recursive_prediction_transformer= recursive_predict(transformer_model, X_test_tensor_transformer, initial_time_index, steps, scaler_y, device)

voltage_re= recursive_prediction_transformer[:, 0]
current_re= recursive_prediction_transformer[:, 1]

# 绘制电压对比图
plt.figure(figsize=(8, 6))
plt.plot(voltage_true, label="True Voltage", linestyle='-', linewidth=2)
plt.plot(voltage_transformer, label="Transformer Predicted Voltage", linestyle='--',)
plt.plot(voltage_lstm, label="LSTM Predicted Voltage", linestyle=':',)
plt.title("Voltage Comparison", fontsize=16)
plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("Voltage (V)", fontsize=12)
plt.legend()

#plt.savefig('Normal_13_V.pdf')
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(current_true, label="True Current", linestyle='-', linewidth=2)
plt.plot(current_transformer, label="Transformer Predicted Current", linestyle='--',)
plt.plot(current_lstm, label="LSTM Predicted Current", linestyle=':', )
plt.title("Current Comparison", fontsize=16)
plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("Current (A)", fontsize=12)
plt.legend()
#plt.savefig('Normal_13_C.pdf')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(true_power, label="True Current", linestyle='-', linewidth=2)
plt.plot(predicted_power_transformer, label="Transformer Predicted Current", linestyle='--',)
plt.plot(predicted_power_lstm, label="LSTM Predicted Current", linestyle=':', )
plt.title("Power Comparison", fontsize=16)
plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("Current (A)", fontsize=12)
plt.legend()

plt.show()
