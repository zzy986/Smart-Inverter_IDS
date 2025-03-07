import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

import time
# Regressor models
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
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        out = self.fc(self.dropout(lstm_out[:, -1, :]))
        return out


class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(GRURegressor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


class RNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(RNNRegressor, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


# SMAPE metric
def smape(y_true, y_pred):
    return 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


# Calculate metrics
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    smape_value = smape(y_true, y_pred)
    return mae, mse, rmse, r2, smape_value


# Visualization function
def visualize_results(y_test_original, predictions_original, model_name):
    plt.figure(figsize=(8, 6))
    plt.plot(y_test_original[0:1000, 0], label='True Voltage (V)', color='blue')
    plt.plot(predictions_original[0:1000, 0], label=f'{model_name} Predicted Voltage (V)', color='orange')
    plt.xlabel('Time Step')
    plt.ylabel('Voltage (V)')
    plt.title(f'{model_name} - Voltage (V) Test Predictions')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(y_test_original[0:1000, 1], label='True Current (A)', color='blue')
    plt.plot(predictions_original[0:1000, 1], label=f'{model_name} Predicted Current (A)', color='orange')
    plt.xlabel('Time Step')
    plt.ylabel('Current (A)')
    plt.title(f'{model_name} - Current (A) Test Predictions')
    plt.legend()
    plt.show()


# Recursive prediction function
def recursive_predict(model, X_test_tensor, initial_time_index, steps, scaler_y, device):
    """
    递归预测函数：使用模型的预测值替换DC_Voltage_Lag和DC_Current_Lag特征，并在每个时间步使用测试集的传感器数据。

    参数:
    - model: 训练好的模型。
    - X_test_tensor: 测试集的特征张量。
    - initial_time_index: 递归预测的起始时间索引。
    - steps: 递归预测的时间步数。
    - scaler_y: 用于逆标准化预测值的scaler。
    - device: 模型运行的设备 (CPU 或 GPU)。

    返回:
    - 递归预测的电压和电流值（已逆标准化）。
    """
    model.eval()  # 设置模型为评估模式
    predictions = []

    # 初始滞后特征：从测试集的初始时间索引获取
    y_previous = None  # 初始时还没有预测值

    for t in range(initial_time_index, initial_time_index + steps):
        # 检查索引是否超出范围
        if t >= len(X_test_tensor):
            break

        # 从测试集获取当前时间步的传感器特征
        X_current = X_test_tensor[t:t+1, :, :].clone().to(device)  # shape: [1, 1, feature_dim]

        # 如果有上一时间步的预测值，更新滞后特征
        if y_previous is not None:
            X_current[:, :, -2] = y_previous[:, 0].unsqueeze(1)  # 更新 DC_Voltage_Lag

            X_current[:, :, -1] = y_previous[:, 1].unsqueeze(1)  # 更新 DC_Current_Lag

        # 使用模型进行预测
        with torch.no_grad():
            y_pred = model(X_current)  # 预测电压和电流

        # 记录预测值
        predictions.append(y_pred.cpu().numpy())

        # 更新上一时间步的预测值
        y_previous = y_pred

    # 将所有的预测结果合并并进行逆标准化
    predictions = np.concatenate(predictions, axis=0)
    predictions_original = scaler_y.inverse_transform(predictions)

    return predictions_original

# Data loading and preprocessing
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
x_data_2023 = pd.read_csv('Sensor_2023_daytime_1.csv')
y_data_2023 = pd.read_csv('Inverter_2023_daytime_1.csv')
x_data_2022 = pd.read_csv('Sensor_2022_daytime_1.csv')
y_data_2022 = pd.read_csv('Inverter_2022_daytime_1.csv')

x_data_combined = pd.concat([x_data_2022, x_data_2023], axis=0, ignore_index=True)
y_data_combined = pd.concat([y_data_2022, y_data_2023], axis=0, ignore_index=True)

y_data_combined['DC_Voltage_Lag'] = y_data_combined['DC_Voltage'].shift(1, fill_value=0)
y_data_combined['DC_Current_Lag'] = y_data_combined['DC_Current'].shift(1, fill_value=0)


# Prepare feature matrix and target matrix
X = np.stack([x_data_combined['Module_Temperature_degF'].values,
              x_data_combined['Ambient_Temperature_degF'].values,
              x_data_combined['Solar_Irradiation_Wpm2'].values,
              x_data_combined['Wind_Speed_mps'].values,
              y_data_combined['DC_Voltage_Lag'].values,
              y_data_combined['DC_Current_Lag'].values], axis=1)

y = np.stack([y_data_combined['DC_Voltage'].values, y_data_combined['DC_Current'].values], axis=1)


scaler_X = StandardScaler()
scaler_y = StandardScaler()



n_samples = X.shape[0]
train_size = int(n_samples * 0.8)

X_train = X[:train_size]
y_train = y[:train_size]



X_temp = X[train_size:]
y_temp = y[train_size:]

# 手动划分验证集和测试集
n_temp_samples = X_temp.shape[0]
valid_size = int(n_temp_samples * 0.5)

# 前50%的临时数据作为验证集
X_valid = X_temp[:valid_size]
y_valid = y_temp[:valid_size]

# 后50%的临时数据作为测试集
X_test = X_temp[valid_size:]
y_test = y_temp[valid_size:]


X_train = scaler_X.fit_transform(X_train)
X_valid = scaler_X.transform(X_valid)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_valid = scaler_y.transform(y_valid)
y_test = scaler_y.transform(y_test)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).unsqueeze(1).to(device)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training set size: {X_train_tensor.size(0)} samples")
print(f"Validation set size: {X_valid_tensor.size(0)} samples")
print(f"Test set size: {X_test_tensor.size(0)} samples")
print(f"Training set size: {X_train_tensor.size(0)} samples")
print(f"Validation set size: {X_valid_tensor.size(0)} samples")
print(f"Test set size: {X_test_tensor.size(0)} samples")
model = TransformerRegressor(input_dim=X_test_tensor.shape[2], output_dim=y_test_tensor.shape[1]).to(device)
model.load_state_dict(torch.load('Transformer_best_model_last_moment_1A.pth'))

# 设置预测步数，例如预测测试集的全部时间步
initial_time_index = 0
steps = len(X_test_tensor) - initial_time_index  # 从起始索引到测试集结束

start_time = time.time()
predicted_values = recursive_predict(model, X_test_tensor, initial_time_index, steps, scaler_y, device)
end_time = time.time()  # 结束计时

# 计算并打印运行时间
execution_time = end_time - start_time
print(f"Function execution time: {execution_time:.4f} seconds")
y_test_original = scaler_y.inverse_transform(y_test_tensor.cpu().numpy())

# 可视化递归预测的结果
plt.figure(figsize=(8, 6))
plt.plot(y_test_original[:, 0], label='True Voltage (V)', color='blue')
plt.plot(predicted_values[:, 0], label='Predicted Voltage (V)', color='orange')
plt.xlabel('Time Step')
plt.ylabel('Voltage (V)')
plt.title('Recursive Voltage (V) Predictions')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(y_test_original[:, 1], label='True Current (A)', color='blue')
plt.plot(predicted_values[:, 1], label='Predicted Current (A)', color='orange')
plt.xlabel('Time Step')
plt.ylabel('Current (A)')
plt.title('Recursive Current (A) Predictions')
plt.legend()
plt.show()
# 计算评估指标
mae_voltage, mse_voltage, rmse_voltage, r2_voltage, smape_voltage = calculate_metrics(y_test_original[:, 0], predicted_values[:, 0])
mae_current, mse_current, rmse_current, r2_current, smape_current = calculate_metrics(y_test_original[:, 1], predicted_values[:, 1])

print(f"Voltage(V) - MAE: {mae_voltage:.4f}, MSE: {mse_voltage:.4f}, RMSE: {rmse_voltage:.4f}, R²: {r2_voltage:.4f}, SMAPE: {smape_voltage:.2f}%")
print(f"Current(A) - MAE: {mae_current:.4f}, MSE: {mse_current:.4f}, RMSE: {rmse_current:.4f}, R²: {r2_current:.4f}, SMAPE: {smape_current:.2f}%")

time_interval_hours = 5 / 60  # 如果是5分钟间隔

# 计算真实功率和预测功率
true_power = y_test_original[:, 0] * y_test_original[:, 1]  # 真实功率 = 真实电压 * 真实电流
predicted_power = predicted_values[:, 0] * predicted_values[:, 1]  # 预测功率 = 预测电压 * 预测电流

# 计算累计电能 (kWh)
true_energy_kwh = np.sum(true_power) * time_interval_hours / 1000  # 转换为kWh
predicted_energy_kwh = np.sum(predicted_power) * time_interval_hours / 1000  # 转换为kWh

# 计算电能误差
energy_error_kwh = predicted_energy_kwh - true_energy_kwh
percentage_error = (energy_error_kwh / true_energy_kwh) * 100

# 打印电能误差结果
print(f"True cumulative energy (kWh): {true_energy_kwh:.4f} kWh")
print(f"Predicted cumulative energy (kWh): {predicted_energy_kwh:.4f} kWh")
print(f"Energy error (kWh): {energy_error_kwh:.4f} kWh")
print(f"Percentage error: {percentage_error:.2f}%")

