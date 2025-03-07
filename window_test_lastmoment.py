import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Regressor models
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
# 加载数据
x_data_2023 = pd.read_csv('Sensor_2023_daytime_1.csv')
y_data_2023 = pd.read_csv('Inverter_2023_daytime_1.csv')
x_data_2022 = pd.read_csv('Sensor_2022_daytime_1.csv')
y_data_2022 = pd.read_csv('Inverter_2022_daytime_1.csv')

# 合并数据
x_data_combined = pd.concat([x_data_2022, x_data_2023], axis=0, ignore_index=True)
y_data_combined = pd.concat([y_data_2022, y_data_2023], axis=0, ignore_index=True)

# 创建滞后特征，Transformer使用
y_data_combined['DC_Voltage_Lag'] = y_data_combined['DC_Voltage'].shift(1, fill_value=0)
y_data_combined['DC_Current_Lag'] = y_data_combined['DC_Current'].shift(1, fill_value=0)

# Transformer模型使用的输入特征（传感器 + 滞后电压和电流）
X_transformer = np.stack([x_data_combined['Module_Temperature_degF'].values,
                          x_data_combined['Ambient_Temperature_degF'].values,
                          x_data_combined['Solar_Irradiation_Wpm2'].values,
                          x_data_combined['Wind_Speed_mps'].values,
                          y_data_combined['DC_Voltage_Lag'].values,
                          y_data_combined['DC_Current_Lag'].values], axis=1)

# LSTM模型使用的输入特征（仅传感器）
X_lstm = np.stack([x_data_combined['Module_Temperature_degF'].values,
                   x_data_combined['Ambient_Temperature_degF'].values,
                   x_data_combined['Solar_Irradiation_Wpm2'].values,
                   x_data_combined['Wind_Speed_mps'].values], axis=1)

y = np.stack([y_data_combined['DC_Voltage'].values, y_data_combined['DC_Current'].values], axis=1)

# 标准化
scaler_X_transformer = StandardScaler()
scaler_X_lstm = StandardScaler()
scaler_y = StandardScaler()

X_transformer_standardized = scaler_X_transformer.fit_transform(X_transformer)
X_lstm_standardized = scaler_X_lstm.fit_transform(X_lstm)
y_standardized = scaler_y.fit_transform(y)

# 验证集切分
n_samples = X_transformer_standardized.shape[0]

# 计算各部分数据集的大小
train_size = int(n_samples * 0.8)  # 80% 训练集
valid_size = int(n_samples * 0.1)  # 10% 验证集
test_size = n_samples - train_size - valid_size  # 剩下的10%用于测试集

# 数据集划分
X_train_transformer = X_transformer_standardized[:train_size]
X_valid_transformer = X_transformer_standardized[train_size:train_size + valid_size]
X_test_transformer = X_transformer_standardized[train_size + valid_size:]

X_train_lstm = X_lstm_standardized[:train_size]
X_valid_lstm = X_lstm_standardized[train_size:train_size + valid_size]
X_test_lstm = X_lstm_standardized[train_size + valid_size:]

y_train = y_standardized[:train_size]
y_valid = y_standardized[train_size:train_size + valid_size]
y_test = y_standardized[train_size + valid_size:]

# 转换为Tensor格式
X_train_tensor_transformer = torch.tensor(X_train_transformer, dtype=torch.float32).unsqueeze(1).to('cuda')
X_valid_tensor_transformer = torch.tensor(X_valid_transformer, dtype=torch.float32).unsqueeze(1).to('cuda')
X_test_tensor_transformer = torch.tensor(X_test_transformer, dtype=torch.float32).unsqueeze(1).to('cuda')

X_train_tensor_lstm = torch.tensor(X_train_lstm, dtype=torch.float32).unsqueeze(1).to('cuda')
X_valid_tensor_lstm = torch.tensor(X_valid_lstm, dtype=torch.float32).unsqueeze(1).to('cuda')
X_test_tensor_lstm = torch.tensor(X_test_lstm, dtype=torch.float32).unsqueeze(1).to('cuda')

y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to('cuda')
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to('cuda')
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to('cuda')

# 加载预训练模型
transformer_model = TransformerRegressor(input_dim=X_valid_tensor_transformer.shape[2], output_dim=2).to('cuda')
lstm_model = LSTMRegressor(input_dim=X_valid_tensor_lstm.shape[2], hidden_dim=128, output_dim=2).to('cuda')

# 假设Transformer和LSTM模型已经训练并保存了权重
transformer_model.load_state_dict(torch.load('Transformer_best_model_last_moment_1A.pth'))
lstm_model.load_state_dict(torch.load('LSTMRegressor_best_model_1A.pth'))

# 定义递归预测函数 (Transformer模型)
def recursive_predict(model, X_test_tensor, steps, scaler_y, device):
    model.eval()
    predictions = []
    y_previous = None

    for t in range(steps):
        X_current = X_test_tensor[t:t+1, :, :].clone().to(device)

        if y_previous is not None:
            X_current[:, :, -2] = y_previous[:, 0].unsqueeze(1)  # 更新滞后电压
            X_current[:, :, -1] = y_previous[:, 1].unsqueeze(1)  # 更新滞后电流

        with torch.no_grad():
            y_pred = model(X_current)

        predictions.append(y_pred.cpu().numpy())
        y_previous = y_pred

    predictions = np.concatenate(predictions, axis=0)
    return scaler_y.inverse_transform(predictions)

# 定义异常检测函数
def detect_anomalies(y_true, y_pred, threshold):
    error = np.abs(y_true - y_pred)
    anomalies = error > threshold
    return anomalies

# 定义替换异常的函数
def replace_anomalies_with_predictions(y_true, y_pred_transformer, y_pred_lstm, anomalies):
    y_corrected = y_true.copy()

    # 只在检测到异常的地方进行替换
    if np.any(anomalies):  # 检测到异常时
        y_combined = (y_pred_transformer + y_pred_lstm) / 2
        y_corrected[anomalies] = y_combined[anomalies]

    return y_corrected

# 使用验证集数据
y_valid_original = scaler_y.inverse_transform(y_valid_tensor.cpu().numpy())

# 设置阈值
voltage_threshold = np.percentile(np.abs(y_valid_original[:, 0] - y_valid_tensor.cpu().numpy()[:, 0]), 95)
current_threshold = np.percentile(np.abs(y_valid_original[:, 1] - y_valid_tensor.cpu().numpy()[:, 1]), 95)

# -------- 这里是加载新测试集的部分 --------

# 加载新的测试集数据 new_test_sensor.csv 和 new_test_inverter.csv attack
x_test_new = pd.read_csv('window_Sensor.csv')
y_test_new = pd.read_csv('window_inverter.csv')

# 创建滞后特征，Transformer使用
y_test_new['DC_Voltage_Lag'] = y_test_new['DC_Voltage'].shift(1, fill_value=0)
y_test_new['DC_Current_Lag'] = y_test_new['DC_Current'].shift(1, fill_value=0)

# Transformer模型使用的输入特征（传感器 + 滞后电压和电流）
X_test_transformer_new = np.stack([x_test_new['Module_Temperature_degF'].values,
                                   x_test_new['Ambient_Temperature_degF'].values,
                                   x_test_new['Solar_Irradiation_Wpm2'].values,
                                   x_test_new['Wind_Speed_mps'].values,
                                   y_test_new['DC_Voltage_Lag'].values,
                                   y_test_new['DC_Current_Lag'].values], axis=1)

# LSTM模型使用的输入特征（仅传感器）
X_test_lstm_new = np.stack([x_test_new['Module_Temperature_degF'].values,
                            x_test_new['Ambient_Temperature_degF'].values,
                            x_test_new['Solar_Irradiation_Wpm2'].values,
                            x_test_new['Wind_Speed_mps'].values], axis=1)

# y值（电压和电流）
y_test_new = np.stack([y_test_new['DC_Voltage'].values, y_test_new['DC_Current'].values], axis=1)

# 对新的测试集数据进行标准化
X_test_transformer_new_standardized = scaler_X_transformer.transform(X_test_transformer_new)
X_test_lstm_new_standardized = scaler_X_lstm.transform(X_test_lstm_new)
y_test_new_standardized = scaler_y.transform(y_test_new)

# 将新的测试集数据转换为Tensor
X_test_tensor_transformer_new = torch.tensor(X_test_transformer_new_standardized, dtype=torch.float32).unsqueeze(1).to('cuda')
X_test_tensor_lstm_new = torch.tensor(X_test_lstm_new_standardized, dtype=torch.float32).unsqueeze(1).to('cuda')
y_test_tensor_new = torch.tensor(y_test_new_standardized, dtype=torch.float32).to('cuda')

# 对新的测试集使用Transformer模型进行递归预测
with torch.no_grad():
    predicted_values_transformer_new = transformer_model(X_test_tensor_transformer_new).cpu().numpy()
    predicted_values_transformer_new = scaler_y.inverse_transform(predicted_values_transformer_new)

# 对新的测试集使用LSTM模型进行预测
with torch.no_grad():
    lstm_predictions_new = lstm_model(X_test_tensor_lstm_new).cpu().numpy()
    lstm_predictions_new = scaler_y.inverse_transform(lstm_predictions_new)

# 将新的测试集的y值逆标准化
y_test_original_new = scaler_y.inverse_transform(y_test_tensor_new.cpu().numpy())

# 使用基于验证集设定的阈值进行异常检测
anomalies_voltage_new = detect_anomalies(y_test_original_new[:, 0], predicted_values_transformer_new[:, 0], voltage_threshold)  # 使用验证集设定的阈值
anomalies_current_new = detect_anomalies(y_test_original_new[:, 1], predicted_values_transformer_new[:, 1], current_threshold)  # 使用验证集设定的阈值

# 当检测到异常时，进行替换操作
if np.any(anomalies_voltage_new) or np.any(anomalies_current_new):
    y_corrected_voltage_new = replace_anomalies_with_predictions(y_test_original_new[:, 0], predicted_values_transformer_new[:, 0],
                                                                 lstm_predictions_new[:, 0], anomalies_voltage_new)
    y_corrected_current_new = replace_anomalies_with_predictions(y_test_original_new[:, 1], predicted_values_transformer_new[:, 1],
                                                                 lstm_predictions_new[:, 1], anomalies_current_new)
else:
    y_corrected_voltage_new = y_test_original_new[:, 0]  # 如果没有检测到异常，保持原始数据
    y_corrected_current_new = y_test_original_new[:, 1]

# 合并修正后的数据
y_corrected_new = np.column_stack((y_corrected_voltage_new, y_corrected_current_new))

# 电能误差计算
time_interval_hours = 5 / 60  # 5分钟间隔

true_power_new = y_test_original_new[:, 0] * y_test_original_new[:, 1]  # 真实功率 = 真实电压 * 真实电流
predicted_power_new = y_corrected_new[:, 0] * y_corrected_new[:, 1]  # 预测功率 = 预测电压 * 预测电流

true_energy_kwh_new = np.sum(true_power_new) * time_interval_hours / 1000  # 真实电能(kWh)
predicted_energy_kwh_new = np.sum(predicted_power_new) * time_interval_hours / 1000  # 预测电能(kWh)

energy_error_kwh_new = predicted_energy_kwh_new - true_energy_kwh_new
percentage_error_new = (energy_error_kwh_new / true_energy_kwh_new) * 100

# 打印电能误差
print(f"True cumulative energy (kWh): {true_energy_kwh_new:.4f} kWh")
print(f"Predicted cumulative energy (kWh): {predicted_energy_kwh_new:.4f} kWh")
print(f"Energy error (kWh): {energy_error_kwh_new:.4f} kWh")
print(f"Percentage error: {percentage_error_new:.2f}%")







plt.figure(figsize=(8, 6))
plt.plot(y_test_original_new[:, 0], label='True Voltage (V)', color='blue')  # 真实电压
plt.plot(y_corrected_new[:, 0], label='Corrected Voltage (V)', color='red')  # 异常替换后的电压
plt.xlabel('Time Step')
plt.ylabel('Voltage (V)')
plt.title('True vs Corrected Voltage (V) on New Test Data')
plt.legend()
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(y_test_original_new[:, 1], label='True Current (A)', color='blue')  # 真实电流
plt.plot(y_corrected_new[:, 1], label='Corrected Current (A)', color='red')  # 异常替换后的电流
plt.xlabel('Time Step')
plt.ylabel('Current (A)')
plt.title('True vs Corrected Current (A) on New Test Data')
plt.legend()
plt.show()
