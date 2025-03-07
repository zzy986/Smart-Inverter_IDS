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


import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 假设 TransformerRegressor 和 LSTMRegressor 已经定义
# from your_model_definitions import TransformerRegressor, LSTMRegressor

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
X_transformer = np.stack([
    x_data_combined['Module_Temperature_degF'].values,
    x_data_combined['Ambient_Temperature_degF'].values,
    x_data_combined['Solar_Irradiation_Wpm2'].values,
    x_data_combined['Wind_Speed_mps'].values,
    y_data_combined['DC_Voltage_Lag'].values,
    y_data_combined['DC_Current_Lag'].values
], axis=1)

# LSTM模型使用的输入特征（仅传感器）
X_lstm = np.stack([
    x_data_combined['Module_Temperature_degF'].values,
    x_data_combined['Ambient_Temperature_degF'].values,
    x_data_combined['Solar_Irradiation_Wpm2'].values,
    x_data_combined['Wind_Speed_mps'].values
], axis=1)

y = np.stack([
    y_data_combined['DC_Voltage'].values,
    y_data_combined['DC_Current'].values
], axis=1)

# 标准化
scaler_X_transformer = StandardScaler()
scaler_X_lstm = StandardScaler()
scaler_y = StandardScaler()

X_transformer_standardized = scaler_X_transformer.fit_transform(X_transformer)
X_lstm_standardized = scaler_X_lstm.fit_transform(X_lstm)
y_standardized = scaler_y.fit_transform(y)

# 验证集切分
# 获取样本数量
n_samples = X_transformer_standardized.shape[0]

# 计算训练集大小 (80%)
train_size = int(n_samples * 0.8)

# 剩下的 20% 均分为验证集和测试集
remaining_size = n_samples - train_size
valid_size = int(remaining_size * 0.5)  # 剩余数据的 50% 作为验证集
test_size = remaining_size - valid_size  # 剩余数据的另外 50% 作为测试集

# 确保划分正确，打印每个数据集的大小
print(f"Total samples: {n_samples}")
print(f"Training set size: {train_size}")
print(f"Validation set size: {valid_size}")
print(f"Test set size: {test_size}")

x_window= pd.read_csv('window_Sensor.csv')
y_window= pd.read_csv('window_inverter.csv')


y_window['DC_Voltage_Lag'] = y_window['DC_Voltage'].shift(1, fill_value=0)
y_window['DC_Current_Lag'] = y_window['DC_Current'].shift(1, fill_value=0)


x_window_transformer= np.stack([
    x_window['Module_Temperature_degF'].values,
    x_window['Ambient_Temperature_degF'].values,
    x_window['Solar_Irradiation_Wpm2'].values,
    x_window['Wind_Speed_mps'].values,
    y_window['DC_Voltage_Lag'].values,
    y_window['DC_Current_Lag'].values
], axis=1)

x_window_transformer_std= scaler_X_transformer.transform(x_window_transformer)


x_window_lstm = np.stack([
    x_window['Module_Temperature_degF'].values,
    x_window['Ambient_Temperature_degF'].values,
    x_window['Solar_Irradiation_Wpm2'].values,
    x_window['Wind_Speed_mps'].values
], axis=1)

x_window_lstm_std= scaler_X_lstm.transform(x_window_lstm)

y_window_test=np.stack([
    y_window['DC_Voltage'].values,
    y_window['DC_Current'].values
], axis=1)

y_window_test_std=scaler_y.transform(y_window_test)
# 数据集划分：Transformer输入
X_train_transformer = X_transformer_standardized[:train_size]
X_valid_transformer = X_transformer_standardized[train_size:train_size + valid_size]
X_test_transformer = X_transformer_standardized[train_size + valid_size:]

# 数据集划分：LSTM输入
X_train_lstm = X_lstm_standardized[:train_size]
X_valid_lstm = X_lstm_standardized[train_size:train_size + valid_size]
X_test_lstm = X_lstm_standardized[train_size + valid_size:]

# 目标变量划分
y_train = y_standardized[:train_size]
y_valid = y_standardized[train_size:train_size + valid_size]
y_test = y_standardized[train_size + valid_size:]


# 转换为Tensor格式
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
X_train_tensor_transformer = torch.tensor(X_train_transformer, dtype=torch.float32).unsqueeze(1).to(device)
X_valid_tensor_transformer = torch.tensor(X_valid_transformer, dtype=torch.float32).unsqueeze(1).to(device)
X_test_tensor_transformer = torch.tensor(X_test_transformer, dtype=torch.float32).unsqueeze(1).to(device)

X_train_tensor_lstm = torch.tensor(X_train_lstm, dtype=torch.float32).unsqueeze(1).to(device)
X_valid_tensor_lstm = torch.tensor(X_valid_lstm, dtype=torch.float32).unsqueeze(1).to(device)
X_test_tensor_lstm = torch.tensor(X_test_lstm, dtype=torch.float32).unsqueeze(1).to(device)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# 加载预训练模型
transformer_model = TransformerRegressor(input_dim=X_test_tensor_transformer.shape[2], output_dim=2).to(device)
lstm_model = LSTMRegressor(input_dim=X_test_tensor_lstm.shape[2], hidden_dim=128, output_dim=y_test_tensor.shape[1]).to(device)

# 假设Transformer和LSTM模型已经训练并保存了权重
transformer_model.load_state_dict(torch.load('Transformer_best_model_last_moment_1A.pth'))
lstm_model.load_state_dict(torch.load('LSTMRegressor_best_model_1A.pth'))
def recursive_predict(model, X_test_tensor, lstm_predictions, initial_transformer_predictions, steps, scaler_y, device, fusion_weight=1):
    """
    递归预测函数
    :param model: 预测模型 (Transformer)
    :param X_test_tensor: 测试集输入数据
    :param lstm_predictions: LSTM 的预测结果
    :param initial_transformer_predictions: Transformer 的初始预测结果
    :param steps: 递归步数
    :param scaler_y: 用于逆标准化的 scaler
    :param device: 使用的设备 (CPU or GPU)
    :param fusion_weight: LSTM 和 Transformer 融合的加权参数，默认为 1 (即仅用LSTM)
    :return: 逆标准化后的融合预测结果
    """
    model.eval()
    predictions = []

    # 否则正常进行递归预测
    y_corrected_dynamic = initial_transformer_predictions.copy()

    for t in range(steps):
        # 获取当前时间步的输入数据
        X_current = X_test_tensor[t:t + 1, :, :].clone().to(device)

        if t > 0:
            # 更新滞后电压和电流特征，仅影响 Transformer，不影响 LSTM
            X_current[:, :, -2] = torch.tensor([y_corrected_dynamic[t - 1, 0]]).unsqueeze(0).to(device)  # 更新滞后电压
            X_current[:, :, -1] = torch.tensor([y_corrected_dynamic[t - 1, 1]]).unsqueeze(0).to(device)  # 更新滞后电流

        # 使用 Transformer 进行预测
        with torch.no_grad():
            y_pred = model(X_current)

        # 获取当前预测值，并转化为 numpy
        y_pred_np = y_pred.cpu().numpy()

        # 逆标准化 Transformer 的预测值
        transformer_pred_original = scaler_y.inverse_transform(y_pred_np)

        # 计算当前步的 LSTM 预测（已逆标准化），直接从 lstm_predictions 获取，不进行更新
        lstm_pred_original = lstm_predictions[t].reshape(1, -1)  # LSTM 预测值在递归过程中不变

        # 根据 LSTM 和 Transformer 预测值加权融合
        y_corrected_original = 0.9*lstm_pred_original+0.1*transformer_pred_original

        # 将融合后的预测值标准化，便于递归使用
        y_corrected_dynamic[t] = scaler_y.transform(y_corrected_original)

        # 保存当前步的融合预测值，而不是 Transformer 的预测值
        predictions.append(y_corrected_original)

    # 将所有融合预测结果拼接
    predictions = np.concatenate(predictions, axis=0)

    # 返回已经逆标准化的融合预测结果
    return (predictions)




# 生成初始Transformer预测，并初始化修正值
steps = X_test_tensor_transformer.shape[0]  # 预测的步数

# LSTM模型的预测
lstm_model.eval()
with torch.no_grad():
    lstm_predictions = lstm_model(X_test_tensor_lstm).cpu().numpy()
    lstm_predictions = scaler_y.inverse_transform(lstm_predictions)  # 逆标准化 LSTM 的预测

# 初始Transformer预测，作为输入
initial_transformer_predictions = recursive_predict(
    transformer_model, X_test_tensor_transformer, lstm_predictions, lstm_predictions, steps, scaler_y, device
)

# 使用加权后的 y 值递归预测
transformer_predictions_corrected = recursive_predict(
    transformer_model, X_test_tensor_transformer, lstm_predictions, initial_transformer_predictions, steps, scaler_y, device, fusion_weight=0.5
)

# 打印或返回最终预测





# 可视化替换效果
# 可视化替换效果
def plot_replacement_effect(y_true, y_corrected, title):
    plt.figure(figsize=(14, 8))

    # 绘制DC电压的原始和修正后的结果
    plt.subplot(2, 1, 1)
    plt.plot(y_true[:, 0], label='Original DC Voltage', color='blue')
    plt.plot(y_corrected[:, 0], label='Corrected DC Voltage', color='green')
    plt.legend()
    plt.title(f'{title} - DC Voltage')
    plt.xlabel("Time Step")
    plt.ylabel("Voltage (V)")

    # 绘制DC电流的原始和修正后的结果
    plt.subplot(2, 1, 2)
    plt.plot(y_true[:, 1], label='Original DC Current', color='blue')
    plt.plot(y_corrected[:, 1], label='Corrected DC Current', color='orange')
    plt.legend()
    plt.title(f'{title} - DC Current')
    plt.xlabel("Time Step")
    plt.ylabel("Current (A)")

    plt.tight_layout()
    plt.show()

    # 定义时间间隔，单位：小时
    time_interval_hours = 5 / 60  # 5分钟采样时间

    # 计算真实功率和预测功率
    true_power = y_true[:, 0] * y_true[:, 1]  # 真实功率 = 真实电压 * 真实电流
    predicted_power = y_corrected[:, 0] * y_corrected[:, 1]  # 预测功率 = 预测电压 * 预测电流

    # 计算累计电能 (kWh)
    true_energy_kwh = np.sum(true_power) * time_interval_hours / 1000  # 转换为kWh
    predicted_energy_kwh = np.sum(predicted_power) * time_interval_hours / 1000  # 转换为kWh

    # 计算电能误差和百分比误差
    energy_error_kwh = predicted_energy_kwh - true_energy_kwh
    percentage_error = (energy_error_kwh / true_energy_kwh) * 100

    # 打印电能误差结果
    print(f"True cumulative energy (kWh): {true_energy_kwh:.4f} kWh")
    print(f"Predicted cumulative energy (kWh): {predicted_energy_kwh:.4f} kWh")
    print(f"Energy error (kWh): {energy_error_kwh:.4f} kWh")
    print(f"Percentage error: {percentage_error:.2f}%")
y_test_original= scaler_y.inverse_transform(y_test)
# 绘制替换效果图，验证集的替换效果
plot_replacement_effect(y_test_original, transformer_predictions_corrected, title="Test Set Data")


