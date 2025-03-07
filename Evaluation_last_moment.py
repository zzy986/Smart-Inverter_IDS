import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
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



class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3,):
        super(GRURegressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 初始化隐藏层状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # 前向传播GRU
        out, _ = self.gru(x, h0)
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out


class RNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(RNNRegressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True,dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 初始化隐藏层状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # 前向传播RNN
        out, _ = self.rnn(x, h0)
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out
# 读取数据
np.random.seed(42)
torch.manual_seed(42)

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 读取数据
x_data_2023= pd.read_csv('Sensor_2023_daytime_1.csv')
y_data_2023= pd.read_csv('Inverter_2023_daytime_1.csv')
x_data_2022 = pd.read_csv('Sensor_2022_daytime_1.csv')
y_data_2022 = pd.read_csv('Inverter_2022_daytime_1.csv')

# 合并2022年和2023年的数据
x_data_combined = pd.concat([x_data_2022, x_data_2023], axis=0, ignore_index=True)
y_data_combined = pd.concat([y_data_2022, y_data_2023], axis=0, ignore_index=True)

# 添加上一时刻的电压和电流作为特征，并用零值填充第一行
y_data_combined['DC_Voltage_Lag'] = y_data_combined['DC_Voltage'].shift(1, fill_value=0)
y_data_combined['DC_Current_Lag'] = y_data_combined['DC_Current'].shift(1, fill_value=0)



print(y_data_combined['DC_Voltage_Lag'])
# 重新构建特征矩阵 X，加入滞后特征
X = np.stack([x_data_combined['Module_Temperature_degF'].values,
              x_data_combined['Ambient_Temperature_degF'].values,
              x_data_combined['Solar_Irradiation_Wpm2'].values,
              x_data_combined['Wind_Speed_mps'].values,
              y_data_combined['DC_Voltage_Lag'].values,  # 上一时刻的电压
              y_data_combined['DC_Current_Lag'].values],  # 上一时刻的电流
              axis=1)

# y 仍然是当前时刻的电压和电流
y = np.stack([y_data_combined['DC_Voltage'].values, y_data_combined['DC_Current'].values], axis=1)
current_min = y_data_combined['DC_Current'].min()
current_max = y_data_combined['DC_Current'].max()

print(f"DC_Current 数据的最小值: {current_min}")
print(f"DC_Current 数据的最大值: {current_max}")
# 标准化特征
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# 按照顺序划分数据集
n_samples = X.shape[0]
train_size = int(n_samples * 0.8)

# 前80%的数据作为训练集
X_train = X[:train_size]
y_train = y[:train_size]

# 后20%的数据作为临时集
X_temp = X[train_size:]
y_temp = y[train_size:]

# 划分验证集和测试集
n_temp_samples = X_temp.shape[0]
valid_size = int(n_temp_samples * 0.5)

# 前50%的临时数据作为验证集
X_valid = X_temp[:valid_size]
y_valid = y_temp[:valid_size]

# 后50%的临时数据作为测试集
X_test = X_temp[valid_size:]
y_test = y_temp[valid_size:]

# 转换为PyTorch的张量格式
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).unsqueeze(1).to(device)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training set size: {X_train_tensor.size(0)} samples")
print(f"Validation set size: {X_valid_tensor.size(0)} samples")
print(f"Test set size: {X_test_tensor.size(0)} samples")


model = TransformerRegressor(input_dim=X_test_tensor.shape[2], output_dim=y_test_tensor.shape[1]).to('cuda')

model.load_state_dict(torch.load('Transformer_best_model_last_moment_1A.pth'))

# 确认加载权重后模型状态
#print("After loading weights, model is in train mode:", transformer_model.training)  # 应该还是 True

# 切换到评估模式
model.eval()

# 再次确认模型是否进入评估模式
#print("After calling eval(), model is in eval mode:", not transformer_model.training)  # 应该返回 False (评估模式)

# 对测试集进行预测
with torch.no_grad():
    predictions =model(X_test_tensor.to('cuda'))

# 将预测值转换为 NumPy 数组
predictions_np = predictions.cpu().numpy()

# 逆标准化预测结果
predictions_original = scaler_y.inverse_transform(predictions_np)

# 逆标准化真实值
y_test_original = scaler_y.inverse_transform(y_test_tensor.cpu().numpy())


# 评估模型
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_test_voltage = y_test_original[:, 0]
y_test_current = y_test_original[:, 1]
y_pred_voltage = predictions_original[:, 0]
y_pred_current = predictions_original[:, 1]

# 分别计算电压和电流的 MAE, MSE, RMSE
mae_voltage = torch.mean(torch.abs(torch.tensor(y_test_voltage) - torch.tensor(y_pred_voltage))).item()
mse_voltage = torch.mean((torch.tensor(y_test_voltage) - torch.tensor(y_pred_voltage)) ** 2).item()
rmse_voltage = torch.sqrt(torch.mean((torch.tensor(y_test_voltage) - torch.tensor(y_pred_voltage)) ** 2)).item()

mae_current = torch.mean(torch.abs(torch.tensor(y_test_current) - torch.tensor(y_pred_current))).item()
mse_current = torch.mean((torch.tensor(y_test_current) - torch.tensor(y_pred_current)) ** 2).item()
rmse_current = torch.sqrt(torch.mean((torch.tensor(y_test_current) - torch.tensor(y_pred_current)) ** 2)).item()

# 计算 R²
r2_voltage = r2_score(y_test_voltage, y_pred_voltage)
r2_current = r2_score(y_test_current, y_pred_current)
def smape(y_true, y_pred):
    return 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

smape_voltage = smape(np.array(y_test_voltage), np.array(y_pred_voltage))
smape_current = smape(np.array(y_test_current), np.array(y_pred_current))

dc_voltage = y_data_combined['DC_Voltage'].values
dc_voltage_min = np.min(dc_voltage)
dc_voltage_max = np.max(dc_voltage)

print(f"DC Voltage Range: {dc_voltage_min} to {dc_voltage_max}")
print(f"Voltage(V)- MAE: {mae_voltage:.4f}, MSE: {mse_voltage:.4f}, RMSE: {rmse_voltage:.4f}, R²: {r2_voltage:.4f},SMAPE: {smape_voltage:.2f}%")
print(f"Current(A)- MAE: {mae_current:.4f}, MSE: {mse_current:.4f}, RMSE: {rmse_current:.4f}, R²: {r2_current:.4f},SMAPE: {smape_current:.2f}%")
# 可视化结果
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(y_test_original[0:1000, 0], label='True Voltage (V)', color='blue')
plt.plot(predictions_original[0:1000, 0], label='Predicted Voltage (V)', color='orange')
plt.xlabel('Time Step')
plt.ylabel('Voltage (V)')
plt.title('LSTM - Voltage (V) Test Predictions')
plt.legend()
#plt.savefig('LSTM_voltage_2_year_V.pdf')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(y_test_original[0:1000, 1], label='True Current (A)', color='blue')
plt.plot(predictions_original[0:1000, 1], label='Predicted Current (A)', color='orange')
plt.xlabel('Time Step')
plt.ylabel('Current (A)')
plt.title('LSTM- Current (A) Test Predictions')
plt.legend()
#plt.savefig('LSTM_Current_2_year_I.pdf')
plt.show()
