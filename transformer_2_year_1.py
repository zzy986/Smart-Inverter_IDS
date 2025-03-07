import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
print(torch.cuda.get_arch_list())



class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=32, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
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
# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

print(torch.cuda.is_available())

# 检查GPU是否可用
device = torch.device('cuda')

x_data_2023= pd.read_csv('Sensor_2023_daytime_1.csv')

y_data_2023= pd.read_csv('Inverter_2023_daytime_1.csv')


x_data_2022 = pd.read_csv('Sensor_2022_daytime_1.csv')
y_data_2022 = pd.read_csv('Inverter_2022_daytime_1.csv')



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 假设 x_data_2022 和 y_data_2022 已经包含了数据，并且 timestamp 已经作为列存在
x_data_combined = pd.concat([x_data_2022], axis=0, ignore_index=True)
y_data_combined = pd.concat([y_data_2022], axis=0, ignore_index=True)

print(x_data_combined.head())  # 检查数据
print(x_data_combined.tail())

# 确保 timestamp 是 datetime 格式，并按时间排序
x_data_combined['Timestamp'] = pd.to_datetime(x_data_combined['Timestamp'])
x_data_combined = x_data_combined.sort_values(by='Timestamp').reset_index(drop=True)
y_data_combined = y_data_combined.sort_values(by='Timestamp').reset_index(drop=True)

# 根据月份划分数据集
train_mask = x_data_combined['Timestamp'].dt.month <= 8
valid_mask = (x_data_combined['Timestamp'].dt.month > 8) & (x_data_combined['Timestamp'].dt.month <= 10)
test_mask = x_data_combined['Timestamp'].dt.month > 10
# 打印对应的实际数据
print("Train Data:")
print(x_data_combined[train_mask].head())
print(x_data_combined[train_mask].tail())

print("Validation Data:")
print(x_data_combined[valid_mask].head())
print(x_data_combined[valid_mask].tail())

print("Test Data:")
print(x_data_combined[test_mask].head())
print(x_data_combined[test_mask].tail())
# 分别提取 X 和 y 的训练、验证、测试集
X_train = x_data_combined[train_mask][['Module_Temperature_degF', 'Ambient_Temperature_degF',
                                       'Solar_Irradiation_Wpm2', 'Wind_Speed_mps']].values
y_train = y_data_combined[train_mask][['DC_Voltage', 'DC_Current']].values

X_valid = x_data_combined[valid_mask][['Module_Temperature_degF', 'Ambient_Temperature_degF',
                                       'Solar_Irradiation_Wpm2', 'Wind_Speed_mps']].values
y_valid = y_data_combined[valid_mask][['DC_Voltage', 'DC_Current']].values

X_test = x_data_combined[test_mask][['Module_Temperature_degF', 'Ambient_Temperature_degF',
                                     'Solar_Irradiation_Wpm2', 'Wind_Speed_mps']].values
y_test = y_data_combined[test_mask][['DC_Voltage', 'DC_Current']].values

# 标准化特征和标签
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_valid = scaler_X.transform(X_valid)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train)
y_valid = scaler_y.transform(y_valid)
y_test = scaler_y.transform(y_test)

# 打印检查划分结果
print(f"训练集样本数: {X_train.shape[0]}")
print(f"验证集样本数: {X_valid.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")


X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)  # 添加时间步维度
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).unsqueeze(1).to(device)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)



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

epochs=5000

class LSTMRegressor_1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.2):
        super(LSTMRegressor_1, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # 修改卷积层的输入通道为 hidden_dim
        self.conv1d = nn.Conv1d(in_channels=hidden_dim, out_channels=output_dim, kernel_size=1)

    def forward(self, x):
        # 初始化LSTM的隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # 通过LSTM层
        out, _ = self.lstm(x, (h0, c0))  # out形状: [batch_size, seq_length, hidden_dim]

        # 调整形状以适应Conv1d: [batch_size, hidden_dim, seq_length]
        out = out.permute(0, 2, 1)

        # 通过卷积层
        out = self.conv1d(out)  # 输出形状: [batch_size, output_dim, seq_length]

        # 压缩输出的最后一个维度（如果需要）
        out = out.squeeze(-1)  # 输出形状: [batch_size, output_dim]

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
# 初始化模型列表
 # 'RNN': RNNRegressor(input_dim=X_train_tensor.shape[2], hidden_dim=256, output_dim=y_train_tensor.shape[1]).to(
       # device),
"""
'LSTM': LSTMRegressor(input_dim=X_train_tensor.shape[2], hidden_dim=256, output_dim=y_train_tensor.shape[1]).to(
        device,),
'GRU': GRURegressor(input_dim=X_train_tensor.shape[2], hidden_dim=256, output_dim=y_train_tensor.shape[1]).to(device),
'RNN': RNNRegressor(input_dim=X_train_tensor.shape[2], hidden_dim=256, output_dim=y_train_tensor.shape[1]).to(
        device),
#'Transformer': TransformerRegressor(input_dim=X_train_tensor.shape[2], output_dim=y_train_tensor.shape[1]).to(
       # device),
       
"""
models = {
    'Transformer': TransformerRegressor(input_dim=X_train_tensor.shape[2], output_dim=y_train_tensor.shape[1]).to(device),

}


class EarlyStopping:
    def __init__(self, patience=30, delta=0, save_path='best_model_1_year.pth'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_path = save_path
        self.best_loss = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''验证损失下降时保存模型'''
        if val_loss < self.best_loss:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), self.save_path)
            self.best_loss = val_loss


def train_model(model, train_loader, valid_loader, optimizer, criterion, epochs, early_stopping):
    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        train_losses.append(epoch_train_loss / len(train_loader))

        model.eval()
        epoch_valid_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                valid_predictions = model(X_batch)
                valid_loss = criterion(valid_predictions, y_batch)
                epoch_valid_loss += valid_loss.item()

        valid_loss_avg = epoch_valid_loss / len(valid_loader)
        valid_losses.append(valid_loss_avg)

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Valid Loss: {valid_loss_avg:.4f}')

        early_stopping(valid_loss_avg, model)

        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    model.load_state_dict(torch.load(f'{model_name}_best_model_1_year.pth'))

    return train_losses, valid_losses

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    test_predictions = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            test_loss += loss.item()
            test_predictions.append(predictions.cpu().numpy())

    test_loss /= len(test_loader)
    test_predictions = np.vstack(test_predictions)

    return test_loss, test_predictions

# 循环遍历每个模型并训练和测试
results = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()
    early_stopping = EarlyStopping(patience=30, save_path=f'{model_name}_best_model_1_year.pth')

    # 训练模型
    train_losses, valid_losses = train_model(model, train_loader, valid_loader, optimizer, criterion, epochs, early_stopping)

    # 测试模型
    test_loss, test_predictions = test_model(model, test_loader, criterion)

    # 存储结果
    results[model_name] = {
        'Train Loss': train_losses,
        'Valid Loss': valid_losses,
        'Test Loss': test_loss,
        'Predictions': test_predictions
    }

    print(f"{model_name} Test Loss: {test_loss:.4f}")

# 进行逆标准化和评估
y_test_tensor_cpu = y_test_tensor.cpu()
y_test_np = scaler_y.inverse_transform(y_test_tensor_cpu.numpy())  # 真实值

# 逆标准化每个模型的预测
y_pred_np = {}
for model_name, result in results.items():
    y_pred_standardized = result['Predictions']
    y_pred_original = scaler_y.inverse_transform(y_pred_standardized)
    y_pred_np[model_name] = y_pred_original

# 计算并输出评估结果
for model_name, y_pred in y_pred_np.items():
    y_test_voltage = y_test_np[:, 0]
    y_test_current = y_test_np[:, 1]
    y_pred_voltage = y_pred[:, 0]
    y_pred_current = y_pred[:, 1]

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

    print(f"\n{model_name} Performance on Test Data_1_year_sensor:")
    print(f"Voltage (V) - MAE: {mae_voltage:.4f}, MSE: {mse_voltage:.4f}, RMSE: {rmse_voltage:.4f}, R²: {r2_voltage:.4f}")
    print(f"Current (I) - MAE: {mae_current:.4f}, MSE: {mse_current:.4f}, RMSE: {rmse_current:.4f}, R²: {r2_current:.4f}")

    # 绘制预测值与真实值的对比图
    plt.figure(figsize=(12, 5))

    # 电压 (V) 对比
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_np[:, 0], y_pred_np[model_name][:, 0], alpha=0.5)
    plt.plot([y_test_np[:, 0].min(), y_test_np[:, 0].max()], [y_test_np[:, 0].min(), y_test_np[:, 0].max()], 'r--')
    plt.xlabel('True Voltage (V)')
    plt.ylabel('Predicted Voltage (V)')
    plt.title(f'{model_name} - Voltage (V) Prediction')

    # 电流 (I) 对比
    plt.subplot(1, 2, 2)
    plt.scatter(y_test_np[:, 1], y_pred_np[model_name][:, 1], alpha=0.5)
    plt.plot([y_test_np[:, 1].min(), y_test_np[:, 1].max()], [y_test_np[:, 1].min(), y_test_np[:, 1].max()], 'r--')
    plt.xlabel('True Current (I)')
    plt.ylabel('Predicted Current (I)')
    plt.title(f'{model_name} - Current (I) Prediction')

    plt.tight_layout()
    plt.show()

    # 绘制时间序列图
    # 独立的电压 (V) 时间序列对比图
    plt.figure(figsize=(8, 6))  # 设置单个图的大小

    plt.plot(y_test_np[0:1000, 0], label='True Voltage (V)', color='blue')
    plt.plot(y_pred_np[model_name][0:1000, 0], label='Predicted Voltage (V)', color='orange',linestyle=':')
    plt.xlabel('Time Step')
    plt.ylabel('Voltage (V)')
    plt.title(f'{model_name} - Voltage (V) Time Series')
    plt.legend()
    plt.tight_layout()
    #plt.savefig(f'{model_name}_voltage_all.pdf')
    plt.show()

    # 独立的电流 (I) 时间序列对比图
    plt.figure(figsize=(8, 6))  # 设置单个图的大小
    plt.plot(y_test_np[0:1000, 1], label='True Current (I)', color='blue')
    plt.plot(y_pred_np[model_name][0:1000, 1], label='Predicted Current (I)', color='orange',linestyle=':')
    plt.xlabel('Time Step')
    plt.ylabel('Current (I)')
    plt.title(f'{model_name} - Current (I) Time Series')
    plt.legend()
    plt.tight_layout()
    #plt.savefig(f'{model_name}_Current_all.pdf')
    plt.show()

# 绘制训练损失曲线
plt.figure(figsize=(12, 6))
for model_name, result in results.items():
    plt.plot(result['Train Loss'], label=f'{model_name} Train Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss for Different Models')
plt.legend()
#plt.savefig('Training_loss.pdf')
plt.show()

# 绘制验证损失曲线
plt.figure(figsize=(12, 6))
for model_name, result in results.items():
    plt.plot(result['Valid Loss'], label=f'{model_name} Valid Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss for Different Models')
plt.legend()
#plt.savefig('Validation_loss.pdf')
plt.show()
