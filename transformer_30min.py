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

print(torch.cuda.get_arch_list())


# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

print(torch.cuda.is_available())

# 检查GPU是否可用
device = torch.device('cuda')

x_data= pd.read_csv('Test_sensor_2023.csv')

y_data= pd.read_csv('Test_2023.csv')


X = np.stack([x_data['Module_Temperature_degF'].values,
              x_data['Ambient_Temperature_degF'].values,
              x_data['Solar_Irradiation_Wpm2'].values,
              x_data['Wind_Speed_mps'].values], axis=1)

y = np.stack([y_data['DC_Voltage'].values, y_data['DC_Current'].values], axis=1)

# 可视化电压和电流数据
plt.plot(y_data['DC_Voltage'][0:500], label='DC Voltage')
plt.plot(y_data['DC_Current'][0:500], label='DC Current')
plt.legend()
plt.title('DC Voltage and Current')
plt.show()

# 标准化特征
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# 按照顺序划分数据集
n_samples = X.shape[0]
train_size = int(n_samples * 0.6)  # 前60%为训练集
temp_size = n_samples - train_size  # 后40%为临时集

# 前60%的数据作为训练集
X_train = X[:train_size]
y_train = y[:train_size]

# 后40%的数据作为临时集
X_temp = X[train_size:]
y_temp = y[train_size:]

# 手动划分验证集和测试集
n_temp_samples = X_temp.shape[0]
valid_size = int(n_temp_samples * 0.5)  # 前50%为验证集

# 前50%的临时数据作为验证集
X_valid = X_temp[:valid_size]
y_valid = y_temp[:valid_size]

# 后50%的临时数据作为测试集
X_test = X_temp[valid_size:]
y_test = y_temp[valid_size:]

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)  # 添加时间步维度
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).unsqueeze(1).to(device)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)


print(f"Training set size: {X_train_tensor.size(0)} samples")
print(f"Validation set size: {X_valid_tensor.size(0)} samples")
print(f"Test set size: {X_test_tensor.size(0)} samples")
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_encoder_layers=8, num_decoder_layers=8,
                 dim_feedforward=128, dropout=0.1):
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
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.4):
        super(LSTMRegressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
"""
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
  'RNN': RNNRegressor(input_dim=X_train_tensor.shape[2], hidden_dim=256, output_dim=y_train_tensor.shape[1]).to(
        device),
    'GRU': GRURegressor(input_dim=X_train_tensor.shape[2], hidden_dim=256, output_dim=y_train_tensor.shape[1]).to(device)
"""
#'Transformer': TransformerRegressor(input_dim=X_train_tensor.shape[2], output_dim=y_train_tensor.shape[1]).to(
       # device),
models = {

    'LSTM': LSTMRegressor(input_dim=X_train_tensor.shape[2], hidden_dim=512, output_dim=y_train_tensor.shape[1]).to(
        device,),

}

epochs=500
# 通用训练和评估函数
def train_and_evaluate_model(model, X_train_tensor, y_train_tensor, X_valid_tensor, y_valid_tensor, X_test_tensor,
                             y_test_tensor, optimizer, criterion, epochs=epochs):
    train_losses = []
    valid_losses = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            valid_predictions = model(X_valid_tensor)
            valid_loss = criterion(valid_predictions, y_valid_tensor)
            valid_losses.append(valid_loss.item())

        scheduler.step(valid_loss)  # 使用学习率调度器

    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor)
        test_loss = criterion(test_predictions, y_test_tensor)

    return train_losses, valid_losses, test_loss.item(), test_predictions

# 存储结果
results = {}

# 训练和评估每个模型
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)  # Ensure correct optimizer initialization
    criterion = torch.nn.MSELoss()  # Ensure correct criterion initialization

    # Correct the function call by passing all required arguments properly
    train_losses, valid_losses, test_loss, test_predictions = train_and_evaluate_model(
        model, X_train_tensor, y_train_tensor, X_valid_tensor, y_valid_tensor, X_test_tensor, y_test_tensor,
        optimizer, criterion, epochs=epochs  # Ensure 'epochs' is passed as an integer
    )

    results[model_name] = {
        'Train Loss': train_losses,
        'Valid Loss': valid_losses,
        'Test Loss': test_loss,
        'Predictions': test_predictions
    }

    print(f"{model_name} Test Loss: {test_loss:.4f}")

# 使用PyTorch进行逆标准化和计算
y_test_tensor_cpu = y_test_tensor.cpu()
y_test_np = scaler_y.inverse_transform(y_test_tensor_cpu.numpy())  # 原始真实值
y_pred_np = {}
for model_name, result in results.items():
    y_pred_standardized = result['Predictions'].cpu().numpy()

    # 逆标准化
    y_pred_np[model_name] = torch.tensor(scaler_y.inverse_transform(y_pred_standardized))

    # 计算MAE, MSE, RMSE, R2
    mae = torch.mean(torch.abs(y_test_tensor_cpu - torch.tensor(y_pred_standardized))).item()
    mse = torch.mean((y_test_tensor_cpu - torch.tensor(y_pred_standardized)) ** 2).item()
    rmse = torch.sqrt(torch.mean((y_test_tensor_cpu - torch.tensor(y_pred_standardized)) ** 2)).item()
    r2 = r2_score(y_test_tensor_cpu.numpy(), y_pred_standardized)

    print(f"\n{model_name} Performance on Test Data (Original Scale):")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")

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
    plt.figure(figsize=(12, 5))

    # 电压 (V) 时间序列对比
    plt.subplot(1, 2, 1)
    plt.plot(y_test_np[0:500, 0], label='True Voltage (V)', color='blue')
    plt.plot(y_pred_np[model_name][0:500, 0], label='Predicted Voltage (V)', color='orange')
    plt.xlabel('Time Step')
    plt.ylabel('Voltage (V)')
    plt.title(f'{model_name} - Voltage (V) Time Series')
    plt.legend()

    # 电流 (I) 时间序列对比
    plt.subplot(1, 2, 2)
    plt.plot(y_test_np[0:500, 1], label='True Current (I)', color='blue')
    plt.plot(y_pred_np[model_name][0:500, 1], label='Predicted Current (I)', color='orange')
    plt.xlabel('Time Step')
    plt.ylabel('Current (I)')
    plt.title(f'{model_name} - Current (I) Time Series')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 绘制训练损失曲线
plt.figure(figsize=(12, 6))
for model_name, result in results.items():
    plt.plot(result['Train Loss'], label=f'{model_name} Train Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss for Different Models')
plt.legend()
plt.show()

# 绘制验证损失曲线
plt.figure(figsize=(12, 6))
for model_name, result in results.items():
    plt.plot(result['Valid Loss'], label=f'{model_name} Valid Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss for Different Models')
plt.legend()
plt.show()
