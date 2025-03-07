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
    plt.plot(y_test_original[:, 0], label='True Voltage (V)', color='blue')
    plt.plot(predictions_original[:, 0], label=f'{model_name} Predicted Voltage (V)', color='orange')
    plt.xlabel('Time Step')
    plt.ylabel('Voltage (V)')
    plt.title(f'{model_name} - Voltage (V) Test Predictions')
    plt.legend()
    plt.savefig(f'{model_name}_voltage_prediction_all.pdf')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(y_test_original[:, 1], label='True Current (A)', color='blue')
    plt.plot(predictions_original[:, 1], label=f'{model_name} Predicted Current (A)', color='orange')
    plt.xlabel('Time Step')
    plt.ylabel('Current (A)')
    plt.title(f'{model_name} - Current (A) Test Predictions')
    plt.legend()
    plt.savefig(f'{model_name}_current_prediction_all.pdf')
    plt.show()


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


# Models to test
models = {
'Transformer': TransformerRegressor(input_dim=X_train_tensor.shape[2], output_dim=y_train_tensor.shape[1]).to(
        device),
'LSTM': LSTMRegressor(input_dim=X_train_tensor.shape[2], hidden_dim=256, output_dim=y_train_tensor.shape[1]).to(
        device,),
'GRU': GRURegressor(input_dim=X_train_tensor.shape[2], hidden_dim=256, output_dim=y_train_tensor.shape[1]).to(device),
'RNN': RNNRegressor(input_dim=X_train_tensor.shape[2], hidden_dim=256, output_dim=y_train_tensor.shape[1]).to(
        device),

}

# Evaluate all models
for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    model.load_state_dict(torch.load(f'{model_name}_best_model_last_moment_1A.pth'))  # Load pre-trained model
    model.eval()

    with torch.no_grad():
        predictions = model(X_test_tensor.to(device))

    predictions_np = predictions.cpu().numpy()
    predictions_original = scaler_y.inverse_transform(predictions_np)
    y_test_original = scaler_y.inverse_transform(y_test_tensor.cpu().numpy())

    # Calculate and print metrics
    mae_voltage, mse_voltage, rmse_voltage, r2_voltage, smape_voltage = calculate_metrics(y_test_original[:, 0], predictions_original[:, 0])
    mae_current, mse_current, rmse_current, r2_current, smape_current = calculate_metrics(y_test_original[:, 1], predictions_original[:, 1])

    print(f"{model_name} - Voltage(V)- MAE: {mae_voltage:.4f}, MSE: {mse_voltage:.4f}, RMSE: {rmse_voltage:.4f}, R²: {r2_voltage:.4f}, SMAPE: {smape_voltage:.2f}%")
    print(f"{model_name} - Current(A)- MAE: {mae_current:.4f}, MSE: {mse_current:.4f}, RMSE: {rmse_current:.4f}, R²: {r2_current:.4f}, SMAPE: {smape_current:.2f}%")

    # Visualize results
    visualize_results(y_test_original, predictions_original, model_name)
print(torch.version.cuda)