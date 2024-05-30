import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse
from model.model import GCNRegression, GraphSAGERegression, GATRegression, GINRegression, EdgeRegression
from data.load_dataset import load_data
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from argument import argument
from util import calculate_mape, calculate_mae, df_to_graph



model_name = argument()
scaler = MinMaxScaler()



print('GNN Starting...')
df = load_data()
batch_size = 8
# num_nodes = 8
data_list = df_to_graph(df)
print(len(data_list))
train_data_list, test_data_list = train_test_split(data_list, test_size=0.2)
train_data_list, val_data_list = train_test_split(train_data_list, test_size=0.25)

train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data_list, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=True)

print(f'train_loader: {len(train_loader.dataset)}, val_loader: {len(val_loader.dataset)}, test_loader: {len(test_loader.dataset)} ')
# loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)


input_dim = 3  # server, payload, send rate
hidden_dim = 256
output_dim = 2  # TPS, avg_lat

if model_name == 'GCN':
    model = GCNRegression(input_dim, hidden_dim, output_dim)
elif model_name == 'GAT':
    model = GATRegression(input_dim, hidden_dim, output_dim)
elif model_name == 'GIN':
    model = GINRegression(input_dim, hidden_dim, output_dim)
elif model_name == 'GraphSAGE':
    model = GraphSAGERegression(input_dim, hidden_dim, output_dim)
elif model_name == 'Edge':
    model = EdgeRegression(input_dim, hidden_dim, output_dim)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# early stopping
best_loss = float('inf')
patience = 8
patience_count = 0

for epoch in tqdm(range(50)):
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        # print('output: ', output, 'data.y: ', data.y)
        # batch size가 1일 경우에는
        # hidden_dim이 64일 때, (157,2) (64) -> batch size가 32이고 결과값이 2개기 때문에 64개가 나옴
        if batch_size == 1:
            loss = criterion(output[0], data.y) # Node 1의 TPS prediction
        elif batch_size > 1:
            loss = criterion(output[::8].reshape(-1), data.y)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Training Loss: {loss.item()}')
    model.eval()
    val_loss = 0.0

    for data in val_loader:
        output = model(data)
        if batch_size == 1:
            val_loss += criterion(output[0], data.y).item()
        elif batch_size > 1:
            val_loss += criterion(output[::8].reshape(-1), data.y).item()
    val_loss /= len(val_loader)

    print(f'Validation Loss: {val_loss}')

    if val_loss < best_loss:
        best_loss = val_loss
        # torch.save(model.state_dict(), f'model/best_{model_name}.pth')
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

print("Training Complete")


model.eval()

# Initialize lists to store predictions and ground truth values
predictions_tps = []
predictions_latency = []
ground_truth_tps = []
ground_truth_latency = []

# Iterate through the test data loader with tqdm
with torch.no_grad():
    for data in tqdm(test_loader, desc='Testing'):
        output = model(data)
        # print('output: ', output, 'data.y: ', data.y)
        # predictions_tps.extend(output[:, 0].tolist())
        # predictions_latency.extend(output[:, 1].tolist())
        if batch_size == 1:
            predictions_tps.append(output[0][0])
            predictions_latency.append(output[0][1])
            ground_truth_tps.append(data.y[0])
            ground_truth_latency.append(data.y[1])
        elif batch_size > 1:
            predictions_tps.extend(output[::8][:, 0].tolist())
            predictions_latency.extend(output[::8][:, 1].tolist())
            ground_truth_tps.extend(data.y[::2].tolist())
            ground_truth_latency.extend(data.y[1::2].tolist())

# Calculate evaluation metrics
predictions_tps = np.array(predictions_tps)
predictions_latency = np.array(predictions_latency)
ground_truth_tps = np.array(ground_truth_tps)
ground_truth_latency = np.array(ground_truth_latency)

mse_tps = np.mean((predictions_tps - ground_truth_tps) ** 2)
mse_latency = np.mean((predictions_latency - ground_truth_latency) ** 2)

mape_tps = calculate_mape(predictions_tps, ground_truth_tps)
mape_latency = calculate_mape(predictions_latency, ground_truth_latency)

mae_tps = calculate_mae(predictions_tps, ground_truth_tps)
mae_latency = calculate_mae(predictions_latency, ground_truth_latency)

print("Testing Complete")
with open('result.txt', 'a') as f:
    f.write(f'Model name: {model_name}\n')
    # f.write(batch_size)
    f.write(f'Mean Squared Error (TPS): {mse_tps}\n')
    f.write(f'Mean Squared Error (Latency): {mse_latency}\n')
    f.write(f'Mean Absolute Error (TPS): {mae_tps}\n')
    f.write(f'Mean Absolute Error (Latency): {mae_latency}\n')
    f.write(f'Mean Absolute Percentage Error (TPS): {mape_tps}\n')
    f.write(f'Mean Absolute Percentage Error (Latency): {mape_latency}\n')

print(f'Mean Squared Error (TPS): {mse_tps}')
print(f'Mean Squared Error (Latency): {mse_latency}')
print("MAE (TPS):", mae_tps)
print("MAE (Latency):", mae_latency)
print("MAPE (TPS):", mape_tps)
print("MAPE (Latency):", mape_latency)
