import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


def calculate_mae(predictions, targets):
    """
    Calculate Mean Absolute Error (MAE).

    Args:
        predictions (numpy.ndarray): 모델의 예측값 배열
        targets (numpy.ndarray): 실제값 배열

    Returns:
        float: MAE 값
    """
    mae = np.mean(np.abs(predictions - targets))
    return mae


def calculate_mape(predictions, targets):
    errors = np.abs(predictions - targets)
    mape = np.mean(errors / targets) * 100
    return mape





def df_to_graph(df):
    # graph initialization
    adj_matrix = []
    features = []
    target_tps = []
    target_latency = []
    for _, row in df.iterrows():
        # server,peer,combination,delay,chaincode,payload,send rate,max lat,min lat,avg lat,TPS,
        # link1,link2,link3,link4,link5,link6,link7,server1,server2,server3,server4,server5,server6,server7,server8
        # 5,6,11112,50.100.200.250,Create,100,786.2,0.82,0.05,0.27,616.1,50,100,200,250,0,0,0,1,1,1,1,2,0,0,0

        # the Number of Nodes
        n_region = int(row['server'])
        num_nodes = n_region

        adj = torch.zeros(8, 8)
        for i in range(8):
            for j in range(8):
                adj[i][j] = 0

        # adjacency
        links = [row[f'link{i}'] for i in range(1, 8)]
        for i, link in enumerate(links):
            adj[0][i+1] = link
            adj[i+1][0] = link

        adj_matrix.append(adj)

        # feature of Nodes
        servers = [row[f'server{i}'] for i in range(1, 9)]
        # chaincode = row['chaincode']
        payload = row['payload']
        send_rate = row['send rate']

        node_features = []
        for server in servers:
            node_feature = [server, payload, send_rate]
            node_features.append(node_feature)

        features.append(torch.tensor(node_features, dtype=torch.float))

        # # etc
        # combination = row['combination']
        # n_peer = row['peer']

        # result -> label of Node
        max_lat = row['max lat']
        min_lat = row['min lat']
        avg_lat = row['avg lat']
        tps = row['TPS']

        target_tps.extend([tps] * num_nodes)
        target_latency.extend([avg_lat] * num_nodes)

    adj_matrix = torch.stack(adj_matrix)
    features = torch.stack(features)
    # Normalization
    features_reshaped = features.view(-1, features.shape[-1])
    features_normalized = scaler.fit_transform(features_reshaped.numpy())
    features = torch.tensor(features_normalized.reshape(features.shape), dtype=torch.float)
    target_tps = torch.tensor(target_tps, dtype=torch.float)
    target_latency = torch.tensor(target_latency, dtype=torch.float)

    data_list = []
    for i in range(len(df)):
        edge_index = dense_to_sparse(adj_matrix[i])[0]
        data = Data(x=features[i],
                    edge_index=edge_index,
                    y=torch.tensor([target_tps[i], target_latency[i]]))
        data_list.append(data)

    return data_list