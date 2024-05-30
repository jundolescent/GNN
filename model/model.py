import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import EdgeConv
import torch.nn as nn


class EdgeRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EdgeRegression, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.fc = nn.Linear(output_dim * 4, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        print(x.shape, edge_index.shape)
        x = F.relu(self.conv3(x, edge_index))
        x = self.fc(x)
        return F.relu(x)


class GCNRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNRegression, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # print(x.shape)  # torch.Size([8, 2])
        # x = F.dropout(x, training=self.training)
        x = self.fc(x)
        x = F.leaky_relu(x)
        return x


class GraphSAGERegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGERegression, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # x = F.dropout(x, training=self.training)
        x = self.fc(x)
        x = F.leaky_relu(x)
        return x


class GATRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GATRegression, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4)
        self.conv2 = GATConv(hidden_dim * 4, output_dim, heads=4)
        self.fc = nn.Linear(output_dim * 4, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # x = F.dropout(x, training=self.training)
        x = self.fc(x)
        x = F.leaky_relu(x)
        return x


class GINRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GINRegression, self).__init__()
        self.conv1 = GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)))
        self.fc = nn.Linear(output_dim, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # x = F.dropout(x, training=self.training)
        x = self.fc(x)
        x = F.leaky_relu(x)
        return x