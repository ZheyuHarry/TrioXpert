import torch
import torch.nn as nn
import pickle
import dgl
from torch.utils.data import Dataset, DataLoader
from utils.public_function import load_pkl
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    def __init__(self, in_dim, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()
        
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,  
            nhead=num_heads,  
            dim_feedforward=in_dim * 8,  
            batch_first=True  
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers 
        )

    def forward(self, features):
        h = F.leaky_relu(self.transformer_encoder(features))  
        return h


class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GRUEncoder, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,  
            hidden_size=hidden_dim,  
            num_layers=num_layers, 
            batch_first=True  
        )

    def forward(self, features):
        output, h_n = self.gru(features)  
        return output, h_n

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, norm):
        super(GraphSAGEEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        hidden_dim = hidden_dim if num_layers > 1 else out_dim
        self.input_conv = dgl.nn.GraphConv(in_dim, hidden_dim, norm=norm)
        self.convs = nn.ModuleList()  
        for _ in range(num_layers - 2):
            self.convs.append(dgl.nn.GraphConv(hidden_dim, hidden_dim, norm=norm))
        if num_layers > 1:
            self.convs.append(dgl.nn.GraphConv(hidden_dim, out_dim, norm=norm))

    def forward(self, g, features):
        h = F.leaky_relu(self.input_conv(g, features))
        h = self.dropout(h)
        for conv in self.convs:
            h = F.leaky_relu(conv(g, h))
            h = self.dropout(h)
        return h
    
    def transform(self, g, features):
        h = F.leaky_relu(self.input_conv(g, features))
        for conv in self.convs:
            h = F.leaky_relu(conv(g, h))
        return h


class AutoEncoder(nn.Module):
    def __init__(self, num_nodes=18, num_features=613, tf_hidden_dim=512, gru_hidden_dim=256, num_heads=8, tf_layers=2, gru_layers=2, gnn_hidden_dim = 64, gnn_out_dim = 32, gnn_layers=2, dropout=0):
        super(AutoEncoder, self).__init__()

        self.transformer_encoder = TransformerEncoder(
            in_dim=num_nodes,  
            num_heads=num_heads,  
            num_layers=tf_layers 
        )

        
        self.gru_encoder = GRUEncoder(
            input_dim=num_features,  
            hidden_dim=gru_hidden_dim,  
            num_layers=gru_layers  
        )
        self.graph_encoder = GraphSAGEEncoder(gru_hidden_dim, gnn_hidden_dim, gnn_out_dim, gnn_layers, dropout, norm='none')

        self.decoder = nn.Linear(gnn_out_dim, num_features)

    def forward(self, g, x):
        batch_size, series_len, instance_num, channel_dim = x.shape 
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, channel_dim, instance_num)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)
        x = x.view(batch_size, series_len, instance_num, channel_dim)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(-1, series_len, channel_dim)
        _, h_n = self.gru_encoder(x)
        h = h_n[-1]
        h = F.leaky_relu(h)
        device = g.device
        h = self.graph_encoder(g, h)
        h = h.view(batch_size, instance_num, -1)
        output = F.leaky_relu(self.decoder(h))

        return output

    



# Creating a DataLoader for auto_regressor.
def collate_AR(samples):
    """
        Used to form the batched data
    """
    timestamps, graphs, feats, targets = map(list, zip(*samples))
    batched_ts = torch.stack(timestamps)
    batched_graphs = dgl.batch(graphs)
    batched_feats = torch.stack(feats)
    batched_targets = torch.stack(targets)
    return  batched_ts, batched_graphs, batched_feats, batched_targets

def create_dataloader_AR(samples, window_size=10, max_gap=60, batch_size=2, shuffle=False):
    # sliding_time_windows
    samples = sorted(samples , key=lambda x: x[0])
    series_samples = [samples[i:i+window_size] for i in range(len(samples) - window_size + 1)] # How many windows here in samples
    series_samples = [
        series_sample for series_sample in series_samples
            if all(abs(series_sample[i][0] - series_sample[i+1][0]) <= max_gap  # Make sure the timestamp difference between adjcent samples is less than the max_gap=60
                for i in range(len(series_sample) - 1))
    ]
    
    """
        Every series_sample is a window
    """
    # create a dataloader
    dataset = [[
            torch.tensor(series_sample[-1][0]), # Last timestamp in a window
            series_sample[-1][1], # Last graph in a window
            torch.stack([torch.tensor(step[2]) for step in series_sample[:-1]]), # Instances' features except the last
            torch.tensor(series_sample[-1][2]) # Last instance's features
        ] for _, series_sample in enumerate(series_samples)]
    """
        dataset is a list of [timestamp, graph, features, target]
        we use the previous features to predict the target features
    """
    dataloader = DataLoader(dataset, batch_size, shuffle, collate_fn=collate_AR)
    return dataloader

# end Collate Function  ------------------------------------------
