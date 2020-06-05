import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.models import InnerProductDecoder, VGAE
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import negative_sampling


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.gcn_shared = GCNConv(in_channels, hidden_channels)
        self.gcn_mu = GCNConv(hidden_channels, out_channels)
        self.gcn_logvar = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn_shared(x, edge_index))
        mu = self.gcn_mu(x, edge_index)
        logvar = self.gcn_logvar(x, edge_index)
        return mu, logvar


class DeepVGAE(VGAE):
    def __init__(self, args):
        super(DeepVGAE, self).__init__(encoder=GCNEncoder(args.enc_in_channels,
                                                          args.enc_hidden_channels,
                                                          args.enc_out_channels),
                                       decoder=InnerProductDecoder())

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        adj_pred = self.decoder.forward_all(z)
        return adj_pred

    def loss(self, x, pos_edge_index):
        z = self.encode(x, pos_edge_index)
        recon_loss = self.recon_loss(z, pos_edge_index)
        kl_loss = self.kl_loss()
        return recon_loss + kl_loss

    def single_test(self, x, pos_edge_index):
        neg_edge_index = negative_sampling(pos_edge_index, x.size(0))
        z = self.encode(x, pos_edge_index)
        pos_edge_index_tmp = pos_edge_index
        roc_auc_score, average_precision_score = self.test(z, pos_edge_index_tmp, neg_edge_index)
        return roc_auc_score, average_precision_score
