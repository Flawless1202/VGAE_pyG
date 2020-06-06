import os

import torch
import torch.nn as nn
from torch.optim import Adam

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges

from model import DeepVGAE
from config.config import parse_args

torch.manual_seed(12345)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = parse_args()

model = DeepVGAE(args).to(device)
optimizer = Adam(model.parameters(), lr=args.lr)

os.makedirs("datasets", exist_ok=True)
dataset = Planetoid("datasets", args.dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)
all_edge_index = data.edge_index
data = train_test_split_edges(data, 0.05, 0.1)

for epoch in range(args.epoch):
    model.train()
    optimizer.zero_grad()
    loss = model.loss(data.x, data.train_pos_edge_index, all_edge_index)
    loss.backward()
    optimizer.step()
    if epoch % 2 == 0:
        model.eval()
        roc_auc, ap = model.single_test(data.x,
                                        data.train_pos_edge_index,
                                        data.test_pos_edge_index,
                                        data.test_neg_edge_index)
        print("Epoch {} - Loss: {} ROC_AUC: {} Precision: {}".format(epoch, loss.cpu().item(), roc_auc, ap))
