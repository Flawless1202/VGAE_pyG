import torch
import torch.nn as nn
from torch.optim import Adam

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj, negative_sampling

from model import DeepVGAE
from config.config import parse_args

args = parse_args()

dataset = Planetoid("datasets", "citeSeer")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepVGAE(args).to(device)
data = dataset[0].to(device)
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

# pos_edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]],
#                           dtype=torch.long)
# dense_adj = to_dense_adj(pos_edge_index)
# neg_edge_index = negative_sampling(pos_edge_index, 3)
# print(neg_edge_index)

# print(data.edge_index.size())
# print(data.x.size())

neg_samples = negative_sampling(data.edge_index, data.x.size(0))

for epoch in range(args.epoch):
    model.train()
    optimizer.zero_grad()
    loss = model.loss(data.x, data.edge_index)
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        model.eval()
        roc_auc_score, average_precision_score = model.single_test(data.x, data.edge_index)
        print("Epoch {} - Loss: {} ROC_AUC: {} Precision: {}".format(epoch, loss.cpu().item(), roc_auc_score, average_precision_score))