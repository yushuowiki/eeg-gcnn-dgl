'''
Created on 2022年1月12日

@author: sunke
'''
import torch.nn as nn
import torch.nn.functional as function
from dgl.nn import GraphConv, SumPooling, GlobalAttentionPooling
import torch
import torch.nn.functional as F
from dgl.nn import GCN2Conv

# https://docs.dgl.ai/api/python/nn-pytorch.html

class GCN2Net(nn.Module):
    """ EEGGraph Convolution Net
        Parameters
        ----------
        num_feats: the number of features per node. In our case, it is 6.
    """
    def __init__(self, num_feats):
        super(GCN2Net, self).__init__()
#         self.conv1 = SGConv(num_feats, 32,k=2)   #在linux 下不能使用 max
#         self.conv2 = SGConv(32, 20,k=2)
        
        self.conv1 = GCN2Conv(num_feats, layer=1, alpha=0.5, \
                         project_initial_features=True, allow_zero_in_degree=True)
        self.conv2 = GCN2Conv(num_feats, layer=2, alpha=0.5, \
                         project_initial_features=True, allow_zero_in_degree=True)
        
        
        self.conv2_bn = nn.BatchNorm1d(num_feats, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc_block1 = nn.Linear(num_feats, 10)
        self.fc_block2 = nn.Linear(10, 2)
        self.chanDim = 1
        self.chanNum = 6
        # -------------------------------------------
        self.pool = SumPooling()
#          ------------------------------------------------
        # Xavier initializations
        self.fc_block1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1))
        self.fc_block2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1))
        

    def forward(self, g, return_graph_embedding=False):
        x = g.ndata['x']
        edge_weight = g.edata['edge_weights']      
#         ---------------------------------------
        x = function.leaky_relu(self.conv1(g, x))
        
        x = self.conv2(g, x)
        
        x = function.leaky_relu(self.conv2_bn(x))

        # NOTE: this takes node-level features/"embeddings"
        # and aggregates to graph-level - use for graph-level classification
#         -----------------------------------------
        out = self.pool(g, x)
#         ------------------------------------------ 
        if return_graph_embedding:
            return out

        out = function.dropout(out, p=0.2, training=self.training)
        out = self.fc_block1(out)
        out = function.leaky_relu(out)
        out = self.fc_block2(out)

        return out
