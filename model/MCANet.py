import torch.nn as nn
import torch.nn.functional as function
from dgl.nn import GraphConv, SumPooling, GlobalAttentionPooling
import torch
import torch.nn.functional as F
import dgl.nn as dglnn


class MCANet(nn.Module):
    """ EEGGraph Convolution Net
        Parameters
        ----------
        num_feats: the number of features per node. In our case, it is 6.
    """
    def __init__(self, num_feats):
        super(MCANet, self).__init__()

        # self.conv1 = GraphConv(num_feats, 32)
        # self.conv2 = GraphConv(32, 20)
        self.conv1 = dglnn.SAGEConv(num_feats, 32,'lstm')
        self.conv2 = dglnn.SAGEConv(32, 20,'lstm')
        self.conv2_bn = nn.BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc_block1 = nn.Linear(20, 10)
        self.fc_block2 = nn.Linear(10, 2)
        self.chanDim = 1
        self.chanNum = 6
        # -------------------------------------------
        self.wfc = nn.Linear(2, 100)
        self.wfc1 = nn.Linear(100, 1)
#         self.hfc = nn.Linear(self.chanNum*self.chanDim,self.chanDim)   #1是每个通道的维度
        self.hfc = nn.Linear(self.chanNum*self.chanDim,self.chanNum*self.chanDim)
        self.attn_fc = nn.Linear(self.chanDim+self.chanNum, 1, bias=False)  #通道维度的二倍，输出attention数值
        self.attn_fc2 = nn.Linear(self.chanNum*2, 1, bias=False)
        
        gate_nn = nn.Linear(20, 1)
        
#         self.pool = SumPooling()
        self.pool = GlobalAttentionPooling(gate_nn)
#          ------------------------------------------------
        # Xavier initializations
        self.fc_block1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1))
        self.fc_block2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1))
        


    def get_dweight(self,g):
        dis = g.edata['dis'].reshape(-1,1)
        spec = g.edata['spec'].reshape(-1,1)
        dweight = torch.cat((dis,spec),1)
        edge_weight = self.wfc(dweight)
        edge_weight = self.wfc1(edge_weight).flatten()
        return edge_weight
    
    
    def chan_att(self,x):
        h = x.reshape(x.shape[0],-1)
        z =  self.hfc(h)
        h_z = z.repeat(1,self.chanNum)
        h_z = h_z.reshape(x.shape[0],x.shape[1],-1)
        #---------------------------------------------
        h_z2 = torch.cat((h_z,z.reshape(x.shape[0],x.shape[1],-1)),2)
        h_z2 = h_z2.reshape(-1,h_z2.shape[-1])
        a = self.attn_fc(h_z2)
        a = F.leaky_relu(a)
        a = a.reshape(x.shape[0],-1)
        #softmax...
        alpha = F.softmax(a,dim=1)
        x = alpha * z
        return x
    
    
    def chan_att2(self,x):
        h = x.reshape(x.shape[0],-1)
        z =  self.hfc(h)
        h_z = z.repeat(1,self.chanNum)
        h_z = h_z.reshape(x.shape[0],x.shape[1],-1)
        #---------------------------------------------
        z_r = z.reshape(x.shape[0],x.shape[1],-1)
        z_r = z_r.repeat(1,1,self.chanNum)
        h_z2 = torch.cat((h_z,z_r),2)        
        h_z2 = h_z2.reshape(-1,h_z2.shape[-1])
        a = self.attn_fc2(h_z2)
        a = F.leaky_relu(a)
        a = a.reshape(x.shape[0],-1)
        #softmax.
        alpha = F.softmax(a,dim=1)
        x = alpha * z
        return x
  
    
    def forward(self, g, return_graph_embedding=False):
        x = g.ndata['x']
#         -----------------------------------------
#         edge_weight = g.edata['edge_weights']      
#       ---------------Channel attention-----------------------------
#         x = self.chan_att(x)
        x = self.chan_att2(x)
        edge_weight = self.get_dweight(g)
#         ---------------------------------------
        x = function.leaky_relu(self.conv1(g, x, edge_weight=edge_weight))
        x = function.leaky_relu(self.conv2_bn(self.conv2(g, x, edge_weight=edge_weight)))

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
