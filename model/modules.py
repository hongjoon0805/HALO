import pdb
import random
from functools import partial
import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from torch.nn import init
from .submodules import Propagate, ZooBP_Propagate
from tqdm import tqdm
import pickle 
import copy

class HeteroUnfolding(nn.Module):
    def __init__(self, d, C, ntypes, canonical_etypes, etypes_inv, category_dim_dict = None):

        super().__init__()

        self.d      = d
        self.C      = C
        self.alp    = C.alp if C.alp > 0 else 1 / (C.lam + 1) # automatic set alpha
        self.lam    = C.lam
        self.prop_step = C.prop_step
        self.train_H = not C.no_train_H
        self.ntypes = ntypes
        self.canonical_etypes = canonical_etypes
        self.etypes_inv = etypes_inv
        self.category_dim_dict = category_dim_dict

        self.prop_layers = nn.ModuleList([Propagate() for _ in range(self.prop_step)])

        H = {}

        for etype in self.canonical_etypes:
            src, rel, dst = etype
            t = src + '-' + rel + '-' + dst
            
            if self.C.residual: # H_t = I + R_t
                H[t] = th.rand(self.d, self.d)
                bound = 1/self.d # normal
                nn.init.normal_(H[t], 0, bound)
                H[t] = H[t] + th.eye(self.d)

            else:
                H[t] = th.rand(self.d, self.d)
                bound = 4/self.d # normal
                nn.init.normal_(H[t], 0, bound)
            

            if self.C.multicategory:
                src_dim = self.category_dim_dict[src]
                dst_dim = self.category_dim_dict[dst]

                H[t][src_dim:] = 0
                H[t][: , dst_dim:] = 0
            
            H[t] = nn.Parameter(H[t])
                
        self.H = nn.ParameterDict(H)

    def forward(self ,g , X, evaluate = False):
        
        Y = X

        for ntype in g.ntypes:
            g.nodes[ntype].data['deg'] = th.zeros(g.num_nodes(ntype),1, device=g.device)
            if self.C.multicategory:
                dim = self.category_dim_dict[ntype]
                Y[ntype][:,dim:] = 0

        for etype in g.canonical_etypes:
            src, rel, dst = etype
            t = src + '-' + rel + '-' + dst
            g.edges[etype].data['w'] = th.ones(g.num_edges(etype), 1, device=g.device)
            g.nodes[src].data['deg'] += g.out_degrees(etype=etype).unsqueeze(1)

            if self.C.multicategory:
                src, rel, dst = etype
                src_dim = self.category_dim_dict[src]
                dst_dim = self.category_dim_dict[dst]

                self.H[t].data[src_dim:] = 0
                self.H[t].data[: , dst_dim:] = 0

            self.H[t] = self.H[t].to(g.device)
                
                

        for k, layer in enumerate(self.prop_layers):

            # do unfolding
            Y = layer(g, self.C, Y, X, self.H, self.alp, self.lam, self.etypes_inv)

        return Y     

        

class ZooBP(nn.Module):
    def __init__(self, d, C, ntypes, canonical_etypes, etypes_inv, category_dim_dict = None):

        super().__init__()

        self.d      = d
        self.C      = C
        self.eps    = C.eps
        self.prop_step = C.prop_step
        self.ntypes = ntypes
        self.canonical_etypes = canonical_etypes
        self.etypes_inv = etypes_inv
        self.category_dim_dict = category_dim_dict

        # For Academic dataset
        author_paper_mat = {}
        author_paper_mat['Academic_reduced'] = th.tensor([
            [ -1,  9, -3, -3,  9, -1, -3, -1, -1, -3, -1, -1, -1, -3, -1, -1, -3,  9,],
            [ -1, -3,  9,  9, -3, -1, -3, -1, -1, -3, -1, -1, -1, -3, -1, -1,  9, -3,],
            [ -1, -3, -3, -3, -3, -1,  9, -1, -1,  9, -1, -1, -1,  9, -1, -1, -3, -3,],
            [  3, -3, -3, -3, -3,  3, -3,  3,  3, -3,  3,  3,  3, -3,  3,  3, -3, -3,],
            ])
        
        # For DBLP bipartite
        author_paper_mat['DBLP_reduced'] = th.tensor([
            [ -1, -1, -1, -1, -1, -1,  3, -1, -1,  3, -1, -1, -1,  3, -1, -1,  3,  3, -1, -1,],
            [ -1, -1, -1, -1, -1, -1, -1,  3, -1, -1,  3,  3,  3, -1,  3, -1, -1, -1, -1, -1,],
            [  3, -1,  3, -1,  3,  3, -1, -1,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,],
            [ -1,  3, -1,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  3, -1, -1,  3,  3,],
            ])

        self.prop_layers = nn.ModuleList([ZooBP_Propagate() for _ in range(self.prop_step)])
        self.H = {}
        for etype in self.canonical_etypes:
            src, rel, dst = etype
            t = src + '-' + rel + '-' + dst

            src_dim = self.category_dim_dict[src]
            dst_dim = self.category_dim_dict[dst]

            if src_dim == dst_dim:
                self.H[t] = (th.eye(self.d) - (1/self.d))
            else:
                self.H[t] = th.zeros(self.d, self.d)
                if src == 'author':
                    self.H[t][:src_dim] = author_paper_mat[self.C.data]
                elif src == 'paper':
                    self.H[t][:,:dst_dim] = author_paper_mat[self.C.data].T
        
    def forward(self ,g , X):
        # Set initial belief
        X = {}
        Y = {}

        for etype in g.canonical_etypes:
            src, rel, dst = etype
            t = src + '-' + rel + '-' + dst
            self.H[t] = self.H[t].to(g.device)

        c = 0.01
        for ntype in g.ntypes:
            num_nodes = g.num_nodes(ntype=ntype)
            X[ntype] = th.zeros(num_nodes, self.d, device=g.device)
            tr_idx = th.where(g.ndata['train_mask'][ntype] == True)[0]
            tr_label = g.ndata['label'][ntype].view(-1)[tr_idx]
            dim = self.category_dim_dict[ntype]
            
            X[ntype][tr_idx] = -c
            X[ntype][tr_idx, tr_label] += dim * c
            Y[ntype] = X[ntype] - 0
            Y[ntype][:, dim:] = 0

        for etype in g.canonical_etypes:
            g.edges[etype].data['w'] = th.ones(g.num_edges(etype), 1, device=g.device)
        

        with th.no_grad():
            for k, layer in tqdm(enumerate(self.prop_layers)):
                # do message passing
                Y = layer(g, Y, X, self.H, self.d, self.eps, self.etypes_inv, self.category_dim_dict)
            
        return Y

class MLP(nn.Module):
    def __init__(self, input_d, hidden_d, output_d, num_layers, dropout, norm, init_activate) :
        super().__init__()

        self.init_activate  = init_activate
        self.norm           = norm
        self.dropout        = dropout

        self.layers = nn.ModuleList([])


        if num_layers == 1:
            self.layers.append(nn.Linear(input_d, output_d))
        elif num_layers > 1:
            self.layers.append(nn.Linear(input_d, hidden_d))
            for k in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_d, hidden_d))
            self.layers.append(nn.Linear(hidden_d, output_d))

        self.norm_cnt = num_layers-1+int(init_activate) # how many norm layers we have
        if norm == "batch":
            self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_d) for _ in range(self.norm_cnt)])
        elif norm == "layer":
            self.norms = nn.ModuleList([nn.LayerNorm  (hidden_d) for _ in range(self.norm_cnt)])


        self.reset_params()

    def reset_params(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight.data)
            nn.init.constant_     (layer.bias.data, 0)

    def activate(self, x):
        if self.norm != "none":
            x = self.norms[self.cur_norm_idx](x) # use the last norm layer
            self.cur_norm_idx += 1
        x = F.relu(x)
        x = F.dropout(x , self.dropout , training = self.training)
        return x 

    def forward(self, x):
        self.cur_norm_idx = 0

        if self.init_activate:
            x = self.activate(x)

        for i , layer in enumerate( self.layers ):
            x = layer(x)
            if i != len(self.layers) - 1: # do not activate in the last layer
                x = self.activate(x)

        return x
