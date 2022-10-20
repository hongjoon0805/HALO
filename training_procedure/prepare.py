from ogb.nodeproppred import DglNodePropPredDataset
import torch as th
import torch.nn as nn
import os
import random
import dgl
from functools import reduce
from tqdm import tqdm
from model import GNNModel
from .load_data import load_data
import pdb

def prepare_train(self , model):
    '''prepare other stuffs for training'''

    C = self.C

    optimizer = th.optim.Adam(
        params          = model.parameters() , 
        lr              = C.lr , 
        weight_decay    = C.weight_decay , 
    )
    loss_func = nn.CrossEntropyLoss(ignore_index = -100)

    return optimizer , loss_func

def prepare_model(self , label_numb , input_size , g , etypes_inv, category_dim_dict = None):
    '''create model'''

    C = self.C
    num_dims_dict = {}
    if (self.C.data == 'AM' or self.C.data == 'Freebase') and 'OnehotFeat' in self.C.date:
        
        for ntype in g.ntypes:
            num_dims_dict[ntype] = g.num_nodes(ntype)
    else:
        for ntype in g.ntypes:
            num_dims_dict[ntype] = g.ndata['h'][ntype].size(-1)



    model = GNNModel(
        C               =  C,
        g               =  g,
        input_d         =  input_size , 
        output_d        =  label_numb , 
        num_dims_dict   =  num_dims_dict,
        etypes_inv      =  etypes_inv,
        category_dim_dict = category_dim_dict
    )

    return model

def init(self, idx, device = 0):
    C = self.C
    logger = self.logger

    g , labels , etypes_inv ,  train_nodes , val_nodes , test_nodes = load_data(self, idx)

    # ------------------- prepare model -------------------
    category_dim_dict = None

    if C.multilabel:
        label_numb = int(labels.size(-1))
    else:
        if C.multicategory:
            label_num_max = 0
            category_dim_dict = {}
            for category in g.ndata['label'].keys():
                label_numb = max(int(max(labels[category])) + 1, label_num_max)
                category_dim_dict[category] = int(max(labels[category])) + 1

        else:
            label_numb = int(max(labels)) + 1
    
    
    input_size = 0
    try:
        category = list(g.ndata['label'].keys())[0]
        input_size = g.ndata['feature'][category].size(-1)
    except:
        pass

    model = self.prepare_model(label_numb , input_size , g, etypes_inv, category_dim_dict)

    logger.log("number of params: %d" % sum( [int(x.view(-1).size(0)) for x in model.parameters()] ))

    optimizer , loss_func = self.prepare_train(model)

    if C.multilabel:
        labels  = labels.type(th.FloatTensor)
    else:
        if C.multicategory:
            for ntype in g.ntypes:
                labels[ntype]  = labels[ntype].type(th.LongTensor)
        else:
            labels  = labels.type(th.LongTensor)
    
    # ------------------- move evrything to gpu -------------------
    if C.multicategory:
        g           = g          .to(device)
        model       = model      .to(device)
        for ntype in g.ntypes:
            labels[ntype]      = labels[ntype]     .to(device)
            train_nodes[ntype] = train_nodes[ntype].to(device)
            val_nodes[ntype]   = val_nodes[ntype]  .to(device)
            test_nodes[ntype]  = test_nodes[ntype] .to(device)
    else:
        g           = g          .to(device)
        labels      = labels     .to(device)
        train_nodes = train_nodes.to(device)
        val_nodes   = val_nodes  .to(device)
        test_nodes  = test_nodes .to(device)
        model       = model      .to(device)
    return (g , labels , etypes_inv) , (train_nodes , val_nodes , test_nodes) , model , (optimizer , loss_func)