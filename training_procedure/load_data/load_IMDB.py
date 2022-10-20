import torch as th
import dgl
from dgl import function as fn
import numpy as np

def load_data_IMDB(self, g):
    category = list(g.ndata['label'].keys())[0]
	
    # Make features for 'keyword' node type
    # =================================================================
    g.ndata['h'] = {'keyword': th.eye(g.num_nodes('keyword'))}
    # =================================================================

    # Split validation set
    # =================================================================
    tr_num = g.ndata['train_mask'][category].sum().item()
    val_num = 400
    tr_idx = th.where(g.ndata['train_mask'][category] == True)[0]
    val_list = np.random.choice(tr_idx, val_num, replace=False)
    val_list.sort()
    val_nodes = th.tensor(val_list)
    g.ndata['train_mask'][category][val_nodes] = 0
    # =================================================================

    # Training & test nodes
    g.ndata['feature'] = g.ndata['h']
    labels = g.ndata['label'][category]
    train_nodes = th.where(g.ndata['train_mask'][category] == True)[0]
    test_nodes = th.where(g.ndata['test_mask'][category] == True)[0]

    
    # Make graph with same node type as bidirectional
    for etype in g.canonical_etypes:
        src, t, dst = etype
        if src != dst:
            continue
        u, v = g.edges(etype=etype)
        g.add_edges(v, u, etype=etype)


    return g, labels, train_nodes, val_nodes, test_nodes