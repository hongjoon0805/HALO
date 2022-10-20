import torch as th
import dgl
from dgl import function as fn
import numpy as np

def load_data_Freebase(self, g):
    category = list(g.ndata['label'].keys())[0]

    # Make new graph to gaurantee the graph as bidirectional
    g_data = {}
    for etype in g.canonical_etypes:
        src, t, dst = etype
        u, v = g.edges(etype=etype)
        g_data[etype] = (u,v)
        if src != dst:
            etype_inv = (dst, t + '-inv', src)
            g_data[etype_inv] = (v,u)

    new_g = dgl.heterograph(g_data)
    for key in g.ndata.keys():
        new_g.ndata[key] = g.ndata[key]
    for key in g.edata.keys():
        new_g.edata[key] = g.edata[key]

    # Make features for all node type
    # =================================================================
    for idx, ntype in enumerate(new_g.ntypes):
        # Dummy vector
        if 'RandFeat' in self.C.date or 'OnehotFeat' in self.C.date:
            new_g.ndata['h'] = {ntype : th.rand(new_g.num_nodes(ntype), 64)}
        # one-hot vector for each node type
        if 'NodeFeat' in self.C.date:
            feat = th.zeros((new_g.num_nodes(ntype), len(new_g.ntypes)))
            feat[:,idx] = 1
            new_g.ndata['h'] = {ntype : feat}
        if 'BinFeat' in self.C.date:
            bin_rep = np.binary_repr(new_g.num_nodes(ntype))
            feat = th.zeros((new_g.num_nodes(ntype), len(bin_rep)))
            for i in range(g.num_nodes(ntype)):
                bin_rep_i = np.binary_repr(i, width = len(bin_rep))
                feat[i] = th.from_numpy(np.array(list(bin_rep_i), dtype=int))
            new_g.ndata['h'] = {ntype : feat}
    # =================================================================
    
    # Split validation set
    # =================================================================
    tr_num = new_g.ndata['train_mask'][category].sum().item()
    val_num = 600
    tr_idx = th.where(new_g.ndata['train_mask'][category] == True)[0]
    val_list = np.random.choice(tr_idx, val_num, replace=False)
    val_list.sort()
    val_nodes = th.tensor(val_list)
    new_g.ndata['train_mask'][category][val_nodes] = 0
    # =================================================================

    # Training & test nodes
    new_g.ndata['feature'] = new_g.ndata['h']
    labels = new_g.ndata['label'][category]
    train_nodes = th.where(new_g.ndata['train_mask'][category] == True)[0]
    test_nodes = th.where(new_g.ndata['test_mask'][category] == True)[0]

    
    # Make graph as bidirectional
    for etype in new_g.canonical_etypes:
        src, t, dst = etype
        if src != dst:
            continue
        u, v = new_g.edges(etype=etype)
        new_g.add_edges(v, u, etype=etype)
        


    return new_g, labels, train_nodes, val_nodes, test_nodes