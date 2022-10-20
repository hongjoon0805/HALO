import torch as th
import dgl
from dgl import function as fn
import numpy as np

def load_data_KG(self, g):
    category = list(g.ndata['label'].keys())[0]

    # Make features for all node types
    if self.C.data == 'AM':
        # It is hard to assign one-hot vector when data == 'AM'
        for idx, ntype in enumerate(g.ntypes):
            if 'NodeFeat' in self.C.date:
                feat = th.zeros((g.num_nodes(ntype), len(g.ntypes)))
                feat[:,idx] = 1
                g.ndata['h'] = {ntype : feat}
            if 'BinFeat' in self.C.date:
                bin_rep = np.binary_repr(g.num_nodes(ntype))
                feat = th.zeros((g.num_nodes(ntype), len(bin_rep)))
                for i in range(g.num_nodes(ntype)):
                    bin_rep_i = np.binary_repr(i, width = len(bin_rep))
                    feat[i] = th.from_numpy(np.array(list(bin_rep_i), dtype=int))
                g.ndata['h'] = {ntype : feat}
            
    else:

        for ntype in g.ntypes:
            # one-hot vector
            g.ndata['h'] = {ntype : th.eye(g.num_nodes(ntype))}
    
    # Split validation set
    # =================================================================
    tr_num = g.ndata['train_mask'][category].sum().item()
    val_num = int(tr_num * 0.15)
    tr_idx = th.where(g.ndata['train_mask'][category] == True)[0]
    val_list = np.random.choice(tr_idx, val_num, replace=False)
    val_list.sort()
    val_nodes = th.tensor(val_list)
    g.ndata['train_mask'][category][val_nodes] = 0
    # =================================================================
    
    # Training & test nodes
    g.ndata['feature'] = g.ndata['h']
    labels = g.ndata['label'][category].view(-1)
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