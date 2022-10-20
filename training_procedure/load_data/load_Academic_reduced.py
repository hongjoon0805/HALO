import torch as th
import dgl
from dgl import function as fn
import numpy as np
from sklearn.utils import shuffle

def load_data_Academic_reduced(self, g):
    """
    From:
    ICML, AAAI, IJCAI, CVPR, ICCV, ECCV, ACL, EMNLP, NAACL, KDD, WSDM, ICDM, SIGMOD, VLDB, ICDE, WWW, SIGIR, CIKM
    To:
    (Top)
    "ML", "Vision", "NLP", "Data"
    """
    
    venue_to_label = th.tensor([3,0,1,1,0,3,2,3,3,2,3,3,3,2,3,3,1,0], dtype=th.long)

    # Construct new bipartite graph
    # =================================================================
    g_data = {}
    label_data = {}
    train_mask_data = {}
    test_mask_data = {}
    h_data = {}

    g_data[('author', 'author-paper', 'paper')] = g.edges(etype=('author', 'author-paper', 'paper'))
    g_data[('paper', 'paper-author', 'author')] = g.edges(etype=('paper', 'paper-author', 'author'))
    g_data[('paper', 'cite', 'paper')] = g.edges(etype=('paper', 'cite', 'paper'))

    u,v = g.edges(etype = ('paper', 'paper-venue', 'venue'))
    label = th.zeros(len(u), dtype=th.long)
    label[u]=venue_to_label[v]
    # label[u]=v


    label_data['author'] = g.ndata['label']['author'].type(th.LongTensor)
    label_data['paper'] = label

    tr_num = 6300

    all_mask = th.ones(len(u), dtype=th.uint8)
    train_mask = th.zeros(len(u), dtype=th.uint8)
    test_mask = th.zeros(len(u), dtype=th.uint8)

    train_idx = shuffle(np.arange(len(u)), random_state=0)[:tr_num]

    train_mask[train_idx] = 1
    test_mask = all_mask ^ train_mask

    train_mask_data['author'] = g.ndata['train_mask']['author']
    test_mask_data['author'] = g.ndata['test_mask']['author']

    train_mask_data['paper'] = train_mask
    test_mask_data['paper'] = test_mask

    h_data['author'] = g.ndata['dw_embedding']['author']
    h_data['paper'] = g.ndata['dw_embedding']['paper']

    new_g = dgl.heterograph(g_data)
    new_g.ndata['label'] = label_data
    new_g.ndata['train_mask'] = train_mask_data
    new_g.ndata['test_mask'] = test_mask_data
    new_g.ndata['h'] = h_data

    g = new_g
    # =================================================================

    # Split Training & Validation & Test set
    # =================================================================
    train_nodes = {}
    val_nodes = {}
    test_nodes = {}

    val_num = {}
    val_num['author'] = 400
    val_num['paper'] = 1500
    for ntype in g.ndata['label'].keys():

        tr_num = g.ndata['train_mask'][ntype].sum().item()
        tr_idx = th.where(g.ndata['train_mask'][ntype] == True)[0]
        val_list = np.random.choice(tr_idx, val_num[ntype], replace=False)
        val_list.sort()
        val_nodes[ntype] = th.tensor(val_list)
        g.ndata['train_mask'][ntype][val_nodes[ntype]] = 0
        
        train_nodes[ntype] = th.where(g.ndata['train_mask'][ntype] == True)[0]
        test_nodes[ntype] = th.where(g.ndata['test_mask'][ntype] == True)[0]

    # =================================================================

    labels = {}
    labels['author'] = g.ndata['label']['author'].view(-1)
    labels['paper'] = g.ndata['label']['paper'].view(-1)

    g.ndata['feature'] = g.ndata['h']

    return g, labels, train_nodes, val_nodes, test_nodes