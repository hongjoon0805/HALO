import pdb

import torch as th
import torch.nn as nn


def train(self, g, labels, nodes, model, loss_func, optimizer):

    model = model.train()
    if self.C.multicategory:
        loss = 0
        for ntype in g.ntypes:
            output = model(g)[ntype][nodes[ntype]]
            label = labels[ntype][nodes[ntype]]

            loss += loss_func(output, label)
    else:

        tr_ntype_list = list(g.ndata['label'].keys())
        # Give assert when len(tr_ntype_list) > 1
        tr_ntype = tr_ntype_list[0]

        output = model(g)[tr_ntype][nodes]
        label  = labels[nodes]

        if self.C.multilabel: # use BCEWithLogitsLoss in multilabel setting
            loss = nn.BCEWithLogitsLoss()(output , label)
        else:
            loss = loss_func(output, label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return model, float(loss)
