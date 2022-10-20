import torch as th
from sklearn.metrics import f1_score
import torch.nn as nn
from utils.ignn_utils import Evaluation
import pdb

@th.no_grad()
def get_eval_result(self, labels, pred_l, loss):

    if self.C.multilabel:
        micro , macro = Evaluation(pred_l , labels)
    else:
        micro = f1_score(labels.cpu(), pred_l.cpu(), average = "micro")
        macro = 0

    return {
        "micro": round(micro * 100 , 2) , # to percentage
        "macro": round(macro * 100 , 2)
    }

@th.no_grad()
def evaluate(self, g, all_labels, nodes_list, model, loss_func):
    
    model = model.eval()
    return_pred_list = []

    if self.C.multicategory:
        all_output = {}
        all_nodes = {}
        for ntype in g.ntypes:
            all_nodes[ntype] = []
            for nodes in nodes_list:
                all_nodes[ntype].append(nodes[ntype])
            
            all_nodes[ntype] = th.cat(all_nodes[ntype])
        
        output_dict = model(g, evaluate = True)
        
        results = []
        idx_from = {}
        idx_to = {}

        for ntype in g.ntypes:
            idx_from[ntype] = 0
            idx_to[ntype] = 0

        for nodes in nodes_list:
            pred_list = []
            labels_list = []
            loss_sum = 0
            for ntype in g.ntypes:
                idx_to[ntype] = idx_from[ntype] + len(nodes[ntype])
                output = output_dict[ntype][all_nodes[ntype]][idx_from[ntype]:idx_to[ntype]]
                idx_from[ntype] = idx_to[ntype]
                labels = all_labels[ntype][nodes[ntype]]
                
                loss = loss_func(output, labels)
                pred = output.argmax(-1)

                pred_list.append(pred)
                labels_list.append(labels)
                loss_sum += loss.item()

                # res = get_eval_result(self, labels, pred, loss_sum)
                # print(ntype, res['micro'])

            pred = th.cat(pred_list)
            labels = th.cat(labels_list)
            loss_sum /= len(g.ntypes)

            results.append(get_eval_result(self, labels, pred, loss_sum))
            return_pred_list.append(pred.cpu().numpy())

    else:
        all_nodes = th.cat(nodes_list)

        tr_ntype_list = list(g.ndata['label'].keys())
        # Give assert when len(tr_ntype_list) > 1
        tr_ntype = tr_ntype_list[0]

        all_output = model(g, evaluate = True)[tr_ntype][all_nodes]
        idx_from = 0
        results = []
        for nodes in nodes_list:
            idx_to = idx_from + len(nodes)
            output = all_output[idx_from:idx_to]
            idx_from = idx_to
            labels = all_labels[nodes]

            if self.C.multilabel: #multilabel
                loss = nn.BCEWithLogitsLoss()(output , labels)
                pred_l = output
            else:
                loss = loss_func(output, labels)
                pred_l = output.argmax(-1)

            results.append(get_eval_result(self, labels, pred_l, loss.item()))
            return_pred_list.append(pred_l.cpu().numpy())

    return results, return_pred_list

