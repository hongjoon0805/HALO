import pdb
import random

import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from torch.nn import init

class Propagate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, g, C, Y, X, H, alp, lam, etypes_inv):
        
        ret = {}

        for ntype in g.ntypes:
            ret[ntype] = th.zeros_like(Y[ntype])
        
        g.srcdata['h'] = Y
        g.apply_edges(fn.u_mul_e('h', 'w', 'm'))

        for etype in g.canonical_etypes:
            etype_inv = etypes_inv[etype]

            src, rel, dst = etype
            t = src + '-' + rel + '-' + dst

            src_inv, rel_inv, dst_inv = etype_inv
            t_inv = src_inv + '-' + rel_inv + '-' + dst_inv

            H_t = H[t]
            H_tinv = H[t_inv]

            HHT = th.matmul(H_t, H_t.T)
            Dpt = g.out_degrees(etype=etype).unsqueeze(1)

            """ Dpt Yp (Ht Ht.T) """
            ret[src] -= Dpt * th.matmul(Y[src], HHT)
            
            """ At Ys' (Ht.T + Htinv) """
            g.edges[etype].data['out'] = th.matmul(g.edata['m'][etype],(H_t + H_tinv.T))


        g.update_all(fn.copy_e('out','m'), fn.sum('m', 'h'))

        for ntype in g.ntypes:
            ret[ntype] += g.dstdata['h'][ntype] + X[ntype]
            ret[ntype] = (1 - alp)*Y[ntype] + alp * lam * (1 / (1 + lam * g.nodes[ntype].data['deg'])) * ret[ntype]
            ret[ntype] = F.relu(ret[ntype])
                
        return ret

class ZooBP_Propagate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, g, Y, X, H, d, eps, etypes_inv, category_dim_dict):
        ret = {}
        for ntype in g.ntypes:
            # This is to avoid call by reference
            ret[ntype] = X[ntype] - 0
        
        g.srcdata['h'] = Y
        g.apply_edges(fn.u_mul_e('h', 'w', 'm'))

        for etype in g.canonical_etypes:
            etype_inv = etypes_inv[etype]

            src, rel, dst = etype
            t = src + '-' + rel + '-' + dst

            src_inv, rel_inv, dst_inv = etype_inv
            t_inv = src_inv + '-' + rel_inv + '-' + dst_inv

            src_dim = category_dim_dict[src]
            dst_dim = category_dim_dict[dst]

            HHT = th.matmul(H[t], H[t].T)
            Dpt = g.out_degrees(etype=etype).unsqueeze(1)          
            

            """ Dpt Yp (Ht Ht.T) """
            ret[src] -= Dpt * th.matmul(Y[src], HHT) * ((eps / src_dim) * (eps / dst_dim))
            
            """ At Ys' Ht.T """
            g.edges[etype].data['out'] = th.matmul(g.edata['m'][etype], H[t_inv].T) * (eps / dst_dim)
            

        g.update_all(fn.copy_e('out','m'), fn.sum('m', 'h'))

        for ntype in g.ntypes:
            ret[ntype] += g.dstdata['h'][ntype]

        return ret