import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from model.modules import HeteroUnfolding , MLP, ZooBP
import pdb

class GNNModel(nn.Module):
    def __init__( self , 
        C           ,
        g           ,
        input_d     , 
        output_d    , 
        num_dims_dict,
        etypes_inv,
        category_dim_dict
    ):
        super().__init__()

        self.C              = C
        self.g              = g
        self.input_d        = input_d
        self.output_d       = output_d
        self.hidden_d       =  C.hidden_size
        self.prop_step      =  C.prop_step 
        self.num_mlp_before =  C.mlp_bef 
        self.num_mlp_after  =  C.mlp_aft
        self.norm           =  C.norm
        self.train_H        =  not C.no_train_H
        self.dropout        =  C.dropout
        self.inp_dropout    =  C.inp_dropout
        self.learn_emb      =  (num_dims_dict, C.learn_emb)
        self.ntypes         =  g.ntypes
        self.canonical_etypes= g.canonical_etypes
        self.etypes_inv      = etypes_inv
        self.category_dim_dict = category_dim_dict

        # ----- initialization of some variables -----

        # whether to learn a embedding for each node.

        if self.learn_emb[1] > 0:
            if (self.C.data == 'AM' or self.C.data == 'Freebase') and 'OnehotFeat' in self.C.date:
                emb = {}
                for ntype in self.ntypes:
                    emb[ntype] = nn.Parameter(th.rand(self.learn_emb[0][ntype], self.learn_emb[1]))
                    nn.init.xavier_normal_(emb[ntype], gain=nn.init.calculate_gain('relu'))
                self.emb = nn.ParameterDict(emb)
                self.input_d  = self.learn_emb[1]
                pass
            else:
                emb = {}
                for ntype in self.ntypes:
                    emb[ntype] = nn.Linear(self.learn_emb[0][ntype], self.learn_emb[1], bias=False)
                    nn.init.xavier_normal_(emb[ntype].weight, gain=nn.init.calculate_gain('relu'))
                self.emb = nn.ModuleDict(emb)
                self.input_d  = self.learn_emb[1]

        # if only one layer, then no hidden size
        self.size_bef_unf = self.hidden_d
        self.size_aft_unf = self.hidden_d
        if self.num_mlp_before == 0:
            self.size_aft_unf = self.input_d  # as the input  of mlp_aft
        if self.num_mlp_after == 0:
            self.size_bef_unf = self.output_d # as the output of mlp_bef

        # ----- computational modules -----
        self.mlp_bef = MLP(self.input_d , self.hidden_d , self.size_bef_unf , self.num_mlp_before , 
                self.dropout , self.norm , init_activate = False)

        if self.C.ZooBP:
            self.unfolding = ZooBP(self.size_bef_unf, self.C, self.ntypes, 
                    self.canonical_etypes, self.etypes_inv, self.category_dim_dict)
        else:
            self.unfolding = HeteroUnfolding(self.size_bef_unf, self.C,
                    self.ntypes, self.canonical_etypes, self.etypes_inv, self.category_dim_dict)

        # if there are really transformations before unfolding, then do init_activate in mlp_aft
        self.mlp_aft = MLP(self.size_aft_unf , self.hidden_d , self.output_d , self.num_mlp_after  , 
            self.dropout , self.norm , 
            init_activate = (self.num_mlp_before > 0) and (self.num_mlp_after > 0) 
        )

    def forward(self , g, evaluate = False):

        x = g.ndata["feature"]

        # use trained node embedding
        if self.learn_emb[1] > 0:
            if (self.C.data == 'AM' or self.C.data == 'Freebase') and 'OnehotFeat' in self.C.date:
                for ntype in g.ntypes:
                    x[ntype] = self.emb[ntype]
            else:
                for ntype in g.ntypes:
                    x[ntype] = self.emb[ntype](x[ntype])
            
        if self.inp_dropout > 0:
            for ntype in g.ntypes:
                x[ntype] = F.dropout(x[ntype], self.inp_dropout, training = self.training)
        
        for ntype in g.ntypes:
            x[ntype] = self.mlp_bef(x[ntype])
            
        x = self.unfolding(g , x, evaluate = evaluate)

        for ntype in g.ntypes:
            x[ntype] = self.mlp_aft(x[ntype])
                

        return x