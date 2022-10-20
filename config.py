from argparse import ArgumentParser

def get_config():
    par = ArgumentParser()

    # ---------------------- Universal ----------------------
    par.add_argument("--seed"         , type = int  , default = 2333)
    par.add_argument("--date"         , type = str )
    par.add_argument("--multirun"     , type = int  , default = 1)
    par.add_argument("--info"         , type = str  , default = "")

    # ---------------------- Data ----------------------
    par.add_argument("--data"         , type = str  , default = "ACM")
    par.add_argument("--rand_edge"    , type = float, default = 0.0)
    par.add_argument("--cache_attack" , action = "store_true" , default = False)

    # for those datasets who don't have a standard split
    par.add_argument("--change_split" , type = int  , default = 1 ) # change split per k runs
    par.add_argument("--train_num"    , type = int  , default = 20)
    par.add_argument("--dev_num"      , type = int  , default = 30)

    # ---------------------- Label ----------------------
    par.add_argument("--multilabel"   , action = 'store_true')
    par.add_argument("--multicategory"   , action = 'store_true')

    # ---------------------- ZooBP ----------------------
    par.add_argument("--ZooBP"   , action = 'store_true')
    par.add_argument("--eps"          , type = float, default = 1)

    # ---------------------- Model ----------------------

    par.add_argument("--hidden_size"  , type = int  , default = 128)
    par.add_argument("--norm"         , type = str  , default = "none")
    par.add_argument("--dropout"      , type = float, default = 0.0)
    par.add_argument("--attn_dropout" , type = float, default = 0.0)
    par.add_argument("--inp_dropout"  , type = float, default = 0.0)
    par.add_argument("--learn_emb"    , type = int  , default = 0)

    par.add_argument("--mlp_bef"  , type = int  , default = 1)
    par.add_argument("--mlp_aft"  , type = int  , default = 0)


    # propagation
    par.add_argument("--no_train_H"   , action = "store_true" , default = False)
    par.add_argument("--residual"   , action = "store_true" , default = False)
    par.add_argument("--shared_H"   , action = "store_true" , default = False)
    par.add_argument("--no_relu"   , action = "store_true" , default = False)
    par.add_argument("--load_and_eval"   , action = "store_true" , default = False)
    par.add_argument("--prop_step"    , type = int  , default = 2)
    par.add_argument("--alp"          , type = float, default = 0)  # 0 for alpha = 1 / (1 + lambda)
    par.add_argument("--lam"          , type = float, default = 1)

    # ---------------------- Train & Test ----------------------
    par.add_argument("--num_epoch"    , type = int  , default = 500)
    par.add_argument("--patience"     , type = int  , default = -1)
    par.add_argument("--lr"           , type = float, default = 1e-3)
    par.add_argument("--weight_decay" , type = float, default = 0)

    # if set, report the performance of last epoch
    par.add_argument("--no_dev"    , action = "store_true" , default = False)

    C = par.parse_args()

    return C
