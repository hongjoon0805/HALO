## **Descent Steps of a Relation-Aware Energy Produce Heterogeneous Graph Neural Networks**

------

This repo is official code for NeurIPS 2022 paper ["Descent Steps of a Relation-Aware Energy Produce Heterogeneous Graph Neural Networks"](https://arxiv.org/abs/2206.11081) 

## **Execution Details**
------

### Requirements
------

- Python 3.7.10
- Pytorch: 1.10.0
- DGL: 0.8.0
- CUDA: 11.4

------
### Downloading dataset
------
- Knowledge graphs: Downloaded from dgl.data.rdf
- HGB datasets: Pre-defined class "HGBDataset" try to download the datasets
- Academic datset: Pre-defined class "AcademicDataset" try to download the datasets

All the datasets are automatically downloaded if you run one of the following command. 

------
### Reproducing Table 1
------

#### Execution command

```
# AIFB
$ python main.py --date NeurIPS2022 --seed 0 --mlp_bef=1 --mlp_aft=1 --prop_step=16 --num_epoch=1000 --lam=1.0 --alp=0.1 --dropout 0.5 --inp_dropout 0.5 --learn_emb=16 --hidden_size=16 --lr 0.001 --weight_decay 1e-05 --data=AIFB

# MUTAG
$ python main.py --date NeurIPS2022 --seed 0 --mlp_bef=1 --mlp_aft=1 --prop_step=16 --num_epoch=1000 --lam=0.01 --alp=1.0 --dropout 0.5 --inp_dropout 0.5 --learn_emb=16 --hidden_size=16 --lr 0.001 --weight_decay 0.0001 --data=MUTAG

# BGS
$ python main.py --date NeurIPS2022 --seed 0 --mlp_bef=1 --mlp_aft=1 --prop_step=8 --num_epoch=1000 --lam=0.1 --alp=1.0 --dropout 0.5 --inp_dropout 0.5 --learn_emb=16 --hidden_size=16 --lr 0.01 --weight_decay 1e-05 --data=BGS

# AM
$ python main.py --date NeurIPS2022BinFeat --seed 0 --mlp_bef=1 --mlp_aft=1 --prop_step=4 --num_epoch=1000 --lam=1.0 --alp=1.0 --dropout 0.5 --inp_dropout 0.5 --learn_emb=16 --hidden_size=16 --lr 0.01 --weight_decay 0.0001 --data=AM

# DBLP
$ python main.py --date NeurIPS2022 --seed 0 --mlp_bef=1 --mlp_aft=1 --prop_step=8 --num_epoch=1000 --lam=1.0 --alp=1.0 --dropout 0.5 --inp_dropout 0.5 --learn_emb=256 --hidden_size=256 --lr 0.0001 --weight_decay 1e-05 --data=DBLP

# IMDB
$ python main.py --date NeurIPS2022 --seed 0 --mlp_bef=1 --mlp_aft=1 --prop_step=32 --num_epoch=1000 --lam=1.0 --alp=1.0 --dropout 0.5 --inp_dropout 0.5 --learn_emb=64 --hidden_size=64 --lr 0.001 --weight_decay 1e-05 --data=IMDB

# ACM
$ python main.py --date NeurIPS2022 --seed 0 --mlp_bef=1 --mlp_aft=1 --prop_step=32 --num_epoch=1000 --lam=0.1 --alp=0.1 --dropout 0.5 --inp_dropout 0.5 --hidden_size=32 --lr 0.01 --weight_decay 0.0001 --data=ACM

# Freebase
$ python main.py --date NeurIPS2022NodeFeat --seed 0 --mlp_bef=1 --mlp_aft=1 --prop_step=4 --num_epoch=1000 --lam=1.0 --alp=1.0 --dropout 0.5 --inp_dropout 0.5 --hidden_size=32 --lr 0.01 --weight_decay 0.001 --data=Freebase
```

------
### Reproducing Table 2
------

```
# DBLP-reduced
$ python main.py --date NeurIPS2022 --seed 0 --mlp_bef=1 --mlp_aft=1 --prop_step=8 --num_epoch=1000 --lam=1.0 --alp=1.0 --dropout 0.5 --inp_dropout 0.5 --learn_emb=256 --hidden_size=256 --lr 0.0001 --weight_decay 1e-05 --data=DBLP_reduced --multicategory

$ python main.py --date NeurIPS2022 --seed 0 --prop_step=1000 --num_epoch=1 --ZooBP --data=DBLP_reduced --multicategory

# Academic-reduced
$ python main.py --date NeurIPS2022 --seed 0 --mlp_bef=1 --mlp_aft=1 --prop_step=8 --num_epoch=1000 --lam=1.0 --alp=1.0 --dropout 0.5 --inp_dropout 0.5 --learn_emb=256 --hidden_size=256 --lr 0.0001 --weight_decay 1e-05 --data=Academic_reduced --multicategory

$ python main.py --date NeurIPS2022 --seed 0 --prop_step=1000 --num_epoch=1 --ZooBP --data=Academic_reduced --multicategory
```

