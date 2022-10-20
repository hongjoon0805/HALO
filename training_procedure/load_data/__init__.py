import pdb
import pickle
import random
from functools import reduce

import dgl
import os
import torch as th
from dgl.data.utils import download, extract_archive
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs
from ogb.nodeproppred import DglNodePropPredDataset
from tqdm import tqdm
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset

from .load_ACM import load_data_ACM
from .load_IMDB import load_data_IMDB
from .load_DBLP import load_data_DBLP
from .load_DBLP_reduced import load_data_DBLP_reduced
from .load_Academic_reduced import load_data_Academic_reduced
from .load_Freebase import load_data_Freebase
from .load_KG import load_data_KG

class HGBDataset(DGLDataset):

    _prefix = 'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/'
    _urls = {

    }

    def __init__(self, name, raw_dir=None, force_reload=False, verbose=True):
        assert name in ['HGBn-ACM', 'HGBn-DBLP', 'HGBn-Freebase', 'HGBn-IMDB',
                        'HGBl-amazon', 'HGBl-LastFM', 'HGBl-PubMed']
        self.prefix_task = name[:4]
        # HGBn means node classification
        # HGBl means link prediction
        self.data_path = './openhgnn/dataset/{}.zip'.format(self.prefix_task)
        self.g_path = './openhgnn/dataset/{}/{}.bin'.format(self.prefix_task, name)
        raw_dir = './openhgnn/dataset'
        url = self._prefix + 'dataset/{}.zip'.format(self.prefix_task)
        super(HGBDataset, self).__init__(name=name,
                                        url=url,
                                        raw_dir=raw_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)

    def download(self):
        # download raw data to local disk
        # path to store the file
        if os.path.exists(self.data_path):  # pragma: no cover
            pass
        else:
            file_path = os.path.join(self.raw_dir)
            # download file
            download(self.url, path=file_path)
        extract_archive(self.data_path, os.path.join(self.raw_dir, self.prefix_task))

    def process(self):
        # process raw data to graphs, labels, splitting masks
        g, _ = load_graphs(self.g_path)
        self._g = g[0]

    def __getitem__(self, idx):
        # get one example by index
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        # number of data examples
        return 1

    def save(self):
        # save processed data to directory `self.save_path`
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        pass

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        pass

class AcademicDataset(DGLDataset):

    _prefix = 'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/'
    _urls = {
        'academic4HetGNN': 'dataset/academic4HetGNN.zip',
        'acm4GTN': 'dataset/acm4GTN.zip',
        'acm4NSHE': 'dataset/acm4NSHE.zip',
        'acm4NARS': 'dataset/acm4NARS.zip',
        'acm4HeCo': 'dataset/acm4HeCo.zip',
        'imdb4MAGNN': 'dataset/imdb4MAGNN.zip',
        'imdb4GTN': 'dataset/imdb4GTN.zip',
        'DoubanMovie': 'dataset/DoubanMovie.zip',
        'dblp4MAGNN': 'dataset/dblp4MAGNN.zip',
        'yelp4HeGAN': 'dataset/yelp4HeGAN.zip',
        'yelp4rec': 'dataset/yelp4rec.zip',
        'HNE-PubMed': 'dataset/HNE-PubMed.zip',
        'MTWM': 'dataset/MTWM.zip',
        'amazon4SLICE': 'dataset/amazon4SLICE.zip'
    }

    def __init__(self, name, raw_dir=None, force_reload=False, verbose=True):
        assert name in ['acm4GTN', 'acm4NSHE', 'academic4HetGNN', 'imdb4MAGNN', 'imdb4GTN', 'HNE-PubMed', 'MTWM',
                        'DoubanMovie', 'dblp4MAGNN', 'acm4NARS', 'acm4HeCo', 'yelp4rec', 'yelp4HeGAN', 'amazon4SLICE']
        self.data_path = './openhgnn/' + self._urls[name]
        self.g_path = './openhgnn/dataset/' + name + '/graph.bin'
        raw_dir = './openhgnn/dataset'
        url = self._prefix + self._urls[name]
        super(AcademicDataset, self).__init__(name=name,
                                        url=url,
                                        raw_dir=raw_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)

    def download(self):
        # download raw data to local disk
        # path to store the file
        if os.path.exists(self.data_path):  # pragma: no cover
           pass
        else:
            file_path = os.path.join(self.raw_dir)
            # download file
            download(self.url, path=file_path)
        extract_archive(self.data_path, os.path.join(self.raw_dir, self.name))

    def process(self):
        # process raw data to graphs, labels, splitting masks
        g, _ = load_graphs(self.g_path)
        self._g = g[0]

    def __getitem__(self, idx):
        # get one example by index
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        # number of data examples
        return 1

    def save(self):
        # save processed data to directory `self.save_path`
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        pass

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        pass

def find_etypes_inv(g):
    etypes_inv = {}

    for etype1 in g.canonical_etypes:
        src1, t1, dst1 = etype1
        if etype1 in etypes_inv:
            continue
        if src1 == dst1:
            etypes_inv[etype1] = etype1
            continue
        u1, v1 = g.edges(etype=etype1)
        for etype2 in g.canonical_etypes:
            src2, t2, dst2 = etype2

            if src1 != dst2 or dst1 != src2:
                continue
            
            u2, v2 = g.edges(etype=etype2)

            if len(u1) != len(u2):
                continue

            u1, v1 = u1.sort()[0], v1.sort()[0]
            u2, v2 = u2.sort()[0], v2.sort()[0]       
            
            if ((u1 - v2)==0).sum()==len(u1) and ((v1 - u2)==0).sum()==len(u1):
                etypes_inv[etype1] = etype2
                etypes_inv[etype2] = etype1
    
    return etypes_inv

"""
load_datas_xxx return:
    graph , labels , train_nodes , dev_nodes , test_nodes


graph: dgl.graph
graph.ndata["feature"]: torch.FloatTensor(n , inp_d)

labels: torch.LongTensor(n)
train_nodes: torch.LongTensor(n)
dev_nodes: torch.LongTensor(n)
test_nodes: torch.LongTensor(n)

"""

def load_data(self, idx):
    C = self.C

    if C.data == 'ACM':
        graph, labels, train_nodes, val_nodes, test_nodes = load_data_ACM(self, HGBDataset('HGBn-ACM')._g)
    elif C.data == 'IMDB':
        graph, labels, train_nodes, val_nodes, test_nodes = load_data_IMDB(self, HGBDataset('HGBn-IMDB')._g)
    elif C.data == 'DBLP':
        graph, labels, train_nodes, val_nodes, test_nodes = load_data_DBLP(self, HGBDataset('HGBn-DBLP')._g)
    elif C.data == 'DBLP_reduced':
        graph, labels, train_nodes, val_nodes, test_nodes = load_data_DBLP_reduced(self, HGBDataset('HGBn-DBLP')._g)
    elif C.data == 'Freebase':
        graph, labels, train_nodes, val_nodes, test_nodes = load_data_Freebase(self, HGBDataset('HGBn-Freebase')._g)
    elif C.data == 'Academic_reduced':
        graph, labels, train_nodes, val_nodes, test_nodes = load_data_Academic_reduced(self, AcademicDataset('academic4HetGNN')[0])
    elif C.data == 'AIFB':
        graph, labels, train_nodes, val_nodes, test_nodes = load_data_KG(self, AIFBDataset()[0])
    elif C.data == 'MUTAG':
        graph, labels, train_nodes, val_nodes, test_nodes = load_data_KG(self, MUTAGDataset()[0])
    elif C.data == 'BGS':
        graph, labels, train_nodes, val_nodes, test_nodes = load_data_KG(self, BGSDataset()[0])
    elif C.data == 'AM':
        graph, labels, train_nodes, val_nodes, test_nodes = load_data_KG(self, AMDataset()[0])

    etypes_inv = find_etypes_inv(graph)
    return graph, labels, etypes_inv, train_nodes, val_nodes, test_nodes

