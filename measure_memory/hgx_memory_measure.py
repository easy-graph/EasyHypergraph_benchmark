import sys
sys.path.insert(0,"/opt/reconstruct/Easy-Graph")

import easygraph as eg
import dhg
# from benchmark import benchmark
import random
import xgi
import hypernetx as hnx
import hypergraphx as hgx
import time
random.seed(42)
import pandas as pd

print("sys path",sys.path)
print("eg",eg.__file__)
print("xgi:",xgi.__file__)

def load_cocitation_dataset(dataset_name = "cora"):
    if dataset_name == 'cora_cocitation':
        cocitation_dataset = eg.CocitationCora()
    elif dataset_name == "cora_authorship":
        cocitation_dataset = dhg.data.CoauthorshipCora()
    elif dataset_name == "pubmed":
        cocitation_dataset = eg.CocitationPubmed()
    elif dataset_name == "dblp_authorship":
        cocitation_dataset = dhg.data.CoauthorshipDBLP()
    else:
        cocitation_dataset = eg.CocitationCiteseer()

    cocitation_dataset.needs_to_load("edge_list")
    edge_lst = set()
    edge_reindex_lst = set()
    node_dict = {}
    node_index = 0
    for e in cocitation_dataset['edge_list']:
        # edge_lst.add(tuple(e))
        edge_reindex = []
        for n in e:
            if n not in node_dict:
                node_dict[n] = node_index
                edge_reindex.append(node_index)
                node_index = node_index + 1
            else:
                edge_reindex.append(node_dict[n])
        edge_reindex_lst.add(tuple(edge_reindex))


    edge_reindex_lst = list(edge_reindex_lst)
    # eg_hg = eg.Hypergraph(num_v=len(node_dict),e_list=edge_reindex_lst)
    # print("hg_eg len:",len(eg_hg.e[0]),"vertices:",eg_hg.num_v)
    # hg_hnx = hnx.Hypergraph(edge_reindex_lst)
    # # hg_hnx = hnx.from_incidence_dataframe(eg_hg.incidence_matrix)
    # print("hg_hhx len:", len(hg_hnx.edges), "vertices:", len(hg_hnx.nodes))
    # hg_xgi = xgi.Hypergraph(edge_reindex_lst)
    # print("hg_xgi len:", len(hg_xgi.edges), "vertices:", len(hg_xgi.nodes))
    hg_hgx = hgx.Hypergraph(edge_list = edge_reindex_lst)
    # print("hg_hgx len:",hg_hgx.num_edges(),"vertices:",hg_hgx.num_nodes())
    # return eg_hg,hg_hnx,hg_xgi,hg_hgx
    return hg_hgx

# def run_fun():
    

if __name__ == "__main__":
    print("start test on cocitation-cora dataset.......")
    hg_hgx = load_cocitation_dataset("dblp_authorship")
    
    # ic = hgx.Hypergraph.incidence_matrix(hg_hgx)
    
    # for node in hg_hgx.get_nodes():
    #     deg = hg_hgx.degree(node)
    
    
    
    for node in hg_hgx.get_nodes():
        cc = hg_hgx.node_connected_component(node)
    
    
    # for node in hg_hgx.get_nodes():
    #     neighbor = hg_hgx.get_neighbors(node)
    
    