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

def load_dataset(nodes_path, edges_path):
    node_num = 0
    with open(nodes_path, 'r') as nodes_file:
        for line in nodes_file:
            node_num += 1

    print("node_num:",node_num)
    # hg = eg.Hypergraph(num_v = node_num)

    edge_set = set()
    with open(edges_path, 'r') as egdes_file:
        for line in egdes_file:
            hyperedge = line.strip()
            hyperedge = [int(i) - 1 for i in hyperedge.split(",")]
            edge_set.add(tuple(hyperedge))

    total_edge_size = 0
    for e in edge_set:
        total_edge_size += len(e)
    print("avg edge size:",total_edge_size/len(edge_set))
    
    hg_hnx = hnx.Hypergraph(list(edge_set))
    # hg_xgi = xgi.Hypergraph(list(edge_set))
    # hg_hgx = hgx.Hypergraph(edge_list = list(edge_set))
    # print('hg_eg:',hg)
    # print("hg_hnx:",len(hg_hnx.nodes),len(hg_hnx.edges))
    # print("hg_xgi:",len(hg_xgi.nodes),len(hg_xgi.edges))
    # print("set len:",len(edge_set))
    return hg_hnx

def load_cocitation_dataset(dataset_name = "cora"):
    if dataset_name == 'trivago_clicks':
        dataset = eg.trivago_clicks()
    elif dataset_name == "pubmed":
        dataset = eg.CocitationPubmed()
    elif dataset_name == "dblp_authorship":
        dataset = eg.CoauthorshipDBLP()
    elif dataset_name == "yelp":
        dataset = eg.YelpRestaurant()
    if dataset_name not in ['trivago_clicks']:
        dataset.needs_to_load("edge_list")

    edge_reindex_lst = set()
    node_dict = {}
    node_index = 0
    for e in dataset['edge_list']:
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
    hg_hnx = hnx.Hypergraph(edge_reindex_lst)
    return hg_hnx

# def run_fun():
    

if __name__ == "__main__":
    hg_hnx = load_cocitation_dataset("trivago_clicks")
    
    # hg_hnx = load_dataset("/home/msn/ybd/Easy-Graph/hypergraph-bench/eg_hypergraph_dataset/walmart-trips/node-labels-walmart-trips.txt","/home/msn/ybd/Easy-Graph/hypergraph-bench/eg_hypergraph_dataset/walmart-trips/hyperedges-walmart-trips.txt")

    ic = hg_hnx.incidence_matrix()
    
    # for node in hg_hnx.nodes:
    #     deg = hg_hnx.degree(node)
    
    
        
    # for i in range(1,len(hg_hnx.nodes)):
    #     dis = hg_hnx.distance(0, i)
    
    
    # cc = list(hg_hnx.s_connected_components(edges=False))
    
    
    # for node in hg_hnx.nodes:
    #     neighbor = hg_hnx.neighbors(node)
    
    