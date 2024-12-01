import sys

import easygraph as eg
import dhg
from benchmark import benchmark
import random
import xgi
import hypernetx as hnx
import time
random.seed(42)
import pandas as pd

def dataset_property(hg):
    degree_dict = hg_eg.degree_node
    avg_degree = sum(degree_dict.values())/len(degree_dict)
    print("avg degree:",avg_degree)

def load_integrated_dataset(dataset_name = "cora"):
    if dataset_name == 'cora_cocitation':
        cocitation_dataset = eg.CocitationCora()
    elif dataset_name == "cora-coauthorship":
        cocitation_dataset = eg.CoauthorshipCora()
    elif dataset_name == "pubmed":
        cocitation_dataset = eg.CocitationPubmed()
    elif dataset_name == "dblp_authorship":
        cocitation_dataset = eg.CoauthorshipDBLP()
    elif dataset_name == "yelp":
        cocitation_dataset = eg.YelpRestaurant()
    else:
        cocitation_dataset = eg.CocitationCiteseer()

    cocitation_dataset.needs_to_load("edge_list")
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


    print("node dict:",len(node_dict))
    print("edge_reindex_lst:",len(edge_reindex_lst))
    total_edge_size = 0
    for e in edge_reindex_lst:
        total_edge_size += len(e)
    print("avg hyperedge size:",total_edge_size/len(edge_reindex_lst))
    edge_reindex_lst = list(edge_reindex_lst)
    eg_hg = eg.Hypergraph(num_v=len(node_dict),e_list=edge_reindex_lst)
    print("hg_eg len:",len(eg_hg.e[0]),"vertices:",eg_hg.num_v)
    hg_hnx = hnx.Hypergraph(edge_reindex_lst)
    print("hg_hhx len:", len(hg_hnx.edges), "vertices:", len(hg_hnx.nodes))
    hg_xgi = xgi.Hypergraph(edge_reindex_lst)
    print("hg_xgi len:", len(hg_xgi.edges), "vertices:", len(hg_xgi.nodes))

    return eg_hg,hg_hnx,hg_xgi






def load_dataset(nodes_path, edges_path):
    node_num = 0
    with open(nodes_path, 'r') as nodes_file:
        for line in nodes_file:
            node_num += 1

    print("node_num:",node_num)

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
    
    hg = eg.Hypergraph(num_v = node_num,e_list=list(edge_set))
    
    hg_hnx = hnx.Hypergraph(list(edge_set))
    hg_xgi = xgi.Hypergraph(list(edge_set))
    print('hg_eg:',hg)
    print("hg_hnx:",len(hg_hnx.nodes),len(hg_hnx.edges))
    print("hg_xgi:",len(hg_xgi.nodes),len(hg_xgi.edges))
    print("set len:",len(edge_set))
    return hg,hg_xgi,hg_hnx



def comp_fun(dataset,hg_eg, hg_xgi, hg_hnx):
    columns = ["metric","EasyGraph","XGI","HyperNetX"]
    df = pd.DataFrame(columns=columns)
    print("finish_loading...")

    print("incidence matrix:")
    eg_incidence_matrix_time = benchmark("hg_eg.incidence_matrix", globals=globals(), n=5)
    print("eg incidence matrix:", eg_incidence_matrix_time)
    
    xgi_incidence_matrix_time = benchmark("xgi.to_incidence_matrix(hg_xgi)", globals=globals(),n=5)
    print("xgi incidence matrix:", xgi_incidence_matrix_time)
    hnx_incidence_matrix_time = benchmark("hg_hnx.incidence_matrix()", globals=globals(), n=5)
    print("hg_hnx incidence_matrix time:", hnx_incidence_matrix_time)

    df_record=["incidence_matrix",eg_incidence_matrix_time, xgi_incidence_matrix_time, hnx_incidence_matrix_time]
    df.loc[len(df)] = df_record

    print()

    print("degree:")
    eg_degree_time = benchmark("hg_eg.degree_node", globals=globals(), n=5)
    print("eg degree:", eg_degree_time)
    
    xgi_degree_time = benchmark("hg_xgi.nodes.degree", globals=globals(),n=5)
    print("xgi degree:", xgi_degree_time)
    
    hnx_degree_mean_time = 0
    for node in hg_hnx.nodes:
        hnx_degree_mean_time += benchmark("hg_hnx.degree("+str(node)+")", globals=globals(), n=5)
    print("hnx degree:", hnx_degree_mean_time)
    

    df_record=["degree",eg_degree_time, xgi_degree_time, hnx_degree_mean_time]
    df.loc[len(df)] = df_record

    print()
    print("distance:")
    eg_distance_time = benchmark("hg_eg.distance(0)", globals=globals(), n=5)
    print("eg distance:", eg_distance_time)
    
    
    xgi_distance_time = benchmark("xgi.single_source_shortest_path_length(hg_xgi,0)", globals=globals(),n=5)
    print("xgi distance time:", xgi_distance_time)

    hnx_distance_time = 0
    for i in range(1,len(hg_hnx.nodes)):
        hnx_distance_time += benchmark("hg_hnx.distance(0,"+str(i)+")", globals=globals(), n=5)
    print("hg_hnx distance:",hnx_distance_time)

    df_record = ["distance",eg_distance_time, xgi_distance_time, hnx_distance_time]

    df.loc[len(df)] = df_record

    print()
    print("neighbors:")
    eg_neighbors_time = 0
    for node in range(hg_eg.num_v):
        eg_neighbors_time += benchmark("hg_eg.neighbor_of_node("+str(node)+")", globals=globals(), n=5)
    print("eg neighbors:", eg_neighbors_time)
    
    xgi_neighbors_time = 0
    for node in range(hg_eg.num_v):
        xgi_neighbors_time += benchmark("hg_xgi.nodes.neighbors("+str(node)+")", globals=globals(), n=5)
    print("xgi neighbors:", xgi_neighbors_time)
    
    hnx_neighbors_time = 0
    for node in range(hg_eg.num_v):
        hnx_neighbors_time += benchmark("hg_hnx.neighbors("+str(node)+")", globals=globals(), n=5)
    print("hg_hnx neighbors:", hnx_neighbors_time)

    
    df_record = ["neighbors",eg_neighbors_time, xgi_neighbors_time, hnx_neighbors_time, hgx_neighbors_time]
    df.loc[len(df)] = df_record

    df.to_csv('results/'+dataset+'_results.csv', mode='a',  index=False)

    
if __name__ == "__main__":

    print("start test on cocitation-pubmed dataset.......")
    hg_eg, hg_hnx, hg_xgi = load_integrated_dataset("pubmed")
    dataset_property(hg_eg)
    comp_fun("pubmed", hg_eg, hg_xgi, hg_hnx)


    print("start test on coauthorship-dblp dataset.......")
    hg_eg, hg_hnx, hg_xgi = load_integrated_dataset("dblp_authorship")
    dataset_property(hg_eg)
    comp_fun("dblp_authorship", hg_eg, hg_xgi, hg_hnx)

    print("start test on yelp dataset.......")
    hg_eg, hg_hnx, hg_xgi = load_integrated_dataset("yelp")
    dataset_property(hg_eg)
    comp_fun("yelp", hg_eg, hg_xgi, hg_hnx)

    
    print("  ")
    print("start test on walmart-trips dataset.......")
    hg_eg,hg_xgi,hg_hnx = load_dataset(nodes_path="eg_hypergraph_dataset/walmart-trips/node-labels-walmart-trips.txt",
                                       edges_path="eg_hypergraph_dataset/walmart-trips/hyperedges-walmart-trips.txt")
    dataset_property(hg_eg)
    print("finish_loading...")
    comp_fun("walmart_trips",hg_eg,hg_xgi,hg_hnx)


    print("  ")
    print("start test on trivago-clicks dataset.......")
    hg_eg,hg_xgi,hg_hnx = load_dataset(nodes_path="eg_hypergraph_dataset/trivago-clicks/node-labels-trivago-clicks.txt",
                                       edges_path="eg_hypergraph_dataset/trivago-clicks/hyperedges-trivago-clicks.txt")
    dataset_property(hg_eg)
    print("finish_loading...")
    comp_fun("trivago_clicks",hg_eg,hg_xgi,hg_hnx)




