import sys
sys.path.insert(0,"/opt/reconstruct/Easy-Graph")
import easygraph as eg
import dhg
import random
random.seed(42)




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
    hg = eg.Hypergraph(num_v = node_num,e_list=list(edge_set))
    print('hg_eg:',hg)

    return hg

def load_integrated_dataset(dataset_name = "cora"):
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
    eg_hg = eg.Hypergraph(num_v=len(node_dict),e_list=edge_reindex_lst)
    return eg_hg



if __name__ == "__main__":
    hg_eg = load_integrated_dataset("trivago_clicks")
    
    # hg_eg = load_dataset("/home/msn/ybd/Easy-Graph/hypergraph-bench/eg_hypergraph_dataset/walmart-trips/node-labels-walmart-trips.txt",
    #                      "/home/msn/ybd/Easy-Graph/hypergraph-bench/eg_hypergraph_dataset/walmart-trips/hyperedges-walmart-trips.txt")
    
    ic = hg_eg.incidence_matrix
    
    # deg = hg_eg.degree_node
    
    # dis = hg_eg.distance(0)
    
    # cc = list(hg_eg.s_connected_components(edges=False))
    # print("cc:",len(cc))
    
    # for node in range(hg_eg.num_v):
    #     neighbor = hg_eg.neighbor_of_node(node)
    
    