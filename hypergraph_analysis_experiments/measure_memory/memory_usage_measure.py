import psutil
import argparse
import dhg
import xgi
import hypernetx as hnx
import hypergraphx as hgx
import sys
sys.path.insert(0,"/opt/Easy-Graph")
import easygraph as eg
def load_cocitation_dataset(dataset_name = "cora", tp = "eg"):
    if dataset_name == 'cora_cocitation':
        cocitation_dataset = eg.CocitationCora()
    elif dataset_name == "cora_authorship":
        cocitation_dataset = dhg.data.CoauthorshipCora()
    elif dataset_name == "pubmed":
        cocitation_dataset = eg.CocitationPubmed()
    elif dataset_name == "dblp_authorship":
        cocitation_dataset = dhg.data.CoauthorshipDBLP()
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


    edge_reindex_lst = list(edge_reindex_lst)
    if tp == "eg":
        eg_hg = eg.Hypergraph(num_v=len(node_dict),e_list=edge_reindex_lst)
        return eg_hg
    # print("hg_eg len:",len(eg_hg.e[0]),"vertices:",eg_hg.num_v)
    elif tp == "hnx":
        hg_hnx = hnx.Hypergraph(edge_reindex_lst)
        return hg_hnx
    else:
        hg_xgi = xgi.Hypergraph(edge_reindex_lst)
        return hg_xgi
    # # hg_hnx = hnx.from_incidence_dataframe(eg_hg.incidence_matrix)
    # print("hg_hhx len:", len(hg_hnx.edges), "vertices:", len(hg_hnx.nodes))
    
    # print("hg_xgi len:", len(hg_xgi.edges), "vertices:", len(hg_xgi.nodes))
    # hg_hgx = hgx.Hypergraph(edge_list = edge_reindex_lst)
    # print("hg_hgx len:",hg_hgx.num_edges(),"vertices:",hg_hgx.num_nodes())
    # return eg_hg,hg_hnx,hg_xgi,hg_hgx

def memory_usage():
    # 获取当前进程的内存使用情况
    process = psutil.Process()
    return process.memory_info().rss / 1024 ** 2  # 返回内存使用，单位MB

def load_dataset(nodes_path, edges_path, tp):
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
    
    
    if tp == "eg":
        eg_hg = eg.Hypergraph(num_v = node_num,e_list=list(edge_set))
        return eg_hg
    elif tp == "hnx":
        hg_hnx = hnx.Hypergraph(list(edge_set))
        return hg_hnx
    else:
        hg_xgi = xgi.Hypergraph(list(edge_set))
        return hg_xgi
    
def calculate_degrees(hg, tp):
    if tp == "eg":
        deg = hg.degree_node
    elif tp == "hnx":
        deg = []
        for node in hg.nodes:
            deg.append(hg.degree(node))
    else:
        deg = hg.nodes.degree
        
    return deg    

def calculate_distance(hg, tp):
    if tp == "eg":
        dis = hg.distance(0)
    elif tp == "hnx":
        dis = []
        for i in range(1,len(hg.nodes)):
            dis.append(hg.distance(0, i))
    else:
        dis = xgi.single_source_shortest_path_length(hg, 0)
        
    return dis

def calculate_incidence_matrix(hg, tp):
    if tp == "eg":
        ic = hg.incidence_matrix
    elif tp == "hnx":
        ic = hg.incidence_matrix()
    else:
        ic = xgi.to_incidence_matrix(hg)

    return ic
    
    
    
def calculate_neighbor(hg, tp):
    if tp == "eg":
        neighbor = []
        for node in range(hg.num_v):
            neighbor.append(hg.neighbor_of_node(node))
    elif tp == "hnx":
        neighbor = []
        for node in hg.nodes:
            neighbor.append(hg.neighbors(node))
    else:
        neighbor = []
        for node in hg.nodes:
            neighbor.append(hg.nodes.neighbors(node))

    return neighbor

def main(labels_path, edges_path, dataset, metric_name, tp):
    initial_memory = memory_usage()
    print(f"初始内存使用: {initial_memory:.2f} MB")

    # 加载图并计算内存使用
    if dataset != None:
        G = load_cocitation_dataset(dataset, tp)
    else:
        G = load_dataset(labels_path, edges_path, tp)
        
    load_memory = memory_usage()
    print(f"加载超图后内存使用: {load_memory:.2f} MB")

    # 计算指标
    if metric_name == "incidence_matric":
        ic = calculate_incidence_matrix(G, tp)
    if metric_name == "degree":
        degrees = calculate_degrees(G, tp)
    if metric_name == "distance":
        distance = calculate_distance(G, tp)
    if metric_name == "neighbor":
        neighbor = calculate_neighbor(G, tp)    

    final_memory = memory_usage()
    print(f"计算指标后内存使用: {final_memory:.2f} MB")

    # 输出内存使用情况
    print(f"图加载所需内存: {load_memory - initial_memory:.2f} MB")
    print(f"计算指标所需内存: {final_memory - load_memory:.2f} MB")
    print(f"总内存使用: {final_memory - initial_memory:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算图指标并统计内存使用")
    parser.add_argument("--labels_path", type=str, help="节点标签路径")
    parser.add_argument("--edges_path", type=str, help="边的路径")
    parser.add_argument("--dataset", type=str, default=None, help="数据集名称")
    parser.add_argument("--metric", type=str, help="指标")
    parser.add_argument("--tp", type=str, help="库类型")
    
    args = parser.parse_args()
    if args.dataset != None:
        print("库：",args.tp," dataset:",args.dataset,"metric:",args.metric)
    if args.dataset != None:
        print("库：",args.tp," dataset:",args.labels_path,"metric:",args.metric)
    main(args.labels_path, args.edges_path, args.dataset, args.metric, args.tp)