import copy
import os, sys
from argparse import ArgumentParser

sys.path.insert(0, "/users/yzhpdh/EasyGraph")
import torch
import easygraph as eg

print("eg:", eg.__file__)
import dhg
import torch.nn as nn
import time
import random
import numpy as np
# from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def eg_data_preprocess(dataset, num_features):
    node_labels = dataset.node_labels
    labels_name = dataset.label_names
    hyperedges = dataset.hyperedges

    node_features = {}
    for i in range(len(node_labels)):
        node_features[i] = np.random.randn(num_features)
        # node_features[i] = np.ones(num_features)

    X = np.array([node_features[node] for node in range(len(node_labels))])
    y = np.array([node_labels[node] for node in range(len(node_labels))])

    train_nodes, test_nodes = train_test_split(list(range(len(node_labels))), test_size=0.25, random_state=42)
    # 将训练集划分为训练集和验证集
    train_nodes, val_nodes = train_test_split(train_nodes, test_size=0.25, random_state=42)

    train_mask = train_nodes
    val_mask = val_nodes
    test_mask = test_nodes
    # train_mask[train_nodes] = 1
    # val_mask[val_nodes] = 1
    # test_mask[test_nodes] = 1

    dataset = {}
    dataset["structure"] = eg.Hypergraph(num_v=len(node_labels), e_list=hyperedges)
    dataset["labels"] = torch.from_numpy(y)

    dataset["features"] = torch.from_numpy(X).float()
    dataset["train_mask"] = train_mask
    dataset["val_mask"] = val_mask
    dataset["test_mask"] = test_mask
    dataset["num_classes"] = len(labels_name)

    dataset_dhg = copy.deepcopy(dataset)
    dataset_dhg["structure"] = dhg.Hypergraph(num_v=len(node_labels), e_list=hyperedges)
    print("dataset:", dataset["structure"], " dataset2:", dataset_dhg["structure"])
    return dataset, dataset_dhg


@torch.no_grad()
def valid(data: dict, model: nn.Module):
    features, structure = data["features"], data["structure"]
    val_mask, labels = data["val_mask"], data["labels"]
    model.eval()
    outputs = model(features, structure).argmax(dim=1)
    # pred = model(data)
    correct = (outputs[val_mask] == labels[val_mask]).sum()
    # print("output:",outputs)
    # res = f1_score(labels[val_mask], outputs[val_mask],average='micro')
    # res = accuracy_score(labels[val_mask], outputs[val_mask])
    res = int(correct) / len(val_mask)
    return res


@torch.no_grad()
def valid2(data: dict, model: nn.Module):
    features, structure = data["features"], data["structure"]
    val_mask, labels = data["val_mask"], data["labels"]
    model.eval()
    outputs = model(features, structure).argmax(dim=1)
    # pred = model(data)
    correct = (outputs[val_mask] == labels[val_mask]).sum()
    # print("output:",outputs)
    # res = f1_score(labels[val_mask], outputs[val_mask],average='micro')
    # res = accuracy_score(labels[val_mask], outputs[val_mask])
    res = int(correct) / int(val_mask.sum())
    return res


@torch.no_grad()
def test(data: dict, model: nn.Module):
    features, structure = data["features"], data["structure"]
    val_mask, labels = data["test_mask"], data["labels"]
    # model.eval()
    outputs = model(features, structure).argmax(dim=1)
    # print("output:",  len(outputs[val_mask]))
    # print("labels:",labels)
    # pred = model(data)
    correct = (outputs[val_mask] == labels[val_mask]).sum()
    res = int(correct) / len(val_mask)
    return res


@torch.no_grad()
def test2(data: dict, model: nn.Module):
    features, structure = data["features"], data["structure"]
    val_mask, labels = data["test_mask"], data["labels"]
    # model.eval()
    outputs = model(features, structure).argmax(dim=1)
    # print("output:",  len(outputs[val_mask]))
    # print("labels:",labels)
    # pred = model(data)
    correct = (outputs[val_mask] == labels[val_mask]).sum()
    res = int(correct) / int(val_mask.sum())
    return res


def train(

        data: dict,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,

):
    r"""Train model for one epoch.

    Args:
        ``data`` (``dict``): The input data.
        ``model`` (``nn.Module``): The model.
        ``optimizer`` (``torch.optim.Optimizer``): The model optimizer.
        ``criterion`` (``nn.Module``): The loss function.
    """

    features, structure = data["features"], data["structure"]
    # print("features:",features.shape)

    train_mask, labels = data["train_mask"], data["labels"]
    start = time.time()

    optimizer.zero_grad()
    outputs = model(features, structure)
    loss = criterion(outputs[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    end = time.time()
    time_add = end - start
    return time_add, loss
    # print("eg one epoch time:",end-start)
    # eg_avg_time += end - start


def extra_data_preprocess(dataset):
    dataset.needs_to_load("edge_list")
    dataset.needs_to_load("features")
    dataset.needs_to_load("labels")
    labels = dataset["labels"]
    features = dataset["features"]
    dim_features = dataset["dim_features"]
    num_classes = dataset["num_classes"]
    train_nodes, test_nodes = train_test_split(list(range(len(labels))), test_size=0.25, random_state=42)
    # 将训练集划分为训练集和验证集
    train_nodes, val_nodes = train_test_split(train_nodes, test_size=0.25, random_state=42)
    train_mask = train_nodes
    val_mask = val_nodes
    test_mask = test_nodes
    edge_list = dataset["edge_list"]
    dataset_new = {}
    dataset_new["train_mask"] = train_mask
    dataset_new["val_mask"] = val_mask
    dataset_new["test_mask"] = test_mask
    dataset_new["structure"] = eg.Hypergraph(num_v=dataset["num_vertices"], e_list=edge_list)
    dataset_new["labels"] = labels
    dataset_new["num_classes"] = num_classes
    dataset_new["dim_features"] = dim_features
    dataset_new["features"] = features
    dataset_dhg = copy.deepcopy(dataset_new)
    dataset_dhg["structure"] = dhg.Hypergraph(num_v=dataset["num_vertices"], e_list=edge_list)

    return dataset_new, dataset_dhg


def acadamic_dataset_preprocess(dataset):
    # cora = eg.CocitationCora()
    dataset.needs_to_load("edge_list")
    dataset.needs_to_load("features")
    dataset.needs_to_load("labels")
    dataset.needs_to_load("train_mask")
    dataset.needs_to_load("val_mask")
    dataset.needs_to_load("test_mask")
    edge_list = dataset["edge_list"]
    # print("edge_list:",edge_list)
    labels = dataset["labels"]
    features = dataset["features"]
    dim_features = dataset["dim_features"]
    train_mask = dataset["train_mask"]
    val_mask = dataset["val_mask"]
    test_mask = dataset["test_mask"]
    num_classes = dataset["num_classes"]
    dataset_new = {}
    dataset_new["structure"] = eg.Hypergraph(num_v=dataset["num_vertices"], e_list=edge_list)
    dataset_new["labels"] = labels
    dataset_new["num_classes"] = num_classes
    dataset_new["dim_features"] = dim_features
    dataset_new["features"] = features
    # print("feature:",cora_features)
    dataset_new["train_mask"] = train_mask
    dataset_new["val_mask"] = val_mask
    dataset_new["test_mask"] = test_mask

    dataset_dhg = copy.deepcopy(dataset_new)
    dataset_dhg["structure"] = dhg.Hypergraph(num_v=dataset["num_vertices"], e_list=edge_list)

    return dataset_new, dataset_dhg


def get_model(input_feature_dim, hidden_dim, output_dim, model_name="hgnn"):
    if model_name == "hgnn":
        return (eg.HGNN(in_channels=input_feature_dim, hid_channels=hidden_dim,
                        num_classes=output_dim),
                dhg.models.HGNN(in_channels=input_feature_dim, hid_channels=hidden_dim,
                                num_classes=output_dim))
    elif model_name == "hgnnp":
        return (eg.HGNNP(in_channels=input_feature_dim, hid_channels=hidden_dim,
                         num_classes=output_dim),
                dhg.models.HGNNP(in_channels=input_feature_dim, hid_channels=hidden_dim,
                                 num_classes=output_dim))
    elif model_name == "hnhn":
        return (eg.HNHN(in_channels=input_feature_dim, hid_channels=hidden_dim,
                        num_classes=output_dim),
                dhg.models.HNHN(in_channels=input_feature_dim, hid_channels=hidden_dim,
                                num_classes=output_dim))
    elif model_name == "hypergcn":
        return (eg.HyperGCN(in_channels=input_feature_dim, hid_channels=hidden_dim,
                            num_classes=output_dim),
                dhg.models.HyperGCN(in_channels=input_feature_dim, hid_channels=hidden_dim,
                                    num_classes=output_dim)
                )
    elif model_name == "unigcn":
        return (
            eg.UniGCN(in_channels=input_feature_dim, hid_channels=hidden_dim,
                      num_classes=output_dim),
            dhg.models.UniGCN(in_channels=input_feature_dim, hid_channels=hidden_dim,
                              num_classes=output_dim)
        )
    elif model_name == "unigat":
        return (
            eg.UniGAT(in_channels=input_feature_dim, hid_channels=hidden_dim,
                      num_classes=output_dim, num_heads=args.heads),
            dhg.models.UniGAT(in_channels=input_feature_dim, hid_channels=hidden_dim,
                              num_classes=output_dim, num_heads=args.heads)
        )
    elif model_name == "unigin":
        return (
            eg.UniGIN(in_channels=input_feature_dim, hid_channels=hidden_dim,
                      num_classes=output_dim),
            dhg.models.UniGIN(in_channels=input_feature_dim, hid_channels=hidden_dim,
                              num_classes=output_dim)
        )
    elif model_name == "unisage":
        return (
            eg.UniSAGE(in_channels=input_feature_dim, hid_channels=hidden_dim,
                       num_classes=output_dim),
            dhg.models.UniSAGE(in_channels=input_feature_dim, hid_channels=hidden_dim,
                               num_classes=output_dim)
        )
    else:
        print("model does not implement")


def draw_test_curve(time, save_path):
    plt.clf()
    epochs = range(1, len(time) + 1)
    # print("acc_dhg:",acc_dhg)
    # time = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    # accuracy = np.array([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95])

    # 创建图形和坐标轴对象
    # fig, ax1 = plt.subplots()

    # 绘制第一个纵坐标轴（时间）
    # ax1.set_xlabel('Epochs')
    # ax1.set_ylabel('Time (s)')
    # print("epochs:",epochs)
    # print("time_dhg:",time_dhg)
    plt.plot(epochs, time, label='eg Training loss')
    # plt.plot(epochs, time_eg,label='EG Training loss')
    # ax1.tick_params(axis='y')
    plt.title('Training Time Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Time (s)')
    plt.legend()
    # ax1.legend(['Time'], loc='upper left')

    # 设置图形标题
    # plt.title('Double Y-Axis Plot')
    plt.savefig(save_path)


def draw_loss_curve(loss1, loss2, save_path):
    plt.clf()
    epochs = range(1, len(loss1) + 1)
    plt.plot(epochs, loss1, 'b', label='EG Training loss')
    plt.plot(epochs, loss2, 'r', label='DHG Training loss')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.xticks(range(1, len(loss1) + 1, 1))
    # 显示网格
    plt.grid(True)

    plt.savefig(save_path)
    # 显示图表
    # plt.show()


def dhg_model_train(dhg_model, dhg_dataset, epoch=100):
    loss_fn = nn.CrossEntropyLoss()
    dhg_model.train()
    dhg_val = []
    dhg_loss = []
    dhg_total_time = 0
    dhg_time = []
    optimizer = torch.optim.Adam(dhg_model.parameters(), lr=args.lr, weight_decay=args.decay)
    for i in range(epoch):
        t, loss = train(data=dhg_dataset, model=dhg_model, optimizer=optimizer, criterion=loss_fn)
        dhg_total_time += t
        dhg_loss.append(loss.detach().numpy())
        dhg_time.append(t)
        if dataset_name not in acadamic_dataset_lst:
            f1 = valid(data=dhg_dataset, model=dhg_model)
        else:
            f1 = valid2(data=dhg_dataset, model=dhg_model)
        dhg_val.append(f1)

    return dhg_total_time, dhg_val, dhg_loss, dhg_time


def eg_model_train(eg_model, eg_dataset, epoch=100):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(eg_model.parameters(), lr=args.lr, weight_decay=args.decay)
    eg_model.train()
    eg_val = []
    eg_loss = []
    eg_time = []
    eg_total_time = 0
    for i in range(epoch):
        t, loss = train(data=eg_dataset, model=eg_model, optimizer=optimizer, criterion=loss_fn)
        eg_total_time += t
        eg_loss.append(loss.detach().numpy())
        eg_time.append(t)
        if dataset_name not in acadamic_dataset_lst:
            f1 = valid(data=eg_dataset, model=eg_model)
        else:
            f1 = valid2(data=eg_dataset, model=eg_model)
        eg_val.append(f1)

    return eg_total_time, eg_val, eg_loss, eg_time


if __name__ == '__main__':

    # 1. init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='walmart_trips')
    parser.add_argument("--model", type=str, default='hgnn')
    parser.add_argument("--log_output_path", type=str, default='./log.txt')
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--hid", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decay", type=float, default=0, help="weight decay")
    parser.add_argument("--heads", type=int, default=8, help="heads number")
    parser.add_argument("--features", type=int, default=100, help="features number")
    args = parser.parse_args()
    seed = args.seed
    seed_everything(seed)
    dataset_name = args.dataset

    num_features = args.features  # 假设每个节点的特征维度


    #  2. load dataset

    acadamic_dataset_lst = ["cocitation_cora", "cocitation_citeseer", "cocitation_pubmed", "coauthorshipDBLP",
                            "coauthorshipCora", "Tencent2k"]

    #  2. load dataset

    if dataset_name == "walmart_trips":
        dataset = eg.walmart_trips()

    elif dataset_name == "cocitation_cora":
        dataset = eg.CocitationCora()

    elif dataset_name == "cocitation_citeseer":
        dataset = eg.CocitationCiteseer()

    elif dataset_name == "senate_committees":
        dataset = eg.senate_committees()

    elif dataset_name == "house_committees":
        dataset = eg.House_Committees()

    elif dataset_name == "trivago_clicks":
        dataset = eg.trivago_clicks()

    elif dataset_name == "cocitation_pubmed":
        dataset = eg.CocitationPubmed()

    elif dataset_name == "coauthorshipDBLP":
        dataset = dhg.data.CoauthorshipDBLP()

    elif dataset_name == "coauthorshipCora":
        dataset = dhg.data.CoauthorshipCora()

    elif dataset_name == "news":
        dataset = dhg.data.News20()

    elif dataset_name == "yelp":
        dataset = dhg.data.YelpRestaurant()

    elif dataset_name == "Tencent2k":
        dataset = dhg.data.Tencent2k()

    # 3. preprocess dataset
    if dataset_name in ["trivago_clicks", "walmart_trips"]:
        dataset, dhg_dataset = eg_data_preprocess(dataset, num_features)
    elif dataset_name in ["news", "yelp"]:
        dataset, dhg_dataset = extra_data_preprocess(dataset)
    else:
        dataset, dhg_dataset = acadamic_dataset_preprocess(dataset)


    # train

    model_name = args.model
    nodes_num = dataset["structure"].num_v
    print("nodes_num:", nodes_num)
    if dataset_name not in acadamic_dataset_lst and dataset_name not in ["yelp", "news"]:
        eg_model, dhg_model = get_model(model_name=model_name, input_feature_dim=num_features, hidden_dim=args.hid,
                                        output_dim=dataset["num_classes"])

    else:
        eg_model, dhg_model = get_model(model_name=model_name, input_feature_dim=dataset["dim_features"],
                                        hidden_dim=args.hid, output_dim=dataset["num_classes"])

    loss_fn1 = nn.CrossEntropyLoss()
    loss_fn2 = nn.CrossEntropyLoss()


    rep_dhg_total_time = 0
    rep_eg_total_time = 0
    eg_time = []
    dhg_time_lst = []
    eg_loss = []
    dhg_loss = []
    eg_val = []
    dhg_val = []
    epoch = args.epoch
    with open(args.log_output_path, 'w') as f:
        f.write(f"DHG Start train on dataset {dataset_name}, model:{model_name}\n")

    if dataset_name not in acadamic_dataset_lst:
        rep_time = 3
    else:
        rep_time = 3
    for rep in range(rep_time):
        dhg_total_time, dhg_val, dhg_loss, dhg_time = dhg_model_train(dhg_model=copy.deepcopy(dhg_model),
                                                                      dhg_dataset=dhg_dataset, epoch=epoch)
        rep_dhg_total_time += dhg_total_time
        dhg_time_lst.append(dhg_total_time)

    print("dhg_total_time:", rep_dhg_total_time / rep_time)


    from torch import tensor

    dhg_time_lst = tensor(dhg_time_lst)
    print("dhg_time_lst:", dhg_time_lst)
    with open(args.log_output_path, 'a+') as f:
        f.write(f'AVG TIME: {float(dhg_time_lst.mean()):.3f} ± {float(dhg_time_lst.std()):.3f} ')


