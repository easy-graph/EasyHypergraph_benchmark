import torch
import torch.nn as nn
import numpy as np
import sys
import torch.nn.functional as F
sys.path.insert(0, '/Users/yizhihenpidehou/Desktop/fdu/eg/Easy-Graph')
import easygraph as eg
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.font_manager as fm
import os, random
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from sklearn.metrics import precision_score, recall_score, f1_score

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# seed_everything(10)
# 设置字体
try:
    # 尝试使用 Helvetica 字体
    helvetica_font = fm.FontProperties(family='Helvetica')
    plt.rcParams['font.family'] = helvetica_font.get_name()
except:
    # 如果 Helvetica 不可用，尝试使用 Arial 作为替代
    plt.rcParams['font.family'] = 'Arial'

class GCN(torch.nn.Module):
    def __init__(self, dataset,hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(dataset["features"].shape[1], hidden_dim)
        self.conv2 = GCNConv(hidden_dim, dataset["num_classes"])

    def forward(self, data):
        x, edge_index = data["features"], data["edge_index"]

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model_lst = ["HGNN","HGNN$^{+}$","HNHN","HyperGCN","UniGCN","UniGAT"]
dataset_lst = [ ("cora", eg.CocitationCora()), ("citeseer", eg.CocitationCiteseer()), ("pubmed", eg.CocitationPubmed()),
    ("house_committees",eg.House_Committees()),("senate_committees",eg.senate_committees())]


def get_model(input_feature_dim, hidden_dim, output_dim, model_name):
    if model_name == "HGNN":
        return eg.HGNN(in_channels=input_feature_dim, hid_channels=hidden_dim,
                        num_classes=output_dim)
    elif model_name == "HGNN$^{+}$":
        return eg.HGNNP(in_channels=input_feature_dim, hid_channels=hidden_dim,
                         num_classes=output_dim)
    elif model_name == "HNHN":
        return eg.HNHN(in_channels=input_feature_dim, hid_channels=hidden_dim,
                        num_classes=output_dim)
    elif model_name == "HyperGCN":
        return eg.HyperGCN(in_channels=input_feature_dim, hid_channels=hidden_dim,
                            num_classes=output_dim)
    elif model_name == "UniGCN":
        return eg.UniGCN(in_channels=input_feature_dim, hid_channels=hidden_dim,
                      num_classes=output_dim)
    elif model_name == "UniGAT":
        return eg.UniGAT(in_channels=input_feature_dim, hid_channels=hidden_dim,
                      num_classes=output_dim, num_heads=4)
    elif model_name == "UniGIN":
        return  eg.UniGIN(in_channels=input_feature_dim, hid_channels=hidden_dim,
                      num_classes=output_dim),

    elif model_name == "UniSAGE":
        return eg.UniSAGE(in_channels=input_feature_dim, hid_channels=hidden_dim,
                       num_classes=output_dim),


    else:
        print("model does not implement")


def preprocess_citation(dataset):
    """Preprocess for HGNN model training.

        Preprocess dataset and model

        Returns:
            bool: dataset and model
    """


    hg = eg.Hypergraph(num_v=dataset["num_vertices"], e_list=dataset["edge_list"])
    clique_graph = hg.get_clique_expansion()
    # hg_exp = eg.Hypergraph(num_v=dataset["num_vertices"], e_list=clique_graph.e[0])

    dataset_hg = {}
    dataset_gnn = {}


    # 将列表转换为 PyTorch 张量
    edge_tensor = torch.tensor(clique_graph.e[0], dtype=torch.long)
    # 调整形状为 [2, num_messages]
    edge_index = edge_tensor.t().contiguous()
    dataset_gnn["edge_index"] = edge_index
    dataset_gnn["features"] = dataset["features"]
    dataset_gnn["train_mask"] = dataset["train_mask"]
    dataset_gnn["val_mask"] = dataset["val_mask"]
    dataset_gnn["test_mask"] = dataset["test_mask"]
    dataset_gnn["num_classes"] = dataset["num_classes"]
    dataset_gnn["node_labels"] = dataset["labels"]

    dataset_hg["structure"] = hg
    dataset_hg["node_labels"] = dataset["labels"]
    dataset_hg["features"] = dataset["features"]
    dataset_hg["train_mask"] = dataset["train_mask"]
    dataset_hg["val_mask"] = dataset["val_mask"]
    dataset_hg["test_mask"] = dataset["test_mask"]
    dataset_hg["num_classes"] = dataset["num_classes"]

    return dataset_hg, dataset_gnn

def preprocess(dataset):
    """Preprocess for HGNN model training.

        Preprocess dataset and model

        Returns:
            bool: dataset and model
    """


    node_labels = dataset.node_labels
    labels_name = dataset.label_names
    hyperedges = dataset.hyperedges
    hg = eg.Hypergraph(num_v=len(node_labels), e_list=hyperedges)
    clique_graph = hg.get_clique_expansion()
    num_features = 100
    node_features = {}
    # hypercore = eg.cycle_ratio_centrality(clique_graph)
    hypernode_features ={}
    for i in range(len(node_labels)):
        node_features[i] = np.random.randn(num_features)


    X = np.array([node_features[node] for node in range(len(node_labels))])
    X2 = np.array([hypernode_features[node] for node in range(len(node_labels))])
    y = np.array([node_labels[node] for node in range(len(node_labels))])

    train_nodes, test_nodes = train_test_split(list(range(len(node_labels))), test_size=0.1)
    # 将训练集划分为训练集和验证集
    train_nodes, val_nodes = train_test_split(train_nodes, test_size=0.1)
    train_mask = train_nodes
    val_mask = val_nodes
    test_mask = test_nodes

    dataset = {}
    dataset_gnn = {}


    hg_exp = eg.Hypergraph(num_v=len(node_labels), e_list=clique_graph.e[0])
    # 将列表转换为 PyTorch 张量
    edge_tensor = torch.tensor(clique_graph.e[0], dtype=torch.long)
    # 调整形状为 [2, num_messages]
    edge_index = edge_tensor.t().contiguous()
    dataset_gnn["edge_index"] = edge_index
    dataset_gnn["features"] = torch.from_numpy(X).float()
    dataset_gnn["train_mask"] = torch.tensor(train_mask)
    dataset_gnn["val_mask"] = torch.tensor(val_mask)
    dataset_gnn["test_mask"] = torch.tensor(test_mask)
    dataset_gnn["num_classes"] = len(labels_name)
    dataset_gnn["node_labels"] = torch.tensor(node_labels)
    hg = eg.Hypergraph(num_v=len(node_labels), e_list=hg_exp.e[0])

    dataset["structure"] = hg
    dataset["node_labels"] = node_labels
    dataset["features"] = torch.from_numpy(X2).float()
    dataset["train_mask"] = train_mask
    dataset["val_mask"] = val_mask
    dataset["test_mask"] = test_mask
    dataset["num_classes"] = len(labels_name)

    return dataset, dataset_gnn

@torch.no_grad()
def valid(model: nn.Module, data: dict):
    features, structure = data["features"], data["structure"]
    val_mask, labels = data["val_mask"], data["labels"]
    model.eval()
    outputs = model(features, structure).argmax(dim=1)
    correct = (outputs[val_mask] == labels[val_mask]).sum()
    acc = int(correct) / len(val_mask)
    return acc


def train(
        data: dict,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
):

    features, structure = data["features"], data["structure"]
    train_mask, labels = data["train_mask"], data["node_labels"]
    optimizer.zero_grad()
    outputs = model(features, structure)
    # print("outputs:", outputs)
    loss = criterion(outputs[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    return loss,outputs

@torch.no_grad()
def test(model: nn.Module, data: dict):

    model.eval()
    features, structure = data["features"], data["structure"]
    test_mask, labels = data["test_mask"], data["node_labels"]

    outputs_embeddings = model(features, structure)
    outputs = outputs_embeddings.argmax(dim=1)
    test_outputs = outputs[test_mask]
    test_labels = labels[test_mask]
    # test_outputs = outputs
    # test_labels = labels
    correct = (test_outputs == test_labels).sum()
    acc = int(correct) / len(test_outputs)
    # print("test_outputs:",test_outputs)
    # print("test_labels:",test_labels)
    precision = precision_score(test_labels, test_outputs, average="weighted")
    recall = recall_score(test_labels, test_outputs, average="weighted")
    f1 = f1_score(test_labels, test_outputs, average="weighted")
    return acc, precision, recall, f1, outputs_embeddings


def draw_loss_curve(loss1, save_path = "loss_pic.png"):
    plt.clf()
    epochs = range(1, len(loss1) + 1)
    plt.plot(epochs, loss1, 'b', label='EG Training loss')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

def draw_acc_curve(acc_lst, color, hatch, save_path = "acc_pic.pdf"):
    plt.clf()
    category_lst = ["HGNN","HGNN$^{+}$","HNHN","HyperGCN", "UniGCN","UniGAT"]
    rects = plt.bar(category_lst, acc_lst, color = color, hatch = hatch)
    plt.bar_label(rects, padding=3, fmt='%.4f')
    plt.title('Training accuracy')
    # plt.xlabel('Models')
    # plt.ylabel('Accuracy')
    # plt.legend()
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

def visualize_embedding(labels, embeddings, colors = None, save_path = "embedding_pic.png"):

    # 使用 t-SNE 进行降维
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    # 可视化降维后的数据
    plt.figure(figsize=(10, 8))
    # colors = ["red","blue"]
    # 根据不同的标签使用不同的颜色进行绘制
    for i, label in enumerate(np.unique(labels)):
        indices = labels == label
        if colors is not None:
            plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], color = colors[i], label=f'Class {label}')
        else:
            plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=f'Class {label}')

    # plt.title(model_name,fontsize=24)
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # plt.legend()
    plt.savefig(save_path)
    # plt.show()


def train_GCN(preprocess_dataset_gnn, model):
    print("Training GCN")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay = 0.0005)
    model.train()
    epoch = 200
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epoch):
        optimizer.zero_grad()
        out = model(preprocess_dataset_gnn)
        # loss = F.nll_loss(out[preprocess_dataset_gnn["train_mask"]], preprocess_dataset_gnn["node_labels"][preprocess_dataset_gnn["train_mask"]])
        loss  = criterion(out[preprocess_dataset_gnn["train_mask"]], preprocess_dataset_gnn["node_labels"][preprocess_dataset_gnn["train_mask"]])
        # print(f"epoch: {epoch}, loss : {loss}")
        loss.backward()
        optimizer.step()
    model.eval()
    output_embedding = model(preprocess_dataset_gnn)
    pred = output_embedding.argmax(dim=1)
    correct = (pred[preprocess_dataset_gnn["test_mask"]] == preprocess_dataset_gnn["node_labels"][preprocess_dataset_gnn["test_mask"]]).sum()
    precision = precision_score(preprocess_dataset_gnn["node_labels"][preprocess_dataset_gnn["test_mask"]],
                                pred[preprocess_dataset_gnn["test_mask"]], average="weighted")
    recall = recall_score(preprocess_dataset_gnn["node_labels"][preprocess_dataset_gnn["test_mask"]],
                          pred[preprocess_dataset_gnn["test_mask"]],average="weighted")


    acc = int(correct) / len(preprocess_dataset_gnn["test_mask"])
    f1= f1_score(preprocess_dataset_gnn["node_labels"][preprocess_dataset_gnn["test_mask"]] ,pred[preprocess_dataset_gnn["test_mask"]], average="weighted"  )
    print(f'Accuracy: {acc:.4f}','Precision:',precision,'Recall:',recall, "f1:",f1)

    model_name = "gnn"
    dataset_name = dataset[0]
    visualize_embedding(preprocess_dataset_gnn["node_labels"], output_embedding.detach().numpy(), None, model_name+"_"+dataset_name+".pdf")

    return acc, precision, recall, f1

def train_hnn(dataset_process, dataset_name, model):
    print("Training HNN")
    epoch = 200
    lr = 0.01
    # loss_fn = F.nll_loss
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 0.0005)
    model.train()
    for i in range(epoch):
        loss, train_output = train(data = dataset_process, model = model, optimizer=optimizer, criterion=loss_fn)
        # val_acc = valid(model = model, data = dataset_process)
        # print(f"epoch: {i}, valid accuracy : {val_acc}, loss : {loss}")

    print("Training finish!")

    test_acc, precision, recall, f1, outputs = test(model = model, data=dataset_process)
    print("test accuracy:", test_acc,"precision:", precision,"recall:", recall, "f1:", f1)
    visualize_embedding(dataset_process["node_labels"], outputs.detach().numpy(), None, model_name+"_"+dataset_name+".pdf")
    return test_acc, precision, recall, f1

if __name__ == "__main__":
    hidden_dict = {"cora":64, "citeseer":64, "pubmed":64 }

    cocitation_dataset_lst = [ ("citeseer", eg.CocitationCiteseer()),("cora", eg.CocitationCora()),
                   ("pubmed", eg.CocitationPubmed())]


    for dataset in cocitation_dataset_lst:
        print("dataset:", dataset)
        acc_gnn_lst = []
        precision_gnn_lst = []
        recall_gnn_lst = []
        f1_gnn_lst = []
        acc_hgnn_dict = {"HGNN": [], "HGNN$^{+}$": [], "HyperGCN": [], "HNHN": [], "UniGCN": [], "UniGAT": []}
        precision_hgnn_dict = {"HGNN": [], "HGNN$^{+}$": [], "HyperGCN": [], "HNHN": [], "UniGCN": [], "UniGAT": []}
        recall_hgnn_dict = {"HGNN": [], "HGNN$^{+}$": [], "HyperGCN": [], "HNHN": [], "UniGCN": [], "UniGAT": []}
        f1_hgnn_dict = {"HGNN": [], "HGNN$^{+}$": [], "HyperGCN": [], "HNHN": [], "UniGCN": [], "UniGAT": []}
        if dataset in cocitation_dataset_lst:
            preprocess_dataset, preprocess_dataset_gnn = preprocess_citation(dataset[1])

        else:
            preprocess_dataset, preprocess_dataset_gnn = preprocess(dataset[1])
        for round in range(3):
            gcn_model = GCN(preprocess_dataset_gnn, hidden_dict[dataset[0]])
            acc_gnn, precison_gnn, recall_gnn, f1_gnn = train_GCN(preprocess_dataset_gnn, gcn_model)
            acc_gnn_lst.append(acc_gnn)
            precision_gnn_lst.append(precison_gnn)
            recall_gnn_lst.append(recall_gnn)
            f1_gnn_lst.append(f1_gnn)

            for model_name in model_lst:
                print("model_name:",model_name)

                hnn_model = get_model(preprocess_dataset["features"].shape[1], hidden_dict[dataset[0]], preprocess_dataset["num_classes"], model_name)
                acc_hgnn, precison_hgnn, recall_hgnn, f1_hgnn = train_hnn(preprocess_dataset, dataset[0], hnn_model)

                acc_hgnn_dict[model_name].append(acc_hgnn)
                precision_hgnn_dict[model_name].append(precison_hgnn)
                recall_hgnn_dict[model_name].append(recall_hgnn)
                f1_hgnn_dict[model_name].append(f1_hgnn)

        print(f"acc_gnn_lst:{np.mean(acc_gnn_lst)} ± {np.std(acc_gnn_lst)}")
        print(f"precison_gnn_lst:{np.mean(precision_gnn_lst)} ± {np.std(precision_gnn_lst)}")
        print(f"recall_gnn_lst:{np.mean(recall_gnn_lst)} ± {np.std(recall_gnn_lst)}")
        print(f"f1_gnn_lst:{np.mean(f1_gnn_lst)} ± {np.std(f1_gnn_lst)}")

        for k, v in acc_hgnn_dict.items():
            print(f"model: {k} acc_hgnn:{np.mean(v)} ± {np.std(v)}")

        for k, v in precision_hgnn_dict.items():
            print(f"model: {k} precision_hgnn:{np.mean(v)} ± {np.std(v)}")

        for k, v in recall_hgnn_dict.items():
            print(f"model: {k} recall_hgnn:{np.mean(v)} ± {np.std(v)}")

        for k, v in f1_hgnn_dict.items():
            print(f"model: {k} f1_hgnn:{np.mean(v)} ± {np.std(v)}")
