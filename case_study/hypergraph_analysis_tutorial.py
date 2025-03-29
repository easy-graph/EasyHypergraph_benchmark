import os
import sys

from grgrlib.h5 import append
from statsmodels.discrete.tests.test_predict import alpha

sys.path.insert(0, '/Users/yizhihenpidehou/Desktop/fdu/eg/Easy-Graph')
import easygraph as eg
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.font_manager as fm



# 设置字体
try:
    # 尝试使用 Helvetica 字体
    helvetica_font = fm.FontProperties(family='Helvetica')
    plt.rcParams['font.family'] = helvetica_font.get_name()
except:
    # 如果 Helvetica 不可用，尝试使用 Arial 作为替代
    plt.rcParams['font.family'] = 'Arial'

from scipy.stats import pearsonr, spearmanr
import pandas as pd
colors_lst= ["#8ECFC9","#FFBE7A"]


def hg_generative_model(hg, hg2):
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    generative_hg = eg.chung_lu_hypergraph(k1 = hg.deg_v_dict, k2 = {i : val for i, val in enumerate(hg.deg_e)}, seed = 42)
    generative_hg2 = eg.chung_lu_hypergraph(k1=hg2.deg_v_dict, k2={i: val for i, val in enumerate(hg2.deg_e)}, seed=42)
    # plt.hist(eg.s_closeness(hg), bins=100)
    # plt.hist(eg.s_closeness(generative_hg), bins=100)

    # 第一个子图
    deg_val = sorted(eg.s_closeness(hg))
    cumulative_degrees = np.cumsum(deg_val) / sum(deg_val)
    axes[0].fill_between(deg_val, cumulative_degrees, color="#8ECFC9", hatch = "//", edgecolor="grey", alpha=0.7,label="House-committees")

    generative_deg_val = sorted(eg.s_closeness(generative_hg))
    generative_cumulative_degrees = np.cumsum(generative_deg_val) / sum(generative_deg_val)
    axes[0].fill_between(generative_deg_val, generative_cumulative_degrees, alpha=0.5, hatch = "...", edgecolor="grey", label = "Generative hypergraph")

    axes[0].tick_params(axis='x', labelsize=20)
    axes[0].tick_params(axis='y', labelsize=20)

    # axes[0].set_title('House_committees',fontsize = 18)
    # axes[0].set_xlabel('s-closeness centrality',fontsize = 18)
    # axes[0].set_ylabel('Cumulative frequency',fontsize = 18)

    # 第二个子图，内容与第一个子图一致
    deg_val2 = sorted(eg.s_closeness(hg2))
    cumulative_degrees2 = np.cumsum(deg_val2) / sum(deg_val)
    axes[1].fill_between(deg_val2, cumulative_degrees2, color="#FFBE7A",alpha=0.7, edgecolor="grey", hatch = "\\\\", label ="Senate-committees")

    generative_deg_val2 = sorted(eg.s_closeness(generative_hg2))
    generative_cumulative_degrees2 = np.cumsum(generative_deg_val2) / sum(generative_deg_val2)
    axes[1].fill_between(generative_deg_val2, generative_cumulative_degrees2, alpha=0.5, edgecolor="grey", hatch = "..",label = "Generative hypergraph")

    # axes[1].set_title('Senate_committees')
    # axes[1].set_xlabel('s-closeness centrality')
    # axes[1].set_ylabel('Cumulative frequency')

    axes[1].tick_params(axis='x', labelsize=20)
    axes[1].tick_params(axis='y', labelsize=20)
    # plt.grid(False)
    # axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left',frameon=False)
    # axes[1].legend(bbox_to_anchor=(1.05, 1), loc='best', frameon=False)

    plt.tight_layout()
    plt.savefig("hypergraph_generative_model.pdf")
    plt.show()


def hg_cycle_ratio(hg:eg.Hypergraph, hg2, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    hg_exp = hg.get_clique_expansion()
    hg2_exp = hg2.get_clique_expansion()
    cycle_ratio1 = eg.cycle_ratio_centrality(hg_exp)
    cycle_ratio2 = eg.cycle_ratio_centrality(hg2_exp)
    cycle_ratio1_val = list(cycle_ratio1.values())
    cycle_ratio2_val = list(cycle_ratio2.values())
    print("avg cycle_ratio1 = ",  np.mean(cycle_ratio1_val),np.max(cycle_ratio1_val))
    print("avg cycle_ratio2 = ", np.mean(cycle_ratio2_val),np.max(cycle_ratio2_val))
    beyond_avg1 = sum(num > np.mean(cycle_ratio1_val) for num in cycle_ratio1_val)
    beyond_avg2 = sum(num > np.mean(cycle_ratio2_val) for num in cycle_ratio2_val)
    print("beyond_avg1:",beyond_avg1, "beyond_avg2:",beyond_avg2)
    sorted_cycle1 = sorted(cycle_ratio1.items(), key = lambda x: x[1], reverse = True)
    sorted_cycle2 = sorted(cycle_ratio2.items(), key=lambda x: x[1], reverse=True)
    democr1 = 0
    repub1 = 0
    for i in sorted_cycle1:
        if house_committees_node_labels[i[0]] == 0 and i[1] >= np.mean(cycle_ratio1_val):
            democr1 += 1
        elif house_committees_node_labels[i[0]] == 1 and i[1] >= np.mean(cycle_ratio1_val):
            repub1+=1
    print(democr1/beyond_avg1,repub1/beyond_avg1)

    democr2 = 0
    repub2 = 0
    for i in sorted_cycle2:
        if senate_committees_node_labels[i[0]] == 0 and i[1] >= np.mean(cycle_ratio2_val):
            democr2 += 1
        elif senate_committees_node_labels[i[0]] == 1 and i[1] >= np.mean(cycle_ratio2_val):
            repub2 += 1
    print(democr2/beyond_avg2, repub2/beyond_avg2)


    axes[0].hist(cycle_ratio1.values(), bins=10, color="#8ECFC9", alpha = 0.6, hatch = "//", edgecolor="grey",label="House-committees",orientation = "horizontal")
    axes[0].hist(cycle_ratio2.values(), bins=10, color="#FFBE7A", alpha = 0.6, hatch = "\\\\",edgecolor="grey",label="Senate-committees",orientation = "horizontal")
    x = np.arange(2)
    multiplier = 0
    width = 0.25
    dict_lst  = {
        "Democrat":[democr1/beyond_avg1,democr2/beyond_avg2],
        "Republica":[repub1/beyond_avg1,repub2/beyond_avg2]
    }
    color_lst = ["#2878B5", "#C82423"]
    hatch_lst = ["||｜","---"]
    i = 0
    for attribute, measurement in dict_lst.items():
        offset = width * multiplier
        print(f"{attribute}: {measurement}")
        rects = axes[1].bar(x + offset, measurement, width, color = color_lst[i], edgecolor='black', hatch = hatch_lst[i], alpha = 0.8, label = attribute)
        axes[1].bar_label(rects, padding=2, fmt='%.3f',fontsize=16)
        multiplier += 1
        i = i+1

    # plt.xlabel('Frequency', fontsize=18)  # Set x-axis label
    # plt.ylabel('Cycle ratio', fontsize=18)
    # plt.grid(False)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='best',frameon=False)
    axes[0].tick_params(axis="x", labelbottom=False, labelsize=18)
    axes[0].tick_params(axis="y", labelbottom=False, labelsize=18)
    axes[1].tick_params(axis="x", labelbottom=False,labelsize=18)
    axes[1].tick_params(axis="y", labelbottom=False, labelsize=18)

    axes[1].set_xticks([])

    plt.savefig(save_path)
    plt.show()

def hg_coreness(hg, hg2, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    hypercoreness1 = eg.frequency_based_hypercoreness(hg)
    sorted_hypercoreness = sorted(hypercoreness1.items(), key=lambda x: x[1], reverse=True)

    hypercoreness2 = eg.frequency_based_hypercoreness(hg2)
    sorted_hypercoreness2 = sorted(hypercoreness2.items(), key=lambda x: x[1], reverse=True)
    democr_dict = {}
    repub_dict = {}
    for k, v in sorted_hypercoreness:
        if house_committees_node_labels[k] == 0:
            democr_dict[k] = v
        else:
            repub_dict[k] = v

    for k, v in sorted_hypercoreness2:
        if senate_committees_node_labels[k] == 0:
            democr_dict[k] = v
        else:
            repub_dict[k] = v


    axes[0].scatter(hypercoreness1.keys(),hypercoreness1.values(),color="#8ECFC9",alpha=0.8, edgecolor='grey', hatch = "//",label="House-committees")
    axes[0].scatter(hypercoreness2.keys(), hypercoreness2.values(),color="#FFBE7A",alpha=0.8,edgecolor='grey', hatch = "\\\\",label="Senate-committees")

    color_lst = ["#2878B5", "#C82423"]
    axes[1].scatter(repub_dict.keys(), repub_dict.values(), color=color_lst[1], edgecolors="black", alpha=0.8, label="Republica")
    axes[2].scatter(democr_dict.keys(), democr_dict.values(), color=color_lst[0], marker = 's', edgecolors = "black", alpha=0.8, label="Democrat")




    axes[0].tick_params(axis="x", labelbottom=False, labelsize=18)
    axes[0].tick_params(axis="y", labelbottom=False, labelsize=18)
    axes[1].tick_params(axis="x", labelbottom=False, labelsize=18)
    axes[1].tick_params(axis="y", labelbottom=False, labelsize=18)
    axes[2].tick_params(axis="x", labelbottom=False, labelsize=18)
    axes[2].tick_params(axis="y", labelbottom=False, labelsize=18)
    # axes[2].grid(False)
    # axes[1].grid(False)
    # axes[1].legend( loc='best', frameon=False)
    # axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left',frameon=False)

    plt.savefig(save_path)
    plt.show()

def scatter_hist(x, y, ax, ax_histx, ax_histy, color):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, color = color)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins, color=color)
    ax_histy.hist(y, bins=bins, orientation='horizontal',color=color)

def hg_closeness(hg, hg2, save_path):
    s_lst = [3,5,10]
    x = np.arange(2)  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    s_dict = {3:[], 5:[], 10:[]}
    for s in s_lst:
        hg_closeness = eg.s_closeness(hg, s=s)
        hg2_closeness = eg.s_closeness(hg2, s=s)
        hg_between = eg.s_betweenness(hg, s=s)
        hg2_between = eg.s_betweenness(hg2, s=s)
        df1 = pd.DataFrame({'closeness_centrality':hg_closeness, 'betweenness_centrality':hg_between})
        df2 = pd.DataFrame({'closeness_centrality': hg2_closeness, 'betweenness_centrality': hg2_between})
        spearman_corr1, spearman_p1 = spearmanr(df1['closeness_centrality'], df1['betweenness_centrality'])
        print(f"Spearman相关系数: {spearman_corr1:.3f}, p值: {spearman_p1:.3f}")

        spearman_corr2, spearman_p2 = spearmanr(df2['closeness_centrality'], df2['betweenness_centrality'])
        print(f"Spearman相关系数2: {spearman_corr2:.3f}, p值: {spearman_p2:.3f}")
        s_dict[s].append(spearman_corr1)
        s_dict[s].append(spearman_corr2)

    fig, ax = plt.subplots(layout='constrained')
    i = 0
    hatch_lst = ["///","\\\\",'||']
    s_dict_lst = list(s_dict.items())
    for attribute, measurement in s_dict_lst:
        offset = width * multiplier
        print(f"{attribute}: {measurement}")
        rects = ax.bar(x + offset, measurement, width, label="Interaction size >= " + str(attribute), color = ["#8ECFC9","#FFBE7A"], hatch=hatch_lst[i], edgecolor='black',alpha = 0.8)
        ax.bar_label(rects, padding=3, fmt='%.2f',fontsize=16)
        multiplier += 1
        i = i + 1

    # ax.set_ylabel('Spearman correlation')
    # ax.set_yticks(fontSize=16)

    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    ax.set_xticks([])

    # ax.set_xticks(x + width, ["House-committees","Senate-committees"])
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',frameon=False)
    plt.savefig(save_path)
    plt.show()


def hg_hyperdegree(hg, node_names, label_names, save_path, color = "olive"):
    # print(hg.deg_e)

    deg_v = hg.degree_node
    deg_v_author = {node_names[i]: deg_v[i] for i in range(len(deg_v))}
    deg_v_author = sorted(deg_v_author.items(), key=lambda x: x[1], reverse=True)
    top_deg_v_author = deg_v_author[:10]

    for u in top_deg_v_author:
        print("u:",u[0],"degree:",u[1])

    deg_val = sorted(deg_v.values())
    cumulative_degrees = np.cumsum(deg_val) / sum(deg_val)

    plt.fill_between(deg_val, cumulative_degrees, color=color,alpha = 0.3)

    plt.xticks(fontsize = 18)
    plt.yticks(fontsize=18)
    # sns.boxplot([deg_v])
    # plt.xlabel('Hyperdegree',fontsize = 18)  # Set x-axis label
    # plt.ylabel('Frequency',fontsize = 18)  # Set y-axis label

    plt.savefig(save_path)
    plt.show()


def hg_hyperedgedegree2(hg, hg2,node_names, label_names, save_path, color = "blue"):
    deg_e = hg.deg_e
    deg_e2 = hg2.deg_e
    cnt = 0
    for e in deg_e:
        if e >= 10:
            cnt += 1
    print(">=10", cnt)
    print("average hyperedge size:", sum(deg_e) / len(deg_e))
    ax1=sns.histplot(deg_e, bins=10, color=color,hatch = "//", edgecolor="grey",label="House-committees")
    ax2 = sns.histplot(deg_e2, bins=10, color="#FFBE7A",hatch = "\\\\", edgecolor="grey",label="Senate-committees")
    ax1.set_ylabel('')
    ax2.set_ylabel('')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # plt.xlabel('Hyperedge degree', fontsize=18)  # Set x-axis label
    # plt.ylabel('Frequency', fontsize=18)  # Set y-axis label
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',frameon=False)
    plt.savefig(save_path)
    plt.show()

def hg_hyperdegree2(hg1, hg2, node_names, label_names, save_path, color = "olive"):
    # 处理第一个超图 hg1
    # plt.figure(figsize=(8, 6))
    deg_v1 = hg1.degree_node
    deg_v_author1 = {node_names[i]: deg_v1[i] for i in range(len(deg_v1))}
    deg_v_author1 = sorted(deg_v_author1.items(), key=lambda x: x[1], reverse=True)
    top_deg_v_author1 = deg_v_author1[:10]

    for u in top_deg_v_author1:
        print("u:", u[0], "degree:", u[1])

    deg_val1 = sorted(deg_v1.values())
    cumulative_degrees1 = np.cumsum(deg_val1) / sum(deg_val1)

    plt.fill_between(deg_val1, cumulative_degrees1, color=color, alpha = 0.8, hatch = "//", edgecolor="grey", label = "House-committees")

    # 处理第二个超图 hg2
    deg_v2 = hg2.degree_node
    deg_v_author2 = {node_names[i]: deg_v2[i] for i in range(len(deg_v2))}
    deg_v_author2 = sorted(deg_v_author2.items(), key=lambda x: x[1], reverse=True)
    top_deg_v_author2 = deg_v_author2[:10]


    deg_val2 = sorted(deg_v2.values())
    cumulative_degrees2 = np.cumsum(deg_val2) / sum(deg_val2)

    plt.fill_between(deg_val2, cumulative_degrees2, color="#FFBE7A", alpha = 0.8, hatch = "\\\\", edgecolor="grey", label = "Senate-committees")

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # plt.legend( loc='best',frameon=False)
    # plt.grid(False)
    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    house_committees = eg.House_Committees()
    house_committees_node_names = house_committees.node_names
    house_committees_node_labels = house_committees.node_labels
    house_committees_label_names = house_committees.label_names
    democrat1 = len([i for i in house_committees_node_labels if i == 0])
    print("democrat1:",democrat1, "republic1:",len(house_committees_node_labels)-democrat1)


    senate_committees = eg.senate_committees()
    senate_committees_node_names = senate_committees.node_names
    senate_committees_node_labels = senate_committees.node_labels
    senate_committees_label_names = senate_committees.label_names
    democrat2 = len([i for i in senate_committees_node_labels if i == 0])
    print("democrat2:", democrat2, "republic2:", len(senate_committees_node_labels) - democrat2)

    # print("house_committees.node_names:",house_committees_node_names, " house_committees_label_names:",len(house_committees_node_labels))
    # print("senate_committees.node_names:",senate_committees_node_names," senate_committees_label_names:",len(senate_committees_node_labels))

    house_committees_hg = eg.Hypergraph(num_v = len(house_committees.node_names), e_list=house_committees.hyperedges)
    senate_committees_hg = eg.Hypergraph(num_v=len(senate_committees.node_names), e_list=senate_committees.hyperedges)
    # print("senate_committees_hg:",len(senate_committees_hg.e[0]),senate_committees_hg.num_v,senate_committees_hg.e[0])
    # Hyperedges
    # hg_hyperdegree2(house_committees_hg, senate_committees_hg, house_committees_node_names, house_committees_node_labels, "house_committees_hg_hyperdegree.pdf","#8ECFC9")

    # hg_hyperedgedegree2(house_committees_hg, senate_committees_hg,house_committees_node_names, house_committees_node_labels,"senate_committees_hg_hyperedge_degree.pdf","#8ECFC9")



    # hg_coreness(house_committees_hg, senate_committees_hg,"hg_coreness.pdf")
    #
    hg_cycle_ratio(house_committees_hg, senate_committees_hg, "cycle_ratio.pdf")
    #
    # hg_closeness(house_committees_hg, senate_committees_hg, "closeness.pdf")
    # hg_generative_model(house_committees_hg, senate_committees_hg,)

