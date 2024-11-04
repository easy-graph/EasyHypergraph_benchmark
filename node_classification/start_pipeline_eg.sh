#!/bin/bash

# 运行 Python 3 程序并获取其进程 ID

#  features维度对比
# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512  --features 100 --log_output_path "./dimensions/walmart_trips_hgnn_features_100_eg_log.txt"

# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512  --features 10 --log_output_path "./dimensions/walmart_trips_hgnn_features_10_eg_log.txt"
# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512  --features 200 --log_output_path "./dimensions/walmart_trips_hgnn_features_200_eg_log.txt"
# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512  --features 500 --log_output_path "./dimensions/walmart_trips_hgnn_features_500_eg_log.txt"
# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512  --features 1000 --log_output_path "./dimensions/walmart_trips_hgnn_features_1000_eg_log.txt"


# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512 --features 10   --log_output_path "./dimensions/walmart_trips_hgnn_features_10_dhg_log.txt"
# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512 --features 100   --log_output_path "./dimensions/walmart_trips_hgnn_features_100_dhg_log.txt"
# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512  --features 200 --log_output_path "./dimensions/walmart_trips_hgnn_features_200_dhg_log.txt"
# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512   --features 500 --log_output_path "./dimensions/walmart_trips_hgnn_features_500_dhg_log.txt"
# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512  --features 1000 --log_output_path "./dimensions/walmart_trips_hgnn_features_1000_dhg_log.txt"


# check 24.8.17 learning rate & hidden

# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset yelp --lr 0.1 --hid 256  --log_output_path "./results/yelp_hgnn_01_256_eg_log.txt"
# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset yelp --lr 0.01 --hid 256  --log_output_path "./results/yelp_hgnn_001_256_eg_log.txt"
# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset yelp --lr 0.005 --hid 256  --log_output_path "./results/yelp_hgnn_005_256_eg_log.txt"
# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset yelp --lr 0.001 --hid 256  --log_output_path "./results/yelp_hgnn_0001_256_eg_log.txt"


# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset yelp --lr 0.1 --hid 256  --log_output_path "./results/yelp_hgnn_01_256_dhg_log.txt"
# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset yelp --lr 0.01 --hid 256  --log_output_path "./results/yelp_hgnn_001_256_dhg_log.txt"
# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset yelp --lr 0.005 --hid 256  --log_output_path "./results/yelp_hgnn_005_256_dhg_log.txt"
# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset yelp --lr 0.001 --hid 256  --log_output_path "./results/yelp_hgnn_0001_256_dhg_log.txt"


# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset yelp --lr 0.001 --hid 128  --log_output_path "./results/yelp_hgnn_128_eg_log.txt"
# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset yelp --lr 0.001 --hid 128  --log_output_path "./results/yelp_hgnn_128_dhg_log.txt"

# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset yelp --lr 0.001 --hid 256  --log_output_path "./results/yelp_hgnn_256_eg_log.txt"
# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset yelp --lr 0.001 --hid 256  --log_output_path "./results/yelp_hgnn_256_dhg_log.txt"

# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset yelp --lr 0.001 --hid 512  --log_output_path "./results/yelp_hgnn_512_eg_log.txt"
# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset yelp --lr 0.001 --hid 512  --log_output_path "./results/yelp_hgnn_512_dhg_log.txt"


# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset trivago_clicks --lr 0.1 --hid 512  --log_output_path "./results/trivago_clicks_hgnn_01_512_eg_log.txt"
# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset trivago_clicks --lr 0.01 --hid 512  --log_output_path "./results/trivago_clicks_hgnn_001_512_eg_log.txt"
# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset trivago_clicks --lr 0.005 --hid 512  --log_output_path "./results/trivago_clicks_hgnn_005_512_eg_log.txt"
# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512  --log_output_path "./results/trivago_clicks_hgnn_0001_512_eg_log.txt"


# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset trivago_clicks --lr 0.1 --hid 512  --log_output_path "./results/trivago_clicks_hgnn_01_512_dhg_log.txt"
# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset trivago_clicks --lr 0.01 --hid 512  --log_output_path "./results/trivago_clicks_hgnn_001_512_dhg_log.txt"
# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset trivago_clicks --lr 0.005 --hid 512  --log_output_path "./results/trivago_clicks_hgnn_005_512_dhg_log.txt"
# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512  --log_output_path "./results/trivago_clicks_hgnn_0001_512_dhg_log.txt"


# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 128  --log_output_path "./results/trivago_clicks_hgnn_128_eg_log.txt"
# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 128  --log_output_path "./results/trivago_clicks_hgnn_128_dhg_log.txt"

# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 256  --log_output_path "./results/trivago_clicks_hgnn_256_eg_log.txt"
# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 256  --log_output_path "./results/trivago_clicks_hgnn_256_dhg_log.txt"

# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512  --log_output_path "./results/trivago_clicks_hgnn_512_eg_log.txt"
# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512  --log_output_path "./results/trivago_clicks_hgnn_512_dhg_log.txt"


# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.1 --hid 512  --log_output_path "./results/walmart_trips_hgnn_01_512_eg_log.txt"
# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.01 --hid 512  --log_output_path "./results/walmart_trips_hgnn_001_512_eg_log.txt"
# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.005 --hid 512  --log_output_path "./results/walmart_trips_hgnn_005_512_eg_log.txt"
# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512  --log_output_path "./results/walmart_trips_hgnn_0001_512_eg_log.txt"


# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.1 --hid 512  --log_output_path "./results/walmart_trips_hgnn_01_512_dhg_log.txt"
# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.01 --hid 512  --log_output_path "./results/walmart_trips_hgnn_001_512_dhg_log.txt"
# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.005 --hid 512  --log_output_path "./results/walmart_trips_hgnn_005_512_dhg_log.txt"
# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512  --log_output_path "./results/walmart_trips_hgnn_0001_512_dhg_log.txt"


# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 128  --log_output_path "./results/walmart_trips_hgnn_128_eg_log.txt"
# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 128  --log_output_path "./results/walmart_trips_hgnn_128_dhg_log.txt"

# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 256  --log_output_path "./results/walmart_trips_hgnn_256_eg_log.txt"
# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 256  --log_output_path "./results/walmart_trips_hgnn_256_dhg_log.txt"

# python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512  --log_output_path "./results/walmart_trips_hgnn_512_eg_log.txt"
# python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512  --log_output_path "./results/walmart_trips_hgnn_512_dhg_log.txt"


###### test on different datasets

# check 24.8.17
python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset cocitation_cora --lr 0.001 --hid 512 --log_output_path "./results/cocitation_cora_hgnn_eg_log.txt"
python3 training_pipeline_eg.py  --model hgnnp --epoch 100 --dataset cocitation_cora --lr 0.001 --hid 512 --log_output_path "./results/cocitation_cora_hgnnp_eg_log.txt"
python3 training_pipeline_eg.py  --model hnhn --epoch 100 --dataset cocitation_cora --lr 0.001 --hid 512 --log_output_path "./results/cocitation_cora_hnhn_eg_log.txt"
python3 training_pipeline_eg.py  --model hypergcn --epoch 100 --dataset cocitation_cora --lr 0.001 --hid 64 --log_output_path "./results/cocitation_cora_hypergcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigcn --epoch 100 --dataset cocitation_cora --lr 0.001 --hid 512 --log_output_path "./results/cocitation_cora_unigcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigin --epoch 100 --dataset cocitation_cora --lr 0.001 --hid 512 --log_output_path "./results/cocitation_cora_unigin_eg_log.txt"
# python3 training_pipeline_eg.py  --model unisage --epoch 100 --dataset cocitation_cora --lr 0.001 --hid 512 --log_output_path "./results/cocitation_cora_unisage_eg_log.txt"
python3 training_pipeline_eg.py  --model unigat --epoch 100 --dataset cocitation_cora --lr 0.001 --hid 512 --heads 8 --log_output_path "./results/cocitation_cora_unigat_eg_log.txt"

# check 24.8.17
python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset coauthorshipCora --lr 0.001 --hid 128 --log_output_path "./results/coauthorshipCora_hgnn_eg_log.txt"
python3 training_pipeline_eg.py  --model hgnnp --epoch 100 --dataset coauthorshipCora --lr 0.001 --hid 512 --log_output_path "./results/coauthorshipCora_hgnnp_eg_log.txt"
python3 training_pipeline_eg.py  --model hnhn --epoch 100 --dataset coauthorshipCora --lr 0.001 --hid 512 --log_output_path "./results/coauthorshipCora_hnhn_eg_log.txt"
python3 training_pipeline_eg.py  --model hypergcn --epoch 100 --dataset coauthorshipCora --lr 0.01 --hid 64 --decay 0.000001 --log_output_path "./results/coauthorshipCora_hypergcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigcn --epoch 100 --dataset coauthorshipCora --lr 0.001 --hid 512 --log_output_path "./results/coauthorshipCora_unigcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigin --epoch 100 --dataset coauthorshipCora --lr 0.001 --hid 512 --log_output_path "./results/coauthorshipCora_unigin_eg_log.txt"
# python3 training_pipeline_eg.py  --model unisage --epoch 100 --dataset coauthorshipCora --lr 0.001 --hid 512 --log_output_path "./results/coauthorshipCora_unisage_eg_log.txt"
python3 training_pipeline_eg.py  --model unigat --epoch 100 --dataset coauthorshipCora --lr 0.001 --hid 512 --heads 4 --log_output_path "./results/coauthorshipCora_unigat_eg_log.txt"

# check 24.8.17
python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset cocitation_citeseer --lr 0.001 --hid 256 --log_output_path "./results/cocitation_citeseer_hgnn_eg_log.txt"
python3 training_pipeline_eg.py  --model hgnnp --epoch 100 --dataset cocitation_citeseer --lr 0.001 --hid 256 --log_output_path "./results/cocitation_citeseer_hgnnp_eg_log.txt"
python3 training_pipeline_eg.py  --model hnhn --epoch 100 --dataset cocitation_citeseer --lr 0.001 --hid 256  --log_output_path "./results/cocitation_citeseer_hnhn_eg_log.txt"
python3 training_pipeline_eg.py  --model hypergcn --epoch 100 --dataset cocitation_citeseer --lr 0.01 --hid 64 --decay 0.000001 --log_output_path "./results/cocitation_citeseer_hypergcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigcn --epoch 100 --dataset cocitation_citeseer --lr 0.001 --hid 128 --log_output_path "./results/cocitation_citeseer_unigcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigin --epoch 100 --dataset cocitation_citeseer --lr 0.001 --hid 128 --log_output_path "./results/cocitation_citeseer_unigin_eg_log.txt"
# python3 training_pipeline_eg.py  --model unisage --epoch 100 --dataset cocitation_citeseer --lr 0.01 --hid 512 --drop_rate 0.5 --log_output_path "./results/cocitation_citeseer_unisage_eg_log.txt"
python3 training_pipeline_eg.py  --model unigat --epoch 100 --dataset cocitation_citeseer --lr 0.001 --hid 128 --heads 1 --log_output_path "./results/cocitation_citeseer_unigat_eg_log.txt"

# check 24.8.17
python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset cocitation_pubmed --lr 0.001 --hid 512 --log_output_path "./results/cocitation_pubmed_hgnn_eg_log.txt"
python3 training_pipeline_eg.py  --model hgnnp --epoch 100 --dataset cocitation_pubmed --lr 0.001 --hid 512 --log_output_path "./results/cocitation_pubmed_hgnnp_eg_log.txt"
python3 training_pipeline_eg.py  --model hnhn --epoch 100 --dataset cocitation_pubmed --lr 0.001 --hid 512 --log_output_path "./results/cocitation_pubmed_hnhn_eg_log.txt"
python3 training_pipeline_eg.py  --model hypergcn --epoch 100 --dataset cocitation_pubmed --lr 0.01 --hid 64  --decay 0.000001 --log_output_path "./results/cocitation_pubmed_hypergcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigcn --epoch 100 --dataset cocitation_pubmed --lr 0.001 --hid 128 --log_output_path "./results/cocitation_pubmed_unigcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigin --epoch 100 --dataset cocitation_pubmed --lr 0.001 --hid 128 --log_output_path "./results/cocitation_pubmed_unigin_eg_log.txt"
# python3 training_pipeline_eg.py  --model unisage --epoch 100 --dataset cocitation_pubmed --lr 0.001 --hid 128 --log_output_path "./results/cocitation_pubmed_unisage_eg_log.txt"
python3 training_pipeline_eg.py  --model unigat --epoch 100 --dataset cocitation_pubmed --lr 0.001 --hid 128 --heads 1 --log_output_path "./results/cocitation_pubmed_unigat_eg_log.txt"

# check 24.8.17
python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 256 --log_output_path "./results/coauthorshipDBLP_hgnn_eg_log.txt"
python3 training_pipeline_eg.py  --model hgnnp --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 256 --log_output_path "./results/coauthorshipDBLP_hgnnp_eg_log.txt"
python3 training_pipeline_eg.py  --model hnhn --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 512 --log_output_path "./results/coauthorshipDBLP_hnhn_eg_log.txt"
python3 training_pipeline_eg.py  --model hypergcn --epoch 100 --dataset coauthorshipDBLP --lr 0.01 --hid 64  --decay 0.000001 --log_output_path "./results/coauthorshipDBLP_hypergcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigcn --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 256 --log_output_path "./results/coauthorshipDBLP_unigcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigin --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 256 --log_output_path "./results/coauthorshipDBLP_unigin_eg_log.txt"
# python3 training_pipeline_eg.py  --model unisage --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 256 --log_output_path "./results/coauthorshipDBLP_unisage_eg_log.txt"
python3 training_pipeline_eg.py  --model unigat --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 256 --heads 8 --log_output_path "./results/coauthorshipDBLP_unigat_eg_log.txt"

# check 24.8.17
python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset news --lr 0.1 --hid 64 --log_output_path "./results/news_hgnn_eg_log.txt"
python3 training_pipeline_eg.py  --model hgnnp --epoch 100 --dataset news --lr 0.1 --hid 64 --log_output_path "./results/news_hgnnp_eg_log.txt"
python3 training_pipeline_eg.py  --model hnhn --epoch 100 --dataset news --lr 0.001 --hid 512 --log_output_path "./results/news_hnhn_eg_log.txt"
python3 training_pipeline_eg.py  --model hypergcn --epoch 100 --dataset news --lr 0.01 --hid 64  --decay 0.000001 --log_output_path "./results/news_hypergcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigcn --epoch 100 --dataset news --lr 0.001 --hid 128 --log_output_path "./results/news_unigcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigin --epoch 100 --dataset news --lr 0.001 --hid 128 --log_output_path "./results/news_unigin_eg_log.txt"
# python3 training_pipeline_eg.py  --model unisage --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 128 --log_output_path "./results/news_unisage_eg_log.txt"
python3 training_pipeline_eg.py  --model unigat --epoch 100 --dataset news --lr 0.001 --hid 128 --heads 8 --log_output_path "./results/news_unigat_eg_log.txt"

# check 24.8.17
python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset yelp --lr 0.001 --hid 256 --log_output_path "./results/yelp_hgnn_eg_log.txt"
python3 training_pipeline_eg.py  --model hgnnp --epoch 100 --dataset yelp --lr 0.001 --hid 256 --log_output_path "./results/yelp_hgnnp_eg_log.txt"
python3 training_pipeline_eg.py  --model hnhn --epoch 100 --dataset yelp --lr 0.001 --hid 128 --log_output_path "./results/yelp_hnhn_eg_log.txt"
python3 training_pipeline_eg.py  --model hypergcn --epoch 100 --dataset yelp --lr 0.01 --hid 64  --decay 0.000001 --log_output_path "./results/yelp_hypergcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigcn --epoch 100 --dataset yelp --lr 0.001 --hid 128 --log_output_path "./results/yelp_unigcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigin --epoch 100 --dataset yelp --lr 0.001 --hid 128 --log_output_path "./results/yelp_unigin_eg_log.txt"
# python3 training_pipeline_eg.py  --model unisage --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 128 --log_output_path "./results/yelp_unisage_eg_log.txt"
python3 training_pipeline_eg.py  --model unigat --epoch 100 --dataset yelp --lr 0.001 --hid 128 --heads 1 --log_output_path "./results/yelp_unigat_eg_log.txt"

# check 24.8.17
python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512 --log_output_path "./results/walmart_trips_hgnn_eg_log.txt"
python3 training_pipeline_eg.py  --model hgnnp --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512 --log_output_path "./results/walmart_trips_hgnnp_eg_log.txt"
python3 training_pipeline_eg.py  --model hnhn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512 --log_output_path "./results/walmart_trips_hnhn_eg_log.txt"
python3 training_pipeline_eg.py  --model hypergcn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 128  --decay 0.000001 --log_output_path "./results/walmart_trips_hypergcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigcn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 256 --log_output_path "./results/walmart_trips_unigcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigin --epoch 100 --dataset walmart_trips --lr 0.001 --hid 256 --log_output_path "./results/walmart_trips_unigin_eg_log.txt"
# # python3 training_pipeline_eg.py  --model unisage --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512  --log_output_path "./results/walmart_trips_unisage_eg_log.txt"
python3 training_pipeline_eg.py  --model unigat --epoch 100 --dataset walmart_trips --lr 0.001 --hid 256  --heads 4 --log_output_path "./results/walmart_trips_unigat_eg_log.txt"


# check 24.8.17
python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512 --log_output_path "./results/trivago_clicks_hgnn_eg_log.txt"
python3 training_pipeline_eg.py  --model hgnnp --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512 --log_output_path "./results/trivago_clicks_hgnnp_eg_log.txt"
python3 training_pipeline_eg.py  --model hnhn --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512 --log_output_path "./results/trivago_clicks_hnhn_eg_log.txt"
python3 training_pipeline_eg.py  --model hypergcn --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512 --log_output_path "./results/trivago_clicks_hypergcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigcn --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512 --log_output_path "./results/trivago_clicks_unigcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigin --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512 --log_output_path "./results/trivago_clicks_unigin_eg_log.txt"
# python3 training_pipeline_eg.py  --model unisage --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512 --log_output_path "./results/trivago_clicks_unisage_eg_log.txt"
python3 training_pipeline_eg.py  --model unigat --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 64 --heads 1 --log_output_path "./results/trivago_clicks_unigat_eg_log.txt"


### dhg

# check 24.8.17
python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset cocitation_cora --lr 0.001 --hid 512 --log_output_path "./results/cocitation_cora_hgnn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hgnnp --epoch 100 --dataset cocitation_cora --lr 0.001 --hid 512 --log_output_path "./results/cocitation_cora_hgnnp_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hnhn --epoch 100 --dataset cocitation_cora --lr 0.001 --hid 512 --log_output_path "./results/cocitation_cora_hnhn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hypergcn --epoch 100 --dataset cocitation_cora --lr 0.001 --hid 64 --log_output_path "./results/cocitation_cora_hypergcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigcn --epoch 100 --dataset cocitation_cora --lr 0.001 --hid 512 --log_output_path "./results/cocitation_cora_unigcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigin --epoch 100 --dataset cocitation_cora --lr 0.001 --hid 512 --log_output_path "./results/cocitation_cora_unigin_dhg_log.txt"
# python3 training_pipeline_dhg.py  --model unisage --epoch 100 --dataset cocitation_cora --lr 0.001 --hid 512 --log_output_path "./results/cocitation_cora_unisage_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigat --epoch 100 --dataset cocitation_cora --lr 0.001 --hid 512 --heads 8 --log_output_path "./results/cocitation_cora_unigat_dhg_log.txt"

# check 24.8.17
python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset coauthorshipCora --lr 0.001 --hid 128 --log_output_path "./results/coauthorshipCora_hgnn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hgnnp --epoch 100 --dataset coauthorshipCora --lr 0.001 --hid 512 --log_output_path "./results/coauthorshipCora_hgnnp_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hnhn --epoch 100 --dataset coauthorshipCora --lr 0.001 --hid 512 --log_output_path "./results/coauthorshipCora_hnhn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hypergcn --epoch 100 --dataset coauthorshipCora --lr 0.01 --hid 64 --decay 0.000001 --log_output_path "./results/coauthorshipCora_hypergcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigcn --epoch 100 --dataset coauthorshipCora --lr 0.001 --hid 512 --log_output_path "./results/coauthorshipCora_unigcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigin --epoch 100 --dataset coauthorshipCora --lr 0.001 --hid 512 --log_output_path "./results/coauthorshipCora_unigin_dhg_log.txt"
# python3 training_pipeline_dhg.py  --model unisage --epoch 100 --dataset coauthorshipCora --lr 0.001 --hid 512 --log_output_path "./results/coauthorshipCora_unisage_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigat --epoch 100 --dataset coauthorshipCora --lr 0.001 --hid 512 --heads 4 --log_output_path "./results/coauthorshipCora_unigat_dhg_log.txt"

# check 24.8.17
python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset cocitation_citeseer --lr 0.001 --hid 256 --log_output_path "./results/cocitation_citeseer_hgnn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hgnnp --epoch 100 --dataset cocitation_citeseer --lr 0.001 --hid 256 --log_output_path "./results/cocitation_citeseer_hgnnp_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hnhn --epoch 100 --dataset cocitation_citeseer --lr 0.001 --hid 256  --log_output_path "./results/cocitation_citeseer_hnhn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hypergcn --epoch 100 --dataset cocitation_citeseer --lr 0.01 --hid 64 --decay 0.000001 --log_output_path "./results/cocitation_citeseer_hypergcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigcn --epoch 100 --dataset cocitation_citeseer --lr 0.001 --hid 128 --log_output_path "./results/cocitation_citeseer_unigcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigin --epoch 100 --dataset cocitation_citeseer --lr 0.001 --hid 128 --log_output_path "./results/cocitation_citeseer_unigin_dhg_log.txt"
# python3 training_pipeline_dhg.py  --model unisage --epoch 100 --dataset cocitation_citeseer --lr 0.01 --hid 512 --drop_rate 0.5 --log_output_path "./results/cocitation_citeseer_unisage_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigat --epoch 100 --dataset cocitation_citeseer --lr 0.001 --hid 128 --heads 1 --log_output_path "./results/cocitation_citeseer_unigat_dhg_log.txt"

# check 24.8.17
python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset cocitation_pubmed --lr 0.001 --hid 512 --log_output_path "./results/cocitation_pubmed_hgnn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hgnnp --epoch 100 --dataset cocitation_pubmed --lr 0.001 --hid 512 --log_output_path "./results/cocitation_pubmed_hgnnp_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hnhn --epoch 100 --dataset cocitation_pubmed --lr 0.001 --hid 512 --log_output_path "./results/cocitation_pubmed_hnhn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hypergcn --epoch 100 --dataset cocitation_pubmed --lr 0.01 --hid 64  --decay 0.000001 --log_output_path "./results/cocitation_pubmed_hypergcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigcn --epoch 100 --dataset cocitation_pubmed --lr 0.001 --hid 128 --log_output_path "./results/cocitation_pubmed_unigcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigin --epoch 100 --dataset cocitation_pubmed --lr 0.001 --hid 128 --log_output_path "./results/cocitation_pubmed_unigin_dhg_log.txt"
# python3 training_pipeline_dhg.py  --model unisage --epoch 100 --dataset cocitation_pubmed --lr 0.001 --hid 128 --log_output_path "./results/cocitation_pubmed_unisage_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigat --epoch 100 --dataset cocitation_pubmed --lr 0.001 --hid 128 --heads 1 --log_output_path "./results/cocitation_pubmed_unigat_dhg_log.txt"

# check 24.8.17
python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 256 --log_output_path "./results/coauthorshipDBLP_hgnn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hgnnp --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 256 --log_output_path "./results/coauthorshipDBLP_hgnnp_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hnhn --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 512 --log_output_path "./results/coauthorshipDBLP_hnhn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hypergcn --epoch 100 --dataset coauthorshipDBLP --lr 0.01 --hid 64  --decay 0.000001 --log_output_path "./results/coauthorshipDBLP_hypergcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigcn --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 256 --log_output_path "./results/coauthorshipDBLP_unigcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigin --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 256 --log_output_path "./results/coauthorshipDBLP_unigin_dhg_log.txt"
# python3 training_pipeline_dhg.py  --model unisage --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 256 --log_output_path "./results/coauthorshipDBLP_unisage_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigat --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 256 --heads 8 --log_output_path "./results/coauthorshipDBLP_unigat_dhg_log.txt"

# check 24.8.17
python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset news --lr 0.1 --hid 64 --log_output_path "./results/news_hgnn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hgnnp --epoch 100 --dataset news --lr 0.1 --hid 64 --log_output_path "./results/news_hgnnp_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hnhn --epoch 100 --dataset news --lr 0.001 --hid 512 --log_output_path "./results/news_hnhn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hypergcn --epoch 100 --dataset news --lr 0.01 --hid 64  --decay 0.000001 --log_output_path "./results/news_hypergcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigcn --epoch 100 --dataset news --lr 0.001 --hid 128 --log_output_path "./results/news_unigcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigin --epoch 100 --dataset news --lr 0.001 --hid 128 --log_output_path "./results/news_unigin_dhg_log.txt"
# python3 training_pipeline_dhg.py  --model unisage --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 128 --log_output_path "./results/news_unisage_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigat --epoch 100 --dataset news --lr 0.001 --hid 128 --heads 8 --log_output_path "./results/news_unigat_dhg_log.txt"

# check 24.8.17
python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset yelp --lr 0.001 --hid 256 --log_output_path "./results/yelp_hgnn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hgnnp --epoch 100 --dataset yelp --lr 0.001 --hid 256 --log_output_path "./results/yelp_hgnnp_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hnhn --epoch 100 --dataset yelp --lr 0.001 --hid 128 --log_output_path "./results/yelp_hnhn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hypergcn --epoch 100 --dataset yelp --lr 0.01 --hid 64  --decay 0.000001 --log_output_path "./results/yelp_hypergcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigcn --epoch 100 --dataset yelp --lr 0.001 --hid 128 --log_output_path "./results/yelp_unigcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigin --epoch 100 --dataset yelp --lr 0.001 --hid 128 --log_output_path "./results/yelp_unigin_dhg_log.txt"
# python3 training_pipeline_dhg.py  --model unisage --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 128 --log_output_path "./results/yelp_unisage_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigat --epoch 100 --dataset yelp --lr 0.001 --hid 128 --heads 1 --log_output_path "./results/yelp_unigat_dhg_log.txt"

# check 24.8.17
python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512 --log_output_path "./results/walmart_trips_hgnn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hgnnp --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512 --log_output_path "./results/walmart_trips_hgnnp_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hnhn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512 --log_output_path "./results/walmart_trips_hnhn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hypergcn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 128  --decay 0.000001 --log_output_path "./results/walmart_trips_hypergcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigcn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 256 --log_output_path "./results/walmart_trips_unigcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigin --epoch 100 --dataset walmart_trips --lr 0.001 --hid 256 --log_output_path "./results/walmart_trips_unigin_dhg_log.txt"
# # python3 training_pipeline_dhg.py  --model unisage --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512  --log_output_path "./results/walmart_trips_unisage_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigat --epoch 100 --dataset walmart_trips --lr 0.001 --hid 256  --heads 4 --log_output_path "./results/walmart_trips_unigat_dhg_log.txt"


# check 24.8.17
python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512 --log_output_path "./results/trivago_clicks_hgnn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hgnnp --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512 --log_output_path "./results/trivago_clicks_hgnnp_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hnhn --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512 --log_output_path "./results/trivago_clicks_hnhn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hypergcn --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512 --log_output_path "./results/trivago_clicks_hypergcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigcn --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512 --log_output_path "./results/trivago_clicks_unigcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigin --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512 --log_output_path "./results/trivago_clicks_unigin_dhg_log.txt"
# python3 training_pipeline_dhg.py  --model unisage --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512 --log_output_path "./results/trivago_clicks_unisage_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigat --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 64 --heads 1 --log_output_path "./results/trivago_clicks_unigat_dhg_log.txt"

