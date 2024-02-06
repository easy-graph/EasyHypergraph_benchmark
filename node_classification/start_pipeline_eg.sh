#!/bin/bash

# 运行 Python 3 程序并获取其进程 ID
python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset cocitation_pubmed --lr 0.001 --hid 512 --log_output_path "./results/cocitation_pubmed_hgnn_eg_log.txt"
python3 training_pipeline_eg.py  --model hgnnp --epoch 100 --dataset cocitation_pubmed --lr 0.001 --hid 512 --log_output_path "./results/cocitation_pubmed_hgnnp_eg_log.txt"
python3 training_pipeline_eg.py  --model hnhn --epoch 100 --dataset cocitation_pubmed --lr 0.001 --hid 512 --log_output_path "./results/cocitation_pubmed_hnhn_eg_log.txt"
python3 training_pipeline_eg.py  --model hypergcn --epoch 100 --dataset cocitation_pubmed --lr 0.01 --hid 64  --decay 0.000001 --log_output_path "./results/cocitation_pubmed_hypergcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigcn --epoch 100 --dataset cocitation_pubmed --lr 0.001 --hid 128 --log_output_path "./results/cocitation_pubmed_unigcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigat --epoch 100 --dataset cocitation_pubmed --lr 0.001 --hid 128 --heads 1 --log_output_path "./results/cocitation_pubmed_unigat_eg_log.txt"

python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 256 --log_output_path "./results/coauthorshipDBLP_hgnn_eg_log.txt"
python3 training_pipeline_eg.py  --model hgnnp --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 512 --log_output_path "./results/coauthorshipDBLP_hgnnp_eg_log.txt"
python3 training_pipeline_eg.py  --model hnhn --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 512 --log_output_path "./results/coauthorshipDBLP_hnhn_eg_log.txt"
python3 training_pipeline_eg.py  --model hypergcn --epoch 100 --dataset coauthorshipDBLP --lr 0.01 --hid 64  --decay 0.000001 --log_output_path "./results/coauthorshipDBLP_hypergcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigcn --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 256 --log_output_path "./results/coauthorshipDBLP_unigcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigat --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 256 --heads 4 --log_output_path "./results/coauthorshipDBLP_unigat_eg_log.txt"

python3.10 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset yelp --lr 0.001 --hid 256 --log_output_path "./results/yelp_hgnn_eg_log.txt"
python3 training_pipeline_eg.py  --model hgnnp --epoch 100 --dataset yelp --lr 0.001 --hid 256 --log_output_path "./results/yelp_hgnnp_eg_log.txt"
python3 training_pipeline_eg.py  --model hnhn --epoch 100 --dataset yelp --lr 0.001 --hid 128 --log_output_path "./results/yelp_hnhn_eg_log.txt"
python3 training_pipeline_eg.py  --model hypergcn --epoch 100 --dataset yelp --lr 0.01 --hid 64  --decay 0.000001 --log_output_path "./results/yelp_hypergcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigcn --epoch 100 --dataset yelp --lr 0.001 --hid 128 --log_output_path "./results/yelp_unigcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigat --epoch 100 --dataset yelp --lr 0.001 --hid 128 --heads 1 --log_output_path "./results/yelp_unigat_eg_log.txt"


python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512 --log_output_path "./results/walmart_trips_hgnn_eg_log.txt"
python3 training_pipeline_eg.py  --model hgnnp --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512 --log_output_path "./results/walmart_trips_hgnnp_eg_log.txt"
python3 training_pipeline_eg.py  --model hnhn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512 --log_output_path "./results/walmart_trips_hnhn_eg_log.txt"
python3 training_pipeline_eg.py  --model hypergcn --epoch 100 --dataset walmart_trips --lr 0.01 --hid 64  --decay 0.000001 --log_output_path "./results/walmart_trips_hypergcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigcn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 256 --log_output_path "./results/walmart_trips_unigcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigat --epoch 100 --dataset walmart_trips --lr 0.001 --hid 256  --heads 4 --log_output_path "./results/walmart_trips_unigat_eg_log.txt"


python3 training_pipeline_eg.py  --model hgnn --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512 --log_output_path "./results/trivago_clicks_hgnn_eg_log.txt"
python3 training_pipeline_eg.py  --model hgnnp --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512 --log_output_path "./results/trivago_clicks_hgnnp_eg_log.txt"
python3 training_pipeline_eg.py  --model hnhn --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512 --log_output_path "./results/trivago_clicks_hnhn_eg_log.txt"
python3 training_pipeline_eg.py  --model hypergcn --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512 --log_output_path "./results/trivago_clicks_hypergcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigcn --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512 --log_output_path "./results/trivago_clicks_unigcn_eg_log.txt"
python3 training_pipeline_eg.py  --model unigat --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 16 --heads 1 --log_output_path "./results/trivago_clicks_unigat_eg_log.txt"


python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset cocitation_pubmed --lr 0.001 --hid 512 --log_output_path "./results/cocitation_pubmed_hgnn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hgnnp --epoch 100 --dataset cocitation_pubmed --lr 0.001 --hid 512 --log_output_path "./results/cocitation_pubmed_hgnnp_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hnhn --epoch 100 --dataset cocitation_pubmed --lr 0.001 --hid 512 --log_output_path "./results/cocitation_pubmed_hnhn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hypergcn --epoch 100 --dataset cocitation_pubmed --lr 0.01 --hid 64  --decay 0.000001 --log_output_path "./results/cocitation_pubmed_hypergcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigcn --epoch 100 --dataset cocitation_pubmed --lr 0.001 --hid 128 --log_output_path "./results/cocitation_pubmed_unigcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigat --epoch 100 --dataset cocitation_pubmed --lr 0.001 --hid 128 --heads 1 --log_output_path "./results/cocitation_pubmed_unigat_dhg_log.txt"


python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 256 --log_output_path "./results/coauthorshipDBLP_hgnn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hgnnp --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 512 --log_output_path "./results/coauthorshipDBLP_hgnnp_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hnhn --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 512 --log_output_path "./results/coauthorshipDBLP_hnhn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hypergcn --epoch 100 --dataset coauthorshipDBLP --lr 0.01 --hid 64  --decay 0.000001 --log_output_path "./results/coauthorshipDBLP_hypergcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigcn --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 256 --log_output_path "./results/coauthorshipDBLP_unigcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigat --epoch 100 --dataset coauthorshipDBLP --lr 0.001 --hid 256 --heads 4 --log_output_path "./results/coauthorshipDBLP_unigat_dhg_log.txt"


python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset yelp --lr 0.001 --hid 256 --log_output_path "./results/yelp_hgnn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hgnnp --epoch 100 --dataset yelp --lr 0.001 --hid 256 --log_output_path "./results/yelp_hgnnp_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hnhn --epoch 100 --dataset yelp --lr 0.001 --hid 128 --log_output_path "./results/yelp_hnhn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hypergcn --epoch 100 --dataset yelp --lr 0.01 --hid 64  --decay 0.000001 --log_output_path "./results/yelp_hypergcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigcn --epoch 100 --dataset yelp --lr 0.001 --hid 128 --log_output_path "./results/yelp_unigcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigat --epoch 100 --dataset yelp --lr 0.001 --hid 128 --heads 1 --log_output_path "./results/yelp_unigat_dhg_log.txt"


python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512  --log_output_path "./results/walmart_trips_hgnn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hgnnp --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512 --log_output_path "./results/walmart_trips_hgnnp_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hnhn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 512  --log_output_path "./results/walmart_trips_hnhn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hypergcn --epoch 100 --dataset walmart_trips --lr 0.01 --hid 64  --decay 0.000001 --log_output_path "./results/walmart_trips_hypergcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigcn --epoch 100 --dataset walmart_trips --lr 0.001 --hid 256 --log_output_path "./results/walmart_trips_unigcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigat --epoch 100 --dataset walmart_trips --lr 0.001 --hid 256 --heads 4 --log_output_path "./results/walmart_trips_unigat_dhg_log.txt"


python3 training_pipeline_dhg.py  --model hgnn --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512 --log_output_path "./results/trivago_clicks_hgnn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hgnnp --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512 --log_output_path "./results/trivago_clicks_hgnnp_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hnhn --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512 --log_output_path "./results/trivago_clicks_hnhn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model hypergcn --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512 --log_output_path "./results/trivago_clicks_hypergcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigcn --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 512 --log_output_path "./results/trivago_clicks_unigcn_dhg_log.txt"
python3 training_pipeline_dhg.py  --model unigat --epoch 100 --dataset trivago_clicks --lr 0.001 --hid 64 --heads 1 --log_output_path "./results/trivago_clicks_unigat_dhg_log.txt"