# EasyHypergraph_benchmark
Benchmark codes for EasyHypergraph

## Installation

### Environment
The Ubuntu 20.04 Server features an Intel(R) Xeon(R) CPU E5 - 2690 v2 with a clock speed of 3.00GHz and 31 GB of Random Access Memory.

### Supported Python
3.8 <= Python <= 3.13 is required.

### Package install
```
pip install --upgrade Python-EasyGraph
```

or

```
    git clone --recursive https://github.com/easy-graph/Easy-Graph
    export EASYGRAPH_ENABLE_GPU="TRUE"  # for users who want to enable GPUs
    pip install ./Easy-Graph
```

## Hypergraph analysis experiments

```
cd hypergraph_analysis_experiments
python metric_benchmark.py
```


## Hypergraph learning experiments

```
cd hypergraph_learning_experiments
bash start_pipeline.sh
```


## Datasets
All datasets from eg_hypergraph_dataset.zip or integrated dataset in https://easy-graph.github.io/docs/reference/easygraph.datasets.html
