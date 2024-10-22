
# An experimental comparison of tree-data structures for connectivity queries on fully-dynamic undirected graphs


We implement and emprically evaluate all major data structures for the dynamic connectivity problem: [D-tree](https://dl.acm.org/doi/abs/10.14778/3551793.3551868), 
[Link-cut tree](https://dl.acm.org/doi/10.1145/800076.802464), [HK](https://dl.acm.org/doi/10.1145/320211.320215), 
[HKS (a simplified version of HK)](https://dl.acm.org/doi/10.1145/264216.264223), [HDT](https://dl.acm.org/doi/10.1145/502090.502095),
[structural tree ST](https://dl.acm.org/doi/10.1145/502090.502095), 
[a variant STV of structural tree](https://dl.acm.org/doi/10.5555/2627817.2627943), 
[local tree LT](https://dl.acm.org/doi/10.1145/335305.335345), 
[a variant LTV of local tree]((https://dl.acm.org/doi/10.5555/2627817.2627943)), 
and [lazy local trees LzT](https://dl.acm.org/doi/10.1145/335305.335345). 

Memory footprints, update performances and query performances are evaluated for all above data structures.  
Source codes for implementing the data structures and running our experiments are  included in this repository. 

[**Project Structure**](#project-structure) | [**Prerequisites**](#prerequisites) | [**Get Started**](#get-start) | [**Reproduce**](#Reproduce)


## Project Structure

```
.
|-- Class                  # auxiliary classes used for storing results
|-- datasets               # 
|-- Dtree                  # Source codes for Dtree
|-- ET                     # Source codes for Euler Tour tree   
|-- examples               # examples used for quick start   
|-- HDT                    # Source codes for HDT
|-- HK                     # Source codes for HK
|-- HKS                    # Source codes for HKS
|-- LCT                    # Source codes for link-cut tree LCT
|-- LT                     # Source codes for local tree LT
|-- LTV                    # Source codes a variant LTV of local trees
|-- LzT                    # Source codes for lazy local trees LzT
|-- res                    # folder for experimental results
|-- ST                     # Source codes for structural tree ST
|-- STV                    # Source codes for a variant STV of structural trees
|-- utils                  # utilities and common funtions
`-- workloads              # folder for workloads
```

___
## Prerequisites

- Debian GNU/Linux 12; 200 GB RAM + 80 swap memory
- install Python3
- clone this repository
- install dependencies: 
    ```bash
    pip3 install -r requirements.text
    ```

## Get Started

### Run experiments on the  [fb-forum](https://networkrepository.com/fb-forum.php) dataset in **examples** folder to get started.
- Step 1: go to examples folder
- Step 2: run script
  ```commandline
    bash  ./example.sh
  ```

- Step 3: generate workloads
    ```bash
      python3 generate_fb_workload.py
    ```
- Step 4:  Run experiments on fb datset
  
    - Evaluating memory footprints:
      ```commandline
      python3 exmaple_memory_footprints.py fb
      ```
      
    - Evaluating update performances:
      ```commandline
      python3 example_update_performance.py fb
      ```
      
    - Evaluating query performances:
      ```commandline
      python3 example_query_performance.py fb
      ```

## Reproduce 
As dicussed in the paper, no data structure is robust. When running holistic evaluations, there are two types of errors.

- Run out of memory. Only D-tree can finish the workloads on SC, and all other data structures run out of memory.
- Run out of time. Experiments can take extremely long time to finish.

 Run experiments shown in the paper:
  
- Evaluating memory footprints:
  ```commandline
  python3 evaluate_memory_footprints.py
  ```
  
- Evaluating update performances:
  ```commandline
  python3 evaluate_update_performance.py
  ```
  
- Evaluating query performances:
  ```commandline
  python3 evaluate_query_performance.py
