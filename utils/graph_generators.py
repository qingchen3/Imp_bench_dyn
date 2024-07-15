import random
import networkx as nx
import math
from pathlib import Path


def order(a, b):
    if a < b:
        return a, b
    else:
        return b, a


def generate_power_law_graph(n, m, p, edge_num):
    G = nx.random_graphs.powerlaw_cluster_graph(n, m, p)
    writer = open("datasets/" + "power_law_%d" %edge_num, 'w')
    for u, v, _ in G.edges.data():
        writer.write("%d %d\n" % (u, v))
    writer.flush()
    writer.close()
    print("Finishing generating powerlaw graph..")
    return


def generate_complete_graph(n, edge_num):
    G = nx.random_graphs.complete_graph(n)
    writer = open("datasets/" + "complete_graph_%d" %edge_num, 'w')
    for u, v, _ in G.edges.data():
        writer.write("%d %d\n" % (u, v))
    writer.flush()
    writer.close()
    print("Finishing generating complete graph..")
    return


def generate_random_graph(n, edge_num):
    G = nx.random_graphs.gnm_random_graph(n, m)
    writer = open("datasets/" + "random_graph_%d" %edge_num, 'w')
    for u, v, _ in G.edges.data():
        writer.write("%d %d\n" % (u, v))
        #print(u, v)
    writer.flush()
    writer.close()
    print("Finishing generating random graph..")
    return


def generate_path_graph(n, edge_num):
    G = nx.generators.classic.path_graph(n)
    writer = open("datasets/" + "path_graph_%d" %edge_num, 'w')
    for u, v, _ in G.edges.data():
        writer.write("%d %d\n" % (u, v))
    writer.flush()
    writer.close()
    print("Finishing generating path graph..")
    return


def generate_star_graph(n, edge_num):
    G = nx.generators.classic.star_graph(n)
    writer = open("datasets/" + "star_graph_%d" %edge_num, 'w')
    for u, v, _ in G.edges.data():
        writer.write("%d %d\n" % (u, v))
    writer.flush()
    writer.close()
    print("Finishing generating star graph..")
    return



def generate_workload(n, type):

    lines = open("datasets/%s_%d" %(type, n), 'r').readlines()
    edges = list()
    writer = open("workloads/" + "%s_%d_workloads" %(type, n), 'w')
    for line in lines:
        items = line.rstrip().split(' ')
        u, v = order(int(items[0]), int(items[1]))
        edges.append((u, v))
        writer.write("ins %d %d\n" % (u, v))
    writer.flush()
    writer.close()

    writer1 = open("workloads/" + "%s_%d_workloads" %(type, n), 'a')
    for u, v in edges:
        writer1.write("del %d %d\n" % (u, v))
    writer1.flush()
    writer1.close()

    return


def generate_workload_ratio(graph_type, ratio):
    # ratio = |insertion| / |deletion|
    if Path("workloads/%s_ratio_%d" % (graph_type, ratio)).exists():
        print("workloads for %s with ratio %d exists" % (graph_type, ratio))
        return
    lines = open("datasets/%s" % graph_type, 'r').readlines()
    edges = list()
    writer = open("workloads/" + "%s_ratio_%d" % (graph_type, ratio), 'w')
    count = 0
    delimiter = ' '
    if graph_type in ['youtube', 'usa', 'trackers']:
        delimiter = '\t'
    for line in lines:
        items = line.rstrip().split(delimiter)
        if len(items) != 2 or items[0] == items[1]:
            continue
        u, v = order(int(items[0]), int(items[1]))
        edges.append((u, v))
        writer.write("ins %d %d\n" % (u, v))
        count += 1
        if count == ratio:
            idx = random.randint(0, len(edges) - 1)
            count = 0
            du, dv = edges[idx]
            writer.write("del %d %d\n" % (du, dv))
            del edges[idx:idx+1]

    writer.flush()
    writer.close()
    print("finish generating workloads for %s with ratio %d" % (graph_type, ratio))
    return


if __name__ == '__main__':

    graph_types = ['star_graph', 'path_graph', 'complete_graph', 'random_graph', 'power_law', 'usa', 'youtube',
                   'stackoverflow', 'trackers']
    m = 10000000
    for graph_type in graph_types:
        if graph_type in ['usa', 'youtube', 'stackoverflow', 'trackers'] or Path("datasets/%s_%d" % (graph_type, m)).exists():
            continue
        if graph_type == 'star_graph':
            generate_star_graph(m, m)
            print("finish generating star graph with %d edges" %m)
        elif graph_type == 'path_graph':
            generate_path_graph(m, m)
            print("finish generating path graph with %d edges" %m)
        elif graph_type == 'complete_graph':
            generate_complete_graph(int(math.sqrt(2 * m)), m)
            print("finish generating complete graph with %d edges" %m)
        elif graph_type == 'random_graph':
            generate_random_graph(m // 10, m)
            print("finish generating random graph with %d edges" %m)
        elif graph_type == 'power_law':
            generate_power_law_graph(m // 10, 10, 0.8, m)
            print("finish generating powerlaw graph with %d edges" %m)

    ratios = [1000, 100, 20, 10, 5]  # ratio = |insertion| / |deletion|, reversion of u_r
    for graph_type in graph_types:
        for ratio in ratios:
            generate_workload_ratio(graph_type, ratio)
