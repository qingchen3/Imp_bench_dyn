from pathlib import Path
import random


def order(a, b):
    if a < b:
        return a, b
    else:
        return b, a


def generate_fb_workload_ratio(graph_type, ratio):
    # ratio = |insertion| / |deletion|
    if Path("workloads/%s_ratio_%d" % (graph_type, ratio)).exists():
        print("workloads for %s with ratio %d exists" % (graph_type, ratio))
        return
    lines = open("%s" % graph_type, 'r').readlines()
    edges = list()
    writer = open("../workloads/" + "%s_ratio_%d" % (graph_type, ratio), 'w')
    count = 0
    delimiter = ','
    edge_set = set()
    for line in lines:
        items = line.rstrip().split(delimiter)
        if len(items) < 2 or items[0] == items[1]:
            continue
        u, v = order(int(items[0]), int(items[1]))
        if (u, v) not in edge_set:
            edges.append((u, v))
            edge_set.add((u, v))
            writer.write("ins %d %d\n" % (u, v))
            count += 1
        if count == ratio:
            idx = random.randint(0, len(edges) - 1)
            count = 0
            du, dv = edges[idx]
            writer.write("del %d %d\n" % (du, dv))
            del edges[idx:idx+1]
            edge_set.remove((du, dv))

    writer.flush()
    writer.close()
    print("finish generating workloads for %s with ratio %d" % (graph_type, ratio))
    return


if __name__ == '__main__':

    graph_type = "fb"
    ratios = [1000, 100, 20, 10, 5]  # ratio = |insertion| / |deletion|, reversion of u_r
    for ratio in ratios:
        generate_fb_workload_ratio(graph_type, ratio)
