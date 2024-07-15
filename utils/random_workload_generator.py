from utils.graph_utils import order
import random


def generate_line_graph():
    # a line graph with 1 million edges
    V = 1000000
    edge_list = list()
    for u in range(0, V):
        edge_list.append((u, u + 1))
    random.shuffle(edge_list)

    return edge_list


def generate_complete_graph():
    # a complete graph with 1 million edges
    V = 1415
    edge_list = list()
    for u in range(0, V):
        for v in range(u + 1, V):
            edge_list.append((u, v))
    random.shuffle(edge_list)
    return edge_list


def generate_workloads(testcase):
    edges = set()
    if "random" in testcase:
        lines = open("datasets/" + testcase, 'r').readlines()

        for line in lines:
            items = line.rstrip().split(' ')
            if items[0] != 'a':
                continue
            u, v = order(int(items[1]), int(items[2]))
            edges.add((u, v))
        edge_list = list(edges)
        random.shuffle(edge_list)
    elif "line_graph" == testcase or "complete_graph" == testcase:
        if "line" in testcase:
            edge_list = generate_line_graph()
        else:
            edge_list = generate_complete_graph()
    else:
        # edge_list = []
        raise ValueError("invalid test case")

    insertion_writer = open("workloads/" + testcase + "_insertions.csv", 'w')
    for u, v in edge_list:
        insertion_writer.write("%d,%d\n" % (u, v))
    insertion_writer.flush()
    insertion_writer.close()

    random.shuffle(edge_list)

    deletion_writer = open("workloads/" + testcase + "_deletions.csv", 'w')
    for u, v in edge_list:
        deletion_writer.write("%d,%d\n" % (u, v))
    deletion_writer.flush()
    deletion_writer.close()

