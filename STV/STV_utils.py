"""
Implementation of the cluster forest in Christian Wulff-Nilsen's
Faster Deterministic Fully-Dynamic Graph Connectivity paper
"""

from ST.STNode import STNode
from ST.ST_utils import cal_node_edge_size
from queue import Queue
from utils.tree_utils import order
import sys


def merge(r_u, r_v):  # merge two cluster trees
    if r_u.N < r_v.N:
        for c in r_u.children:
            c.parent = r_v
            r_v.children.add(c)
        r_v.N += r_u.N
        del r_u
    else:
        for c in r_v.children:
            c.parent = r_u
            r_u.children.add(c)
        r_u.N += r_v.N
        del r_v
    return


def insert(u, v, CT, adjacency_level):

    if u not in CT:
        CT[u] = STNode(u)
        par_u = STNode(None)
        par_u.children = set()
        par_u.children.add(CT[u])
        CT[u].parent = par_u
        par_u.N = 1

        adjacency_level[u] = dict()
        adjacency_level[u][0] = set()

    if v not in CT:
        CT[v] = STNode(v)
        par_v = STNode(None)
        par_v.children = set()
        par_v.children.add(CT[v])
        CT[v].parent = par_v
        par_v.N = 1

        adjacency_level[v] = dict()
        adjacency_level[v][0] = set()

    for i in adjacency_level[u].keys():  # filter out redundant edges
        if v in adjacency_level[u][i]:
            return

    adjacency_level[v][0].add(u)
    adjacency_level[u][0].add(v)

    r_u = toRoot(CT[u])
    r_v = toRoot(CT[v])
    if r_u != r_v:
        merge(r_u, r_v)

    return


def delete_case2(sV, P, adjacency_level, level):
    for u, v in P:
        if u == v:
            continue
        adjacency_level[u][level].remove(v)
        adjacency_level[v][level].remove(u)

        if level + 1 not in adjacency_level[u]:
            adjacency_level[u][level + 1] = set()
        if level + 1 not in adjacency_level[v]:
            adjacency_level[v][level + 1] = set()

        adjacency_level[u][level + 1].add(v)
        adjacency_level[v][level + 1].add(u)

    # if len(sV) == 1:  # only promote edges since there are no loops on leaf nodes.
    #    return

    w = STNode(None)
    p_prime = STNode(None)
    p = None

    for s in sV:
        if p is None:
            p = s.parent
        p.children.remove(s)  # remove s from s.parent
        p.N -= s.N

        w.level = s.level

        if s.val is not None:
            if w.children is None:
                w.children = set()
            w.children.add(s)
            s.parent = w
            s.level = s.level + 1
        else:
            if w.children is None:
                w.children = set()
            w.children.update(s.children)
            for ch in s.children:
                ch.parent = w
        w.N += s.N

    p_prime.level = p.level

    w.parent = p_prime
    if p_prime.children is None:
        p_prime.children = set()
    p_prime.children.add(w)
    p_prime.N = w.N

    p_prime.parent = p.parent
    if p.parent is not None:
        p.parent.children.add(p_prime)

    return


def delete_case1(C, P, adjacency_level, level):
    if len(P) == 0:
        return

    for u, v in P:  # promote edges in P
        if u == v:
            continue
        adjacency_level[u][level].remove(v)
        adjacency_level[v][level].remove(u)

        if level + 1 not in adjacency_level[u]:
            adjacency_level[u][level + 1] = set()

        if level + 1 not in adjacency_level[v]:
            adjacency_level[v][level + 1] = set()

        adjacency_level[u][level + 1].add(v)
        adjacency_level[v][level + 1].add(u)

    if len(C) == 1 and next(iter(C)).val is None:
        # there is only one cluster node in C, don't merge, which was not detailed in the paper.
        return

    n = STNode(None)
    par = None
    for c in C:
        if par is None:
            par = c.parent
        par.children.remove(c)
        n.level = c.level
        if c.val is not None:  # leaf node
            if n.children is None:
                n.children = set()
            n.children.add(c)
            c.parent = n
            c.level = c.level + 1
        else:
            if n.children is None:
                n.children = set()
            n.children.update(c.children)
            for ch in c.children:
                ch.parent = n
        n.N += c.N

    n.parent = par
    par.children.add(n)

    return


def delete(u, v, CT, adjacency_level):

    i = get_level(u, v, adjacency_level)
    if i == -1:
        return
    adjacency_level[u][i].remove(v)
    adjacency_level[v][i].remove(u)

    while i >= 0:
        # print(i, u, v)
        anc_u = ancestor_at_i(CT[u], i)
        stack_u = []  # DFS_u:  DFS starts at anc_u
        edges_u = []  # candidate edges to be visited by DFS_u
        stack_u.append(anc_u)
        C_u = set()
        C_u.add(anc_u)
        P_u = set()
        size_u = anc_u.N

        anc_v = ancestor_at_i(CT[v], i)
        stack_v = []  # DFS_v: DFS starts at anc_v
        edges_v = []  # candidate edges to be visited by DFS_v
        stack_v.append(anc_v)
        C_v = set()
        C_v.add(anc_v)
        P_v = set()
        size_v = anc_v.N

        while (len(stack_u) > 0 or len(edges_u) > 0) and (len(stack_v) > 0 or len(edges_v) > 0):
            if len(edges_u) == 0:
                one_search(stack_u, edges_u, adjacency_level, i)
            else:
                vertex_u, adj_u = edges_u.pop()
                anc_adj_u = ancestor_at_i(CT[adj_u], i)
                if anc_adj_u in C_v:  # case 1
                    if size_u < size_v:
                        delete_case1(C_u, P_u, adjacency_level, i)
                    else:
                        delete_case1(C_v, P_v, adjacency_level, i)
                    return
                x, y = order(vertex_u, adj_u)
                P_u.add((x, y))
                if anc_adj_u not in C_u:
                    stack_u.append(anc_adj_u)
                    C_u.add(anc_adj_u)
                    size_u += anc_adj_u.N

            if len(edges_v) == 0:
                one_search(stack_v, edges_v, adjacency_level, i)
            else:
                vertex_v, adj_v = edges_v.pop()
                anc_adj_v = ancestor_at_i(CT[adj_v], i)
                if anc_adj_v in C_u:  # case 1
                    if size_u < size_v:
                        delete_case1(C_u, P_u, adjacency_level, i)
                    else:
                        delete_case1(C_v, P_v, adjacency_level, i)
                    return
                x, y = order(vertex_v, adj_v)
                P_v.add((x, y))
                if anc_adj_v not in C_v:
                    stack_v.append(anc_adj_v)
                    C_v.add(anc_adj_v)
                    size_v += anc_adj_v.N

        if len(stack_u) == 0 and len(edges_u) == 0:

            while (len(stack_v) > 0 or len(edges_v) > 0) and size_v <= size_u:
                if len(edges_v) == 0:
                    one_search(stack_v, edges_v, adjacency_level, i)
                else:
                    vertex_v, adj_v = edges_v.pop()
                    anc_adj_v = ancestor_at_i(CT[adj_v], i)
                    x, y = order(vertex_v, adj_v)
                    P_v.add((x, y))
                    if anc_adj_v not in C_v:
                        stack_v.append(anc_adj_v)
                        C_v.add(anc_adj_v)
                        size_v += anc_adj_v.N

        if len(stack_v) == 0 and len(edges_v) == 0:
            while (len(stack_u) > 0 or len(edges_u) > 0) and size_u <= size_v:
                if len(edges_u) == 0:
                    one_search(stack_u, edges_u, adjacency_level, i)
                else:
                    vertex_u, adj_u = edges_u.pop()
                    anc_adj_u = ancestor_at_i(CT[adj_u], i)
                    x, y = order(vertex_u, adj_u)
                    P_u.add((x, y))
                    if anc_adj_u not in C_u:
                        stack_u.append(anc_adj_u)
                        C_u.add(anc_adj_u)
                        size_u += anc_adj_u.N

        if size_u <= size_v:
            delete_case2(C_u, P_u, adjacency_level, i)
        else:
            delete_case2(C_v, P_v, adjacency_level, i)
        i -= 1

    return


def one_search(stack, edges, adjacency_level, i):
    if len(stack) == 0:
        return

    cur = stack.pop()
    if cur.val is not None:
        for adj_u in adjacency_level[cur.val][i]:
            edges.append((cur.val, adj_u))
    else:
        for c in cur.children:
            stack.append(c)
    return


def get_level(u, v, adjacency_level):
    for level, adj in adjacency_level[u].items():
        if v in adj:
            return level
    return -1


def ancestor_at_i(ct_node, i):
    anc = ct_node
    while anc is not None:
        if anc.level == i:
            return anc
        anc = anc.parent
    return anc


def toRoot(node):
    r = node
    while r.parent is not None:
        r = r.parent
    return r


def distance2root(ct_node):
    t = ct_node
    dist = -1
    while t.parent is not None:
        dist += 1
        t = t.parent
    return dist


def printCT(ct_root):
    Q = Queue()
    Q.put(ct_root)
    l = 0
    while not Q.empty():
        Q1 = Queue()
        print("level %d" %l)
        while not Q.empty():
            ct_node = Q.get()
            if ct_node.val is None:
                print(ct_node, "cluster node with conns = ", ct_node.conns)
            else:
                print(ct_node, "vertex with conns = ", ct_node.conns)
            for temp in ct_node.children:
                Q1.put(temp)
        l += 1
        Q = Q1

    return


def query(node_u, node_v):

    return toRoot(node_u) == toRoot(node_v)


def cal_total_memory_use(ST, adj_list_group_by_levels):
    # print("memory size for STV dictionary: %d bytes" % sys.getsizeof(ST))
    # get all root nodes
    root_node_set = set()
    for _, node in ST.items():
        while node.parent is not None:
            node = node.parent
        root_node_set.add(node)

    space_n = 0
    space_e = 0
    for root_node in root_node_set:
        q = Queue()
        q.put(root_node)
        while not q.empty():
            node = q.get()
            n_size, e_size = cal_node_edge_size(node)
            space_n += n_size
            space_e += e_size
            if node.children is not None:
                for c in node.children:
                    q.put(c)
    space_n += sys.getsizeof(ST)

    for vertex, adj_list_dict in adj_list_group_by_levels.items():
        for level, vertice_set in adj_list_dict.items():
            if vertice_set is not None:
                space_e += sys.getsizeof(vertice_set)

    print("total memory size : %d bytes" % (space_n + space_e))
    return space_n, space_e


def sanity(CT):
    for v in CT.keys():
        n = CT[v]
        while n.parent is not None:
            if n.dist != n.parent.dist + 1:
                print(v)
            assert n.dist == n.parent.dist + 1
            n = n.parent

