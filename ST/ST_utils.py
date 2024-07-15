"""
Implementation of the cluster forest in Christian Wulff-Nilsen's
Faster Deterministic Fully-Dynamic Graph Connectivity paper
"""

from ST.STNode import STNode
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


def edge_level_up(u, v, adjacency, level):
    adjacency[u][level].remove(v)  # adjacency list for non-tree and tree edge
    adjacency[v][level].remove(u)

    if level + 1 not in adjacency[u]:
        adjacency[u][level + 1] = set()

    if level + 1 not in adjacency[v]:
        adjacency[v][level + 1] = set()

    adjacency[u][level + 1].add(v)
    adjacency[v][level + 1].add(u)


def insert(u, v, ST, adjacency_level, adjacency_level_nt):

    if u not in ST:
        ST[u] = STNode(u)
        par_u = STNode(None)
        if par_u.children is None:
            par_u.children = set()
        par_u.children.add(ST[u])
        ST[u].parent = par_u
        par_u.N = 1
        par_u.level = -1

        adjacency_level[u] = dict()
        adjacency_level[u][0] = None

        adjacency_level_nt[u] = dict()
        adjacency_level_nt[u][0] = None

    if v not in ST:
        ST[v] = STNode(v)
        par_v = STNode(None)
        if par_v.children is None:
            par_v.children = set()
        par_v.children.add(ST[v])
        ST[v].parent = par_v
        par_v.N = 1
        par_v.level = -1

        adjacency_level[v] = dict()
        adjacency_level[v][0] = None

        adjacency_level_nt[v] = dict()
        adjacency_level_nt[v][0] = None

    for i in adjacency_level[u].keys():  # filter out redundant tree edges
        if adjacency_level[u][i] is not None and v in adjacency_level[u][i]:
            return

    for i in adjacency_level_nt[u].keys():  # filter out redundant non-tree edges
        if adjacency_level_nt[u][i] is not None and v in adjacency_level_nt[u][i]:
            return

    if query(ST[u], ST[v]):
        if adjacency_level_nt[u][0] is None:
            adjacency_level_nt[u][0] = set()
        if adjacency_level_nt[v][0] is None:
            adjacency_level_nt[v][0] = set()
        adjacency_level_nt[u][0].add(v)
        adjacency_level_nt[v][0].add(u)
        return

    if adjacency_level[u][0] is None:
        adjacency_level[u][0] = set()
    if adjacency_level[v][0] is None:
        adjacency_level[v][0] = set()

    adjacency_level[v][0].add(u)
    adjacency_level[u][0].add(v)

    r_u = toRoot(ST[u])
    r_v = toRoot(ST[v])
    merge(r_u, r_v)

    return


def delete_case2(sV, P, adjacency_level, adjacency_level_nt, level):
    for u, v in P:  # promote edges in P
        # non tree edges
        assert isNontree(u, v, adjacency_level_nt)
        adjacency_level_nt[u][level].remove(v)
        adjacency_level_nt[v][level].remove(u)

        if level + 1 not in adjacency_level_nt[u]:
            adjacency_level_nt[u][level + 1] = set()

        if level + 1 not in adjacency_level_nt[v]:
            adjacency_level_nt[v][level + 1] = set()

        adjacency_level_nt[u][level + 1].add(v)
        adjacency_level_nt[v][level + 1].add(u)
        # else:
        #    adjacency_level[u][level].remove(v)
        #    adjacency_level[v][level].remove(u)

        #    if level + 1 not in adjacency_level[u]:
        #        adjacency_level[u][level + 1] = set()

        #    if level + 1 not in adjacency_level[v]:
        #        adjacency_level[v][level + 1] = set()

        #    adjacency_level[u][level + 1].add(v)
        #    adjacency_level[v][level + 1].add(u)
    #print(P)

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
            w.children.add(s)
            s.parent = w
            s.level = s.level + 1
        else:
            w.children.update(s.children)
            for ch in s.children:
                ch.parent = w
        w.N += s.N

    p_prime.level = p.level

    w.parent = p_prime

    p_prime.children.add(w)
    p_prime.N = w.N

    p_prime.parent = p.parent
    if p.parent is not None:
        p.parent.children.add(p_prime)

    return


def delete_case1(C, P, adjacency_level, adjacency_level_nt, level):
    if len(P) == 0:
        return

    for u, v in P:  # promote edges in P
        # nontree edges
        assert isNontree(u, v, adjacency_level_nt)
        adjacency_level_nt[u][level].remove(v)
        adjacency_level_nt[v][level].remove(u)

        if level + 1 not in adjacency_level_nt[u]:
            adjacency_level_nt[u][level + 1] = set()

        if level + 1 not in adjacency_level_nt[v]:
            adjacency_level_nt[v][level + 1] = set()

        adjacency_level_nt[u][level + 1].add(v)
        adjacency_level_nt[v][level + 1].add(u)
        # else:
        # adjacency_level[u][level].remove(v)
        # adjacency_level[v][level].remove(u)

        # if level + 1 not in adjacency_level[u]:
        #    adjacency_level[u][level + 1] = set()

        # if level + 1 not in adjacency_level[v]:
        #    adjacency_level[v][level + 1] = set()

        # adjacency_level[u][level + 1].add(v)
        # adjacency_level[v][level + 1].add(u)

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
            n.children.add(c)
            c.parent = n
            c.level = c.level + 1
        else:
            n.children.update(c.children)
            for ch in c.children:
                ch.parent = n
        n.N += c.N

    n.parent = par
    par.children.add(n)

    return


def delete(u, v, ST, adjacency_level, adjacency_level_nt):

    i = get_level(u, v, adjacency_level, adjacency_level_nt)
    if i == -1:
        return
    if isNontree(u, v, adjacency_level_nt):
        adjacency_level_nt[u][i].remove(v)
        adjacency_level_nt[v][i].remove(u)
        return
    else:
        adjacency_level[u][i].remove(v)
        adjacency_level[v][i].remove(u)
    while i >= 0:
        anc_u = ancestor_at_i(ST[u], i)
        stack_u = []  # DFS_u:  DFS starts at anc_u
        edges_u = []  # candidate edges to be visited by DFS_u
        stack_u.append(anc_u)

        C_u = set()
        C_u.add(anc_u)
        P_u = set()
        size_u = anc_u.N

        anc_v = ancestor_at_i(ST[v], i)
        stack_v = []  # DFS_v: DFS starts at anc_v
        edges_v = []  # candidate edges to be visited by DFS_v
        stack_v.append(anc_v)
        C_v = set()
        C_v.add(anc_v)
        P_v = set()
        size_v = anc_v.N

        while (len(stack_u) > 0 or len(edges_u) > 0) and (len(stack_v) > 0 or len(edges_v) > 0):
            if len(edges_u) == 0:
                search(stack_u, i, adjacency_level, edges_u)
            else:
                vertex_u, adj_u = edges_u.pop()
                anc_adj_u = ancestor_at_i(ST[adj_u], i)
                x, y = order(vertex_u, adj_u)
                P_u.add((x, y))
                if anc_adj_u not in C_u:
                    stack_u.append(anc_adj_u)
                    C_u.add(anc_adj_u)
                    size_u += anc_adj_u.N

            if len(edges_v) == 0:
                search(stack_v, i, adjacency_level, edges_v)
            else:
                vertex_v, adj_v = edges_v.pop()
                anc_adj_v = ancestor_at_i(ST[adj_v], i)
                x, y = order(vertex_v, adj_v)
                P_v.add((x, y))
                if anc_adj_v not in C_v:
                    stack_v.append(anc_adj_v)
                    C_v.add(anc_adj_v)
                    size_v += anc_adj_v.N

        if len(stack_u) == 0 and len(edges_u) == 0:
            while (len(stack_v) > 0 or len(edges_v) > 0) and size_v <= size_u:
                if len(edges_v) == 0:
                    search(stack_v, i, adjacency_level, edges_v)
                else:
                    vertex_v, adj_v = edges_v.pop()
                    anc_adj_v = ancestor_at_i(ST[adj_v], i)
                    x, y = order(vertex_v, adj_v)
                    P_v.add((x, y))
                    if anc_adj_v not in C_v:
                        stack_v.append(anc_adj_v)
                        C_v.add(anc_adj_v)
                        size_v += anc_adj_v.N

        if len(stack_v) == 0 and len(edges_v) == 0:
            while (len(stack_u) > 0 or len(edges_u) > 0) and size_u <= size_v:
                if len(edges_u) == 0:
                    search(stack_u, i, adjacency_level, edges_u)
                else:
                    vertex_u, adj_u = edges_u.pop()
                    anc_adj_u = ancestor_at_i(ST[adj_u], i)
                    x, y = order(vertex_u, adj_u)
                    P_u.add((x, y))
                    if anc_adj_u not in C_u:
                        stack_u.append(anc_adj_u)
                        C_u.add(anc_adj_u)
                        size_u += anc_adj_u.N

        if size_u <= size_v:  # nodes in C_u are made Tv
            g, par, Tv = create_Tv(C_u, P_u, adjacency_level, i)
        else:  # nodes in C_v are made Tv
            g, par, Tv = create_Tv(C_v, P_v, adjacency_level, i)

        tx, ty, nontree_edges = search_replacement_edge(Tv, adjacency_level_nt, ST, i)
        for nu, nv in nontree_edges:
            edge_level_up(nu, nv, adjacency_level_nt, i)
        if tx is not None and ty is not None:  # (x, y) is a replacement edge, case 1 for deletion
            adjacency_level_nt[tx][i].remove(ty)
            adjacency_level_nt[ty][i].remove(tx)

            if i not in adjacency_level[tx]:
                adjacency_level[tx][i] = set()
            adjacency_level[tx][i].add(ty)

            if i not in adjacency_level[ty]:
                adjacency_level[ty][i] = set()
            adjacency_level[ty][i].add(tx)

            par.children.add(Tv)
            Tv.parent = par
            par.N += Tv.N

            if g is not None:
                g.children.add(par)
                par.parent = g
                g.N += par.N
            return
        else:  # case 2 for deletion
            p_prime = STNode(None)
            p_prime.children = set()
            p_prime.children.add(Tv)
            Tv.parent = p_prime

            p_prime.N = Tv.N
            p_prime.level = i - 1

            if g is not None:
                g.children.add(par)
                par.parent = g
                g.N += par.N

                g.children.add(p_prime)
                p_prime.parent = g
                g.N += p_prime.N
        i -= 1

    return


def delete1(u, v, ST, adjacency_level, adjacency_level_nt):

    if u == 108 and v == 110:
        print("check...")
    i = get_level(u, v, adjacency_level, adjacency_level_nt)

    if isNontree(u, v, adjacency_level_nt):
        adjacency_level_nt[u][i].remove(v)
        adjacency_level_nt[v][i].remove(u)
        print("delete nontree edge %d-%d" % (u, v))
        return
    else:
        adjacency_level[u][i].remove(v)
        adjacency_level[v][i].remove(u)
    print("delete edge %d-%d" % (u, v))
    while i >= 0:
        # print(i, u, v)
        anc_u = ancestor_at_i(ST[u], i)
        stack_u = []  # DFS_u:  DFS starts at anc_u
        edges_u = []  # candidate edges to be visited by DFS_u
        stack_u.append(anc_u)

        C_u = set()
        C_u.add(anc_u)
        visited_u = set()
        visited_u.add(anc_u)
        P_u = set()
        size_u = anc_u.N

        anc_v = ancestor_at_i(ST[v], i)
        stack_v = []  # DFS_v: DFS starts at anc_v
        edges_v = []  # candidate edges to be visited by DFS_v
        stack_v.append(anc_v)

        C_v = set()
        C_v.add(anc_v)
        visited_v = set()
        visited_v.add(anc_v)
        P_v = set()
        size_v = anc_v.N

        while (len(stack_u) > 0 or len(edges_u) > 0) and (len(stack_v) > 0 or len(edges_v) > 0):
            if len(edges_u) == 0:
                one_search(stack_u, edges_u, ST, visited_u, adjacency_level, adjacency_level_nt, i)
            else:
                vertex_u, adj_u = edges_u.pop()
                anc_adj_u = ancestor_at_i(ST[adj_u], i)
                if anc_adj_u in C_v:  # case 1
                    print("replacement edge:", vertex_u, adj_u)
                    assert isNontree(vertex_u, adj_u, adjacency_level_nt)
                    adjacency_level_nt[vertex_u][i].remove(adj_u)
                    adjacency_level_nt[adj_u][i].remove(vertex_u)

                    if i not in adjacency_level[vertex_u]:
                        adjacency_level[vertex_u][i] = set()
                    adjacency_level[vertex_u][i].add(adj_u)

                    if i not in adjacency_level[adj_u]:
                        adjacency_level[adj_u][i] = set()
                    adjacency_level[adj_u][i].add(vertex_u)

                    if size_u < size_v:
                        delete_case1(C_u, P_u, adjacency_level, adjacency_level_nt, i)
                    else:
                        delete_case1(C_v, P_v, adjacency_level, adjacency_level_nt, i)
                    return
                x, y = order(vertex_u, adj_u)
                P_u.add((x, y))
                if anc_adj_u not in C_u:
                    stack_u.append(anc_adj_u)
                    C_u.add(anc_adj_u)
                    size_u += anc_adj_u.N

            if len(edges_v) == 0:
                one_search(stack_v, edges_v, ST, visited_v, adjacency_level, adjacency_level_nt, i)
            else:
                vertex_v, adj_v = edges_v.pop()
                anc_adj_v = ancestor_at_i(ST[adj_v], i)
                if anc_adj_v in C_u:  # case 1
                    print("replacement edge:", vertex_v, adj_v)
                    assert isNontree(vertex_v, adj_v, adjacency_level_nt)
                    adjacency_level_nt[vertex_v][i].remove(adj_v)
                    adjacency_level_nt[adj_v][i].remove(vertex_v)

                    if i not in adjacency_level[vertex_v]:
                        adjacency_level[vertex_v][i] = set()
                    adjacency_level[vertex_v][i].add(adj_v)
                    if i not in adjacency_level[adj_v]:
                        adjacency_level[adj_v][i] = set()
                    adjacency_level[adj_v][i].add(vertex_v)

                    if size_u < size_v:
                        delete_case1(C_u, P_u, adjacency_level, adjacency_level_nt, i)
                    else:
                        delete_case1(C_v, P_v, adjacency_level, adjacency_level_nt, i)
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
                    one_search(stack_v, edges_v, ST, visited_v, adjacency_level, adjacency_level_nt, i)
                else:
                    vertex_v, adj_v = edges_v.pop()
                    anc_adj_v = ancestor_at_i(ST[adj_v], i)
                    x, y = order(vertex_v, adj_v)
                    P_v.add((x, y))
                    if anc_adj_v not in C_v:
                        stack_v.append(anc_adj_v)
                        C_v.add(anc_adj_v)
                        size_v += anc_adj_v.N

        if len(stack_v) == 0 and len(edges_v) == 0:
            while (len(stack_u) > 0 or len(edges_u) > 0) and size_u <= size_v:
                if len(edges_u) == 0:
                    one_search(stack_u, edges_u, ST, visited_u, adjacency_level, adjacency_level_nt, i)
                else:
                    vertex_u, adj_u = edges_u.pop()
                    anc_adj_u = ancestor_at_i(ST[adj_u], i)
                    x, y = order(vertex_u, adj_u)
                    P_u.add((x, y))
                    if anc_adj_u not in C_u:
                        stack_u.append(anc_adj_u)
                        C_u.add(anc_adj_u)
                        size_u += anc_adj_u.N

        if size_u <= size_v:
            delete_case2(C_u, P_u, adjacency_level, adjacency_level_nt, i)
        else:
            delete_case2(C_v, P_v, adjacency_level, adjacency_level_nt, i)
        i -= 1

    return


def create_Tv(C, P, adjacency_level, level):

    for u, v in P:  # promoting levels on tree edges
        edge_level_up(u, v, adjacency_level, level)

    g = None
    if len(C) == 1:
        # there is only one cluster node in C, don't merge, which was not detailed in the paper.
        tn = next(iter(C))
        par = ancestor_at_i(tn, level - 1)
        # remove par from its parent since par changes
        if par.parent is not None:
            g = ancestor_at_i(par, par.level - 1)
        if g is not None:
            g.children.remove(par)
        par.children.remove(tn)
        return g, par, tn

    Tv = None
    par = None
    for c in C:
        if par is None:
            par = c.parent
            if par.parent is not None:
                g = par.parent
            if g is not None:
                g.children.remove(par)
                par.parent = None
                g.N -= par.N

        par.children.remove(c)
        c.parent = None
        par.N -= c.N

        if c.val is not None:
            c.level = level + 1  # promote leaf node

        if Tv is None:
            if c.val is not None:
                Tv = STNode(None)
                Tv.children = set()
                Tv.children.add(c)
                c.parent = Tv
                Tv.level = level
            else:
                Tv = c
        else:
            if c.val is not None:
                Tv.children.add(c)
                c.parent = Tv
            else:
                Tv.children.update(c.children)
                for ch in c.children:
                    ch.parent = Tv

        Tv.N += c.N

    return g, par, Tv


def search(stack, i, adjacency_level, edges):
    # amortized cost O(h)
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


def search_replacement_edge(x, adjacency_level_nt, ST, i):
    stack = [x]
    nontree_edges = set()
    while len(stack) != 0:
        cur = stack.pop()
        if cur.val is not None:
            if cur.val not in adjacency_level_nt or i not in adjacency_level_nt[cur.val] or \
                    adjacency_level_nt[cur.val][i] is None or  len(adjacency_level_nt[cur.val][i]) == 0:
                continue
            for adj_nt_x in adjacency_level_nt[cur.val][i]:
                if ancestor_at_i(ST[adj_nt_x], i) != ancestor_at_i(ST[cur.val], i):
                    return adj_nt_x, cur.val, nontree_edges
                else:
                    nt_x, nt_y = order(adj_nt_x, cur.val)
                    if (nt_x, nt_y) not in nontree_edges:
                        nontree_edges.add((nt_x, nt_y))
        if cur.children is not None:
            for c in cur.children:
                stack.append(c)

    return None, None, nontree_edges


def one_search(stack, edges, ST, visited, adjacency_level, adjacency_level_nt, i):
    if len(stack) == 0:
        return

    cur = stack.pop()
    if cur.val is not None:
        if i in adjacency_level_nt[cur.val]:
            for adj_nt_u in adjacency_level_nt[cur.val][i]:
                edges.append((cur.val, adj_nt_u))

        if i in adjacency_level[cur.val]:
            for adj_u in adjacency_level[cur.val][i]:
                if ST[adj_u] not in visited:
                    stack.append(ST[adj_u])
                    visited.add(ST[adj_u])
    else:
        for c in cur.children:
            if c not in visited:
                stack.append(c)
                visited.add(c)
    return


def get_level(u, v, adjacency_level, adjacency_level_nt):
    for level, adj in adjacency_level[u].items():
        if adj is not None and v in adj:
            return level
    for level, adj in adjacency_level_nt[u].items():
        if adj is not None and v in adj:
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


def isNontree(u, v, adjacency_level_nt):
    for i in adjacency_level_nt[u]:
        if adjacency_level_nt[u][i] is not None and v in adjacency_level_nt[u][i]:
            return True
    return False


def query(node_u, node_v):

    return toRoot(node_u) == toRoot(node_v)


def cal_node_edge_size(node):
    node_size = 0
    edge_size = 0
    if node.parent is not None:
        edge_size += sys.getsizeof(node.parent)
    if node.children is not None:
        edge_size += sys.getsizeof(node.children)
    node_size += sys.getsizeof(node.val)
    node_size += sys.getsizeof(node.N)
    node_size += sys.getsizeof(node.level)

    return node_size, edge_size


def cal_total_memory_use(ST, adj_list_group_by_levels, adj_list_group_by_levels_nt):
    # collect all root nodes
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

    tree_edge_size = 0
    for vertex, adj_list_dict in adj_list_group_by_levels.items():
        for level, vertice_set in adj_list_dict.items():
            if vertice_set is not None:
                tree_edge_size += sys.getsizeof(vertice_set)

    nontree_edge_size = 0
    for vertex, adj_list_dict in adj_list_group_by_levels_nt.items():
        for level, vertice_set in adj_list_dict.items():
            if vertice_set is not None:
                nontree_edge_size += sys.getsizeof(vertice_set)
    space_e += (tree_edge_size + nontree_edge_size)
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

