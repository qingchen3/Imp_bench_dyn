from LT.bak.LTNode import LTNode
from ST.ST_utils import toRoot
from utils.tree_utils import order
from queue import Queue
import math, sys
from timeit import default_timer as timer

level_changes = 0
max_level = 0


def query(LT, u, v):
    if u not in LT or v not in LT:
        return False
    return toRoot(LT[u]) == toRoot(LT[v])


def tree_edge_level_up(u, v, LT, adj_t_group_by_levels, level):
    adj_t_group_by_levels[u][level].remove(v)  # adjacency list for non-tree and tree edge
    adj_t_group_by_levels[v][level].remove(u)

    if len(adj_t_group_by_levels[u][level]) == 0:
        LT[u].tree_bitmap[level] = 0  # update tree-edge bitmap
        update_bitmap(LT[u], level, type)

    if len(adj_t_group_by_levels[v][level]) == 0:
        LT[v].tree_bitmap[level] = 0
        update_bitmap(LT[v], level, type)

    if level + 1 not in adj_t_group_by_levels[u]:
        adj_t_group_by_levels[u][level + 1] = set()

    if level + 1 not in adj_t_group_by_levels[v]:
        adj_t_group_by_levels[v][level + 1] = set()

    adj_t_group_by_levels[u][level + 1].add(v)
    adj_t_group_by_levels[v][level + 1].add(u)

    global level_changes
    level_changes += 1

    global max_level
    max_level = max(max_level, level + 1)
    # update bitmap
    if LT[u].tree_bitmap[level + 1] == 0:
        LT[u].tree_bitmap[level + 1] = 1
        update_bitmap(LT[u], level + 1, 1)

    if LT[v].tree_bitmap[level + 1] == 0:
        LT[v].tree_bitmap[level + 1] = 1
        update_bitmap(LT[v], level + 1, 1)


def nontree_edge_level_up(u, v, LT, adj_nt_group_by_levels, level):
    adj_nt_group_by_levels[u][level].remove(v)  # adjacency list for non-tree and tree edge
    adj_nt_group_by_levels[v][level].remove(u)


    if len(adj_nt_group_by_levels[u][level]) == 0:
        LT[u].nontree_bitmap[level] = 0  # update tree-edge bitmap
        update_bitmap(LT[u], level, 0)

    if len(adj_nt_group_by_levels[v][level]) == 0:
        LT[v].nontree_bitmap[level] = 0
        update_bitmap(LT[v], level, 0)

    if level + 1 not in adj_nt_group_by_levels[u]:
        adj_nt_group_by_levels[u][level + 1] = set()

    if level + 1 not in adj_nt_group_by_levels[v]:
        adj_nt_group_by_levels[v][level + 1] = set()

    adj_nt_group_by_levels[u][level + 1].add(v)
    adj_nt_group_by_levels[v][level + 1].add(u)

    global level_changes
    level_changes += 1

    global max_level
    max_level = max(max_level, level + 1)

    # update bitmap
    if LT[u].nontree_bitmap[level + 1] == 0:
        LT[u].nontree_bitmap[level + 1] = 1
        update_bitmap(LT[u], level + 1, 0)

    if LT[v].nontree_bitmap[level + 1] == 0:
        LT[v].nontree_bitmap[level + 1] = 1
        update_bitmap(LT[v], level + 1, 0)


def update_bitmap(node, level, type):  # level i
    # type = 0: nontree, type = 1: tree
    # only looking at the sibling's bitmap[level] == 1
    cur = node
    par = node.parent
    while par is not None:
        if cur == par.left:
            sibling = par.right
        else:
            sibling = par.left

        if sibling is None:
            if type == 0:
                par.nontree_bitmap[level] = cur.nontree_bitmap[level]
            else:
                par.tree_bitmap[level] = cur.tree_bitmap[level]
        else:
            if type == 0:
                if sibling.nontree_bitmap[level] == 1:
                    return
                else:
                    par.nontree_bitmap[level] = cur.nontree_bitmap[level] | sibling.nontree_bitmap[level]
            else:
                if sibling.tree_bitmap[level] == 1:
                    return
                else:
                    par.tree_bitmap[level] = cur.tree_bitmap[level] | sibling.tree_bitmap[level]
        cur = par
        par = par.parent
    return


def type(node):  # node type
    if node.left is None or node.right is None:
        return 1  # tree node
    else:
        if rank(node.left) == rank(node.right):
            return node.type
            # return 2  # root of a rank tree
        else:
            if node.level is None:
                return 3  # connecting (path) node
            else:
                return 1


def edge_type(u, v, adj_t):
    # 0: non-tree edge, 1: tree edge
    # complexity: O(lg n)
    for i in adj_t[u].keys():
        if v in adj_t[u][i]:
            return 1
    return 0


def rank(node):
    if node.N <= 0:
        return -1
    return int(math.floor(math.log(node.N, 2)))


def remove_connecting_nodes(r):
    # given a root of a local tree, remove all connecting nodes and return a list of local rank trees
    # with ranks in a descending order

    if type(r) == 2:  # r is a local rank tree
        l = [pair_up(r.left, r.right)]
        return l, 0
    else:
        if r.val is not None:
            l= [r]
            return l, 0
        else:
            l = []
            cur = r
            while cur is not None:
                if cur.right is not None:
                    l.append(cur.right)
                if cur.left is not None and type(cur.left) != 3:
                    l.append(cur.left)
                    break
                cur = cur.left
            return l, 1


def remove_connecting_nodes_and_Rb(r, Rb):
    # given a root of a local tree, remove all connecting nodes and Rb
    # return a list of local rank trees with ranks in a descending order
    l = []
    cur = r
    while cur is not None:
        if cur.right is not None and cur.right != Rb:
            l.append(cur.right)
        if cur.left is not None and type(cur.left) != 3:
            if cur.left != Rb:
                l.append(cur.left)
            break
        cur = cur.left
    return l, 0


def pair_up(c1, c2):  # c1.rank = c2.rank
    p = LTNode(None)
    if c1.N < c2.N:
        left_child = c1
        right_child = c2
    else:
        left_child = c2
        right_child = c1
    p.N = left_child.N + right_child.N  # update size
    p.nontree_bitmap = left_child.nontree_bitmap | right_child.nontree_bitmap
    p.tree_bitmap = left_child.tree_bitmap | right_child.tree_bitmap

    p.left = left_child
    left_child.parent = p

    p.right = right_child
    right_child.parent = p
    p.type = 2
    return p


def merge_lists(l_u, l_v):
    # nodes in both l_u and l_v are sorted by ranks in a descending order
    # nodes in l are sorted by ranks in an ascending order, which is used for construct the local tree

    l = []
    carry = None
    while len(l_u) != 0 or len(l_v) != 0 or carry is not None:  # pair up nodes with the same rank
        if carry is None:
            if len(l_v) == 0:
                l.append(l_u.pop())
            elif len(l_u) == 0:
                l.append(l_v.pop())
            else:
                if rank(l_u[-1]) < rank(l_v[-1]):  # if l_u[-1].rank < l_v[-1].rank:
                    l.append(l_u.pop())
                elif rank(l_u[-1]) > rank(l_v[-1]):  # elif l_u[-1].rank > l_v[-1].rank:
                    l.append(l_v.pop())
                else:
                    carry = pair_up(l_u.pop(), l_v.pop())
                    carry.type = 2
        else:
            if len(l_v) == 0:
                l_v.append(carry)
                carry = None
            elif len(l_u) == 0:
                l_u.append(carry)
                carry = None
            else:
                if rank(l_u[-1]) == rank(carry) and rank(l_v[-1]) == rank(carry):
                    if l_u[-1].N <= carry.N and l_u[-1].N <= l_v[-1].N:
                        l.append(l_u.pop())
                        new_carry = pair_up(carry, l_v.pop())
                    elif l_v[-1].N <= carry.N and l_v[-1].N <= l_u[-1].N:  # l_v[-1].N is the smallest
                        l.append(l_v.pop())
                        new_carry = pair_up(carry, l_u.pop())
                    else:  # carry.N is the smallest
                        l.append(carry)
                        new_carry = pair_up(l_u.pop(), l_v.pop())
                    new_carry.type = 2
                    carry = new_carry
                elif rank(l_u[-1]) == rank(carry):
                    new_carry = pair_up(carry, l_u.pop())
                    new_carry.type = 2
                    carry = new_carry
                elif rank(l_v[-1]) == rank(carry):
                    new_carry = pair_up(carry, l_v.pop())
                    new_carry.type = 2
                    carry = new_carry
                else:
                    l.append(carry)
                    carry = None

    return l, 1


def construct(l, r):  # a list of local rank trees, with ranks in an ascending order
    # r is the root
    if len(l) == 0:  # l is empty, generated by the case where r has only one right child that is removed.
        r.left = None
        r.right = None
        r.N = 0
        # r.nontree_bitmap = BitVector(size = 40)
        # r.tree_bitmap = BitVector(size = 40)
        r.nontree_bitmap.setall(0)
        r.tree_bitmap.setall(0)
        r.type = None
        return r, 0
    elif len(l) == 1:  # corner cases1!!!!!!!!!! multiple pointers to leaf nodes
        r.N = l[0].N
        r.right = None
        r.left = l[0]
        l[0].parent = r
        r.nontree_bitmap = l[0].nontree_bitmap.copy()
        r.tree_bitmap = l[0].tree_bitmap.copy()
        return r, 0
    else:
        if len(l) == 2:
            cur = r
        else:
            cur = LTNode(None)  # a connecting (path) node
            cur.type = 3

        left_child = l[0]
        right_child = l[1]

        cur.left = left_child
        cur.right = right_child
        left_child.parent = cur
        right_child.parent = cur

        cur.N = left_child.N + right_child.N
        cur.nontree_bitmap = left_child.nontree_bitmap | right_child.nontree_bitmap
        cur.tree_bitmap = left_child.tree_bitmap | right_child.tree_bitmap

        i = 2
        while i < len(l):
            if i == len(l) - 1:
                par = r
            else:
                par = LTNode(None)  # a connecting (path) node
                par.type = 3

            par.left = cur
            cur.parent = par

            par.right = l[i]
            l[i].parent = par

            par.nontree_bitmap = par.left.nontree_bitmap | par.right.nontree_bitmap
            par.tree_bitmap = par.left.tree_bitmap | par.right.tree_bitmap
            par.N = par.left.N + par.right.N

            cur = par
            i += 1
        r.type = 1
        return r, 1


def insert(u, v, LT, adj_nt_group_by_levels, adj_t_group_by_levels):

    # print("insertion:", u, v)
    if u not in LT or v not in LT:
        if u not in LT and v not in LT:
            LT[u] = LTNode(u)  # a tree node
            # LTV[u].rank = 0
            LT[u].N = 1
            adj_t_group_by_levels[u] = dict()
            adj_t_group_by_levels[u][0] = set()
            LT[u].level = 0

            LT[v] = LTNode(v)  # a tree node
            # LTV[v].rank = 0
            LT[v].N = 1
            adj_t_group_by_levels[v] = dict()
            adj_t_group_by_levels[v][0] = set()
            LT[v].level = 0

            adj_t_group_by_levels[u][0].add(v)  # update adjacency list
            LT[u].tree_bitmap[0] = 1
            adj_t_group_by_levels[v][0].add(u)  # update adjacency list
            LT[v].tree_bitmap[0] = 1

            new_r = LTNode(None)  # a rank node, since left_child and right_child has the same rank
            new_r.left = LT[u]
            LT[u].parent = new_r
            new_r.right = LT[v]
            LT[v].parent = new_r
            new_r.N = 2
            new_r.level = -1
            new_r.type = 2
            # print("initialize a new edge")
        elif u not in LT:
            LT[u] = LTNode(u)
            # LTV[u].rank = 0
            LT[u].N = 1
            adj_t_group_by_levels[u] = dict()
            adj_t_group_by_levels[u][0] = set()
            LT[u].level = 0

            adj_t_group_by_levels[u][0].add(v)  # update adjacency list for tree edges
            LT[u].tree_bitmap[0] = 1
            update_bitmap(LT[u], 0, 1)

            adj_t_group_by_levels[v][0].add(u)  # update adjacency list for tree edges
            LT[v].tree_bitmap[0] = 1
            update_bitmap(LT[v], 0, 1)

            r_v = ancester_at_level(LT[v], -1)  # root has level - 1
            merge(r_v, LT[u])
        else:
            LT[v] = LTNode(v)
            # LTV[v].rank = 0
            LT[v].N = 1
            adj_t_group_by_levels[v] = dict()
            adj_t_group_by_levels[v][0] = set()

            LT[v].level = 0

            adj_t_group_by_levels[u][0].add(v)  # update adjacency list for tree edges
            LT[u].tree_bitmap[0] = 1
            update_bitmap(LT[u], 0, 1)

            adj_t_group_by_levels[v][0].add(u)  # update adjacency list for tree edges
            LT[v].tree_bitmap[0] = 1
            update_bitmap(LT[v], 0, 1)

            r_u = ancester_at_level(LT[u], -1)

            merge(r_u, LT[v])
    else:
        if u in adj_nt_group_by_levels:
            for i in adj_nt_group_by_levels[u].keys():  # filter out redundant non-tree edges
                if v in adj_nt_group_by_levels[u][i]:
                    return
        if u in adj_t_group_by_levels:
            for i in adj_t_group_by_levels[u].keys():  # filter out redundant tree edges
                if v in adj_t_group_by_levels[u][i]:
                    return

        r_u = ancester_at_level(LT[u], -1)
        r_v = ancester_at_level(LT[v], -1)
        if r_u != r_v:
            adj_t_group_by_levels[u][0].add(v)  # update adjacency list
            LT[u].tree_bitmap[0] = 1
            update_bitmap(LT[u], 0, 1)
            adj_t_group_by_levels[v][0].add(u)  # update adjacency list
            LT[v].tree_bitmap[0] = 1
            update_bitmap(LT[v], 0, 1)
            merge(r_u, r_v)
        else:  # (u, v) is a non-tree edge
            # print("add a non-tree edge")
            # print(u, v , "is a non-tree edge")
            if u not in adj_nt_group_by_levels:
                adj_nt_group_by_levels[u] = dict()
                adj_nt_group_by_levels[u][0] = set()

            if v not in adj_nt_group_by_levels:
                adj_nt_group_by_levels[v] = dict()
                adj_nt_group_by_levels[v][0] = set()

            for i in adj_nt_group_by_levels[u].keys():  # filter out redundant edges
                if v in adj_nt_group_by_levels[u][i]:
                    return

            # adj_nt_group_by_levels[u][0].add(v)  # update adjacency list
            if v not in adj_nt_group_by_levels[u][0]:
                adj_nt_group_by_levels[u][0].add(v)  # update adjacency list

            # adj_nt_group_by_levels[v][0].add(u)  # update adjacency list
            if u not in adj_nt_group_by_levels[v][0]:
                adj_nt_group_by_levels[v][0].add(u)  # update adjacency list

            LT[u].nontree_bitmap[0] = 1
            update_bitmap(LT[u], 0, 0)
            LT[v].nontree_bitmap[0] = 1
            update_bitmap(LT[v], 0, 0)

    return


def remove_from_local_rank_tree(Rb, b):  # remove b from a local rank tree
    # nodes are sorted with ranks in an ascending order
    cur = b
    l = []
    while cur != Rb:
        par = cur.parent
        if par.left == cur:
            sibling = par.right
        else:
            sibling = par.left
        l.append(sibling)
        cur = par
    return l, 1


def remove(r, b):  # remove b from the local tree rooted at a, which involves several cases
    # type = 1: tree node in local tree
    # type = 2: node of a local rank tree
    # type = 3: connecting node
    OPS = 0
    if type(r) == 2:  # remove b from a local rank tree
        l, ops_num = remove_from_local_rank_tree(r, b)  # ranks in an ascending order
        OPS += ops_num
        r = construct(l, r)
        OPS += 1
    else:  # remove b from a local tree
        Rb = b
        while type(Rb.parent) == 2:  #
            Rb = Rb.parent
        l, ops_num = remove_connecting_nodes_and_Rb(r, Rb)  # remove connecting nodes and remove Rb from l
        OPS += ops_num
        if Rb == b:
            l.reverse()
            OPS += 1
            r, ops_num = construct(l, r)
            OPS += ops_num
        else:
            lb, ops_num = remove_from_local_rank_tree(Rb, b)
            OPS += ops_num
            lb.reverse()
            OPS += 1
            new_l, ops_num = merge_lists(l, lb)
            OPS += ops_num
            r, ops_num = construct(new_l, r)
            OPS += ops_num
    return r, OPS


def attach(r, b):
    OPS = 0
    if r is None:
        new_r = LTNode(None)
        new_r.N = b.N
        new_r.left = b
        b.parent = new_r
        new_r.nontree_bitmap = b.nontree_bitmap.copy()
        new_r.tree_bitmap = b.tree_bitmap.copy()
        return new_r, OPS

    l_r, ops_num = remove_connecting_nodes(r)
    OPS += ops_num
    l_b = [b]
    new_l, ops_num = merge_lists(l_r, l_b)
    OPS += ops_num
    new_r, ops_num = construct(new_l, r)
    OPS += ops_num
    return new_r, OPS


def merge(a, b):
    # a and b are sibling or both roots
    OPS = 0
    l_a, ops_num = remove_connecting_nodes(a)
    OPS += ops_num
    l_b, ops_num = remove_connecting_nodes(b)
    OPS += ops_num
    new_l, ops_num = merge_lists(l_a, l_b)
    OPS += ops_num
    new_r, ops_num = construct(new_l, a)  # new_r = a
    OPS += ops_num
    return new_r, OPS


def merge_set(C, i):  # m a level-i node
    # there are more than 1 nodes in C

    w = None
    par = None
    for c in C:
        assert c.level is not None

        par = ancester_at_level(c, i - 1)
        par = remove(par, c)

        if c.val is not None and len(C) > 1:
            c.level = i + 1  # promote leaf node

        w = attach(w, c)

    w.level = i
    return par, w


def create_Tv(C, P, LT, adj_t_group_by_levels, level):

    #for u, v in P:  # promoting levels on edges
    #    tree_edge_level_up(u, v, LTV, adj_t_group_by_levels, level)
    OPS = 0
    g = None
    if len(C) == 1:
        # there is only one cluster node in C, don't merge, which was not mentioned in the paper.
        tn = next(iter(C))
        par = ancester_at_level(tn, level - 1)

        # remove par from its parent since par changes
        if par.parent is not None:
            g = ancester_at_level(par, par.level - 1)
        if g is not None:
            g, ops_num = remove(g, par)
            OPS += ops_num
        par, ops_num = remove(par, tn)
        OPS += ops_num
        return g, par, tn, OPS

    Tv = None
    par = None
    for c in C:
        # assert c.level is not None
        if par is None:
            par = ancester_at_level(c, level - 1)

            # remove par from its parent since par changes
            if par.parent is not None:
                g = ancester_at_level(par, par.level - 1)
            if g is not None:
                g, ops_num = remove(g, par)
                OPS += ops_num

        par, ops_num = remove(par, c)
        OPS += ops_num

        if c.val is not None:
            c.level = level + 1  # promote leaf node

        if Tv is None:
            if c.val is not None:
                Tv = LTNode(None)
                Tv.N = c.N
                Tv.left = c
                c.parent = Tv
                Tv.tree_bitmap = c.tree_bitmap.copy()
                Tv.nontree_bitmap = c.nontree_bitmap.copy()
                Tv.level = level
            else:
                Tv = c
        else:
            l_Tv, ops_num = remove_connecting_nodes(Tv)
            OPS += ops_num
            l_c, ops_num = remove_connecting_nodes(c)
            OPS += ops_num
            new_l, ops_num = merge_lists(l_Tv, l_c)
            OPS += ops_num
            Tv, ops_num = construct(new_l, Tv)
            OPS += ops_num
    return g, par, Tv, OPS


def delete(u, v, LT, adj_nt_group_by_levels, adj_t_group_by_levels):
    OPS = 0
    i_t = get_level(u, v, adj_t_group_by_levels)  # determine level-i tree edge
    i_nt = get_level(u, v, adj_nt_group_by_levels)  # determine level-i non-tree edge
    if i_t == -1 and i_nt == -1:
        return
    # if edge_type(u, v, adj_t_group_by_levels) == 0:  # (u, v) is a non-tree edge
    if i_nt != -1:
        i = i_nt
        adj_nt_group_by_levels[u][i].remove(v)
        adj_nt_group_by_levels[v][i].remove(u)

        if len(adj_nt_group_by_levels[u][i]) == 0:  # update bitmap
            LT[u].nontree_bitmap[i] = 0
            update_bitmap(LT[u], i, 0)

        if len(adj_nt_group_by_levels[v][i]) == 0:  # update bitmap
            LT[v].nontree_bitmap[i] = 0
            update_bitmap(LT[v], i, 0)

        return 0

    # remove edges (u, v)
    i = i_t
    # print("deleting tree edge:", u, v, ", level:", i)
    adj_t_group_by_levels[u][i].remove(v)
    adj_t_group_by_levels[v][i].remove(u)

    if len(adj_t_group_by_levels[u][i]) == 0:  # update bitmap for tree edge
        LT[u].tree_bitmap[i] = 0
        update_bitmap(LT[u], i, 1)

    if len(adj_t_group_by_levels[v][i]) == 0:  # update bitmap for tree edge
        LT[v].tree_bitmap[i] = 0
        update_bitmap(LT[v], i, 1)
    count = 0
    while i >= 0:
        anc_u = ancester_at_level(LT[u], i)
        stack_u = []  # DFS_u:  DFS starts at anc_u
        edges_u = []  # candidate edges to be visited by DFS_u
        stack_u.append(anc_u)

        C_u = set()
        C_u.add(anc_u)
        P_u = set()
        size_u = anc_u.N

        anc_v = ancester_at_level(LT[v], i)
        stack_v = []  # DFS_v: DFS starts at anc_v
        edges_v = []  # candidate edges to be visited by DFS_v
        stack_v.append(anc_v)
        C_v = set()
        C_v.add(anc_v)
        P_v = set()
        size_v = anc_v.N
        start = timer()
        while (len(stack_u) > 0 or len(edges_u) > 0) and (len(stack_v) > 0 or len(edges_v) > 0):
            if len(edges_u) == 0:
                search(stack_u, i, adj_t_group_by_levels, edges_u)
            else:
                vertex_u, adj_u = edges_u.pop()
                anc_adj_u = ancester_at_level(LT[adj_u], i)
                count += 1
                x, y = order(vertex_u, adj_u)
                P_u.add((x, y))
                if anc_adj_u not in C_u:
                    stack_u.append(anc_adj_u)
                    C_u.add(anc_adj_u)
                    size_u += anc_adj_u.N

            if len(edges_v) == 0:
                search(stack_v, i, adj_t_group_by_levels, edges_v)
            else:
                vertex_v, adj_v = edges_v.pop()
                anc_adj_v = ancester_at_level(LT[adj_v], i)
                count += 1
                x, y = order(vertex_v, adj_v)
                P_v.add((x, y))
                if anc_adj_v not in C_v:
                    stack_v.append(anc_adj_v)
                    C_v.add(anc_adj_v)
                    size_v += anc_adj_v.N

        if len(stack_u) == 0 and len(edges_u) == 0:
            while (len(stack_v) > 0 or len(edges_v) > 0) and size_v <= size_u:
                if len(edges_v) == 0:
                    search(stack_v, i, adj_t_group_by_levels, edges_v)
                else:
                    vertex_v, adj_v = edges_v.pop()
                    anc_adj_v = ancester_at_level(LT[adj_v], i)
                    count += 1
                    x, y = order(vertex_v, adj_v)
                    P_v.add((x, y))
                    if anc_adj_v not in C_v:
                        stack_v.append(anc_adj_v)
                        C_v.add(anc_adj_v)
                        size_v += anc_adj_v.N

        if len(stack_v) == 0 and len(edges_v) == 0:
            while (len(stack_u) > 0 or len(edges_u) > 0) and size_u <= size_v:
                if len(edges_u) == 0:
                    search(stack_u, i, adj_t_group_by_levels, edges_u)
                else:
                    vertex_u, adj_u = edges_u.pop()
                    anc_adj_u = ancester_at_level(LT[adj_u], i)
                    count += 1
                    x, y = order(vertex_u, adj_u)
                    P_u.add((x, y))
                    if anc_adj_u not in C_u:
                        stack_u.append(anc_adj_u)
                        C_u.add(anc_adj_u)
                        size_u += anc_adj_u.N
        '''
        t1 = timer() - start
        if size_u <= size_v:  # nodes in C_u are made Tv
            for u, v in P_u:  # promoting levels on edges
                tree_edge_level_up(u, v, LT, adj_t_group_by_levels, i)
            start = timer()
            g, par, Tv, ops_num = create_Tv(C_u, P_u, LT, adj_t_group_by_levels, i)
            OPS += ops_num
        else:  # nodes in C_v are made Tv
            for u, v in P_v:  # promoting levels on edges
                tree_edge_level_up(u, v, LT, adj_t_group_by_levels, i)
            start = timer()
            g, par, Tv, ops_num = create_Tv(C_v, P_v, LT, adj_t_group_by_levels, i)
            OPS += ops_num
        t2 = timer() - start
        '''

        if size_u <= size_v:  # nodes in C_u are made Tv
            C = C_u
            P = P_u
        else:  # nodes in C_v are made Tv
            C = C_v
            P = P_v

        g, par, Tv, ops_num = create_Tv(C, P, LT, adj_t_group_by_levels, i)

        #print("%d nodes change levels" %(min(size_u, size_v)), "total run time: %f" %t2,
        #      "average runtime: %f" %(t2/(min(size_u, size_v))))
        #print("total run time of searching nodes: %f" %t1)

        tx, ty, nontree_edges = search_replacement_edge(Tv, adj_nt_group_by_levels, LT, i)
        if tx is not None and ty is not None:  # (x, y) is a replacement edge, case 1 for deletion

            if len(C) > 1:
                for u, v in P:  # promoting levels on edges
                    tree_edge_level_up(u, v, LT, adj_t_group_by_levels, i)

            adj_nt_group_by_levels[tx][i].remove(ty)
            adj_nt_group_by_levels[ty][i].remove(tx)

            if len(adj_nt_group_by_levels[tx][i]) == 0:  # update bitmap
                LT[tx].nontree_bitmap[i] = 0
                update_bitmap(LT[tx], i, 0)

            if len(adj_nt_group_by_levels[ty][i]) == 0:  # update bitmap
                LT[ty].nontree_bitmap[i] = 0
                update_bitmap(LT[ty], i, 0)

            adj_t_group_by_levels[tx][i].add(ty)
            LT[tx].tree_bitmap[i] = 1
            update_bitmap(LT[tx], i, 1)

            adj_t_group_by_levels[ty][i].add(tx)
            LT[ty].tree_bitmap[i] = 1
            update_bitmap(LT[ty], i, 1)

            par, ops_num = attach(par, Tv)
            OPS += ops_num
            if g is not None:
                g, ops_num = attach(g, par)
                OPS += ops_num
            for nu, nv in nontree_edges:
                nontree_edge_level_up(nu, nv, LT, adj_nt_group_by_levels, i)
            return count
        else:  # case 2 for deletion
            if i > 0 and len(C) > 1:
                for u, v in P:  # promoting levels on edges
                    tree_edge_level_up(u, v, LT, adj_t_group_by_levels, i)
                for nu, nv in nontree_edges:
                    nontree_edge_level_up(nu, nv, LT, adj_nt_group_by_levels, i)

            p_prime = LTNode(None)
            p_prime.left = Tv
            Tv.parent = p_prime
            p_prime.N = Tv.N
            p_prime.nontree_bitmap = Tv.nontree_bitmap.copy()
            p_prime.tree_bitmap = Tv.tree_bitmap.copy()
            p_prime.level = i - 1

            if g is not None:
                g, ops_num = attach(g, par)
                OPS += ops_num
                g, ops_num = attach(g, p_prime)
                OPS += ops_num
        i -= 1
    return count


def search(stack, i, adj_list_group_by_levels, edges):
    # amortized cost O(h)
    if len(stack) == 0:
        return

    cur = stack.pop()
    if cur.val is not None:
        for adj_u in adj_list_group_by_levels[cur.val][i]:
            edges.append((cur.val, adj_u))

    if cur.left is not None and cur.left.tree_bitmap[i] != 0:
        stack.append(cur.left)
    if cur.right is not None and cur.right.tree_bitmap[i] != 0:
        stack.append(cur.right)

    return


def search_replacement_edge(x, adj_nt_group_by_levels, LT, i):
    stack = [x]
    nontree_edges = set()
    while len(stack) != 0:
        cur = stack.pop()
        if cur.val is not None:
            if cur.val not in adj_nt_group_by_levels or i not in adj_nt_group_by_levels[cur.val] or \
                    len(adj_nt_group_by_levels[cur.val][i]) == 0:
                continue
            for adj_nt_x in adj_nt_group_by_levels[cur.val][i]:
                if ancester_at_level(LT[adj_nt_x], i) != ancester_at_level(LT[cur.val], i):
                    return adj_nt_x, cur.val, nontree_edges
                else:
                    nt_x, nt_y = order(adj_nt_x, cur.val)
                    if (nt_x, nt_y) not in nontree_edges:
                        nontree_edges.add((nt_x, nt_y))

        if cur.left is not None and cur.left.nontree_bitmap[i] != 0:
            stack.append(cur.left)
        if cur.right is not None and cur.right.nontree_bitmap[i] != 0:
            stack.append(cur.right)

    return None, None, None


def ancester_at_level(node, i):
    anc = node
    if i == -1:
        while anc.parent is not None:
            anc = anc.parent
        return anc
    else:
        while anc is not None:
            if anc.level == i:  # tree node
                return anc
            anc = anc.parent
        return None


def get_level(u, v, adjacency_level):
    if u in adjacency_level:
        for level, adj in adjacency_level[u].items():
            if v in adj:
                return level
    return -1


def cal_node_size(node):
    node_size = 0
    if node.parent is not None:
        node_size += sys.getsizeof(node.parent)
    if node.left is not None:
        node_size += sys.getsizeof(node.left)
    if node.right is not None:
        node_size += sys.getsizeof(node.right)

    node_size += sys.getsizeof(node.val)
    node_size += sys.getsizeof(node.N)
    node_size += sys.getsizeof(node.level)
    node_size += sys.getsizeof(node.type)
    if hasattr(node, 'tree_bitmap'):
        node_size += sys.getsizeof(node.tree_bitmap)
    if hasattr(node, 'nontree_bitmap'):
        node_size += sys.getsizeof(node.nontree_bitmap)
    if hasattr(node, 'bitmap'):
        node_size += sys.getsizeof(node.bitmap)
    return node_size


def cal_total_memory_use(LT, adj_t_group_by_levels, adj_nt_group_by_levels):
    print("memory size for dictionary: %d bytes" % sys.getsizeof(LT))
    # get all root nodes
    root_node_set = set()
    for _, node in LT.items():
        while node.parent is not None:
            node = node.parent
        root_node_set.add(node)

    node_size = 0
    for root_node in root_node_set:
        q = Queue()
        q.put(root_node)
        while not q.empty():
            node = q.get()
            node_size += cal_node_size(node)
            if node.left is not None:
                q.put(node.left)
            if node.right is not None:
                q.put(node.right)
    node_size += sys.getsizeof(LT)
    tree_edge_size = 0
    for vertex, adj_list_dict in adj_t_group_by_levels.items():
        for level, vertice_set in adj_list_dict.items():
            if vertice_set is not None:
                tree_edge_size += sys.getsizeof(vertice_set)

    nontree_edge_size = 0
    for vertex, adj_list_dict in adj_nt_group_by_levels.items():
        for level, vertice_set in adj_list_dict.items():
            if vertice_set is not None:
                nontree_edge_size += sys.getsizeof(vertice_set)
    edge_size = tree_edge_size + nontree_edge_size
    print("total memory size : %d bytes" % (node_size + edge_size))
    return node_size, edge_size


def validateLT1(r):
    level_check(r)
    validate_local_rank_rank_tree(r)
    q = Queue()
    q.put(r)
    visited = set()
    while not q.empty():
        cur = q.get()
        validate_children(cur)
        validate_rank(cur)
        visited.add(cur)
        if cur.level == -1 and cur.parent is not None:
            raise ValueError("level == -1 but not root ")
        if cur.parent is None and cur.level is None:
            raise ValueError("level == -1 but not root ")

        if cur.left is not None and cur.right is None:
            raise ValueError("Right child is None while left is not")

        if cur.left is None and cur.right is not None:
            assert cur.N == cur.right.N

        if cur.left is not None and cur.right is not None:
            #    if cur.left.rank > cur.right.rank:
            #        raise ValueError("cur.left.rank: %d , cur.right.rank: %d" %(cur.left.rank, cur.right.rank))

            if cur.left.N + cur.right.N != cur.N:
                raise ValueError("Left.N + Right.N != Parent.N", cur.left.N + cur.right.N,  cur.N)
        if cur.right is not None and cur.left is None:
            if cur.N != cur.right.N:
                raise ValueError("wrong N value")

        if cur.left is not None:
            assert cur.left.parent == cur
            if cur.left in visited:
                raise ValueError("wrong linking for .....", cur.left)
            q.put(cur.left)
        if cur.right is not None:
            assert cur.right.parent == cur
            if cur.right in visited:
                raise ValueError("wrong linking for .....", cur.right)
            q.put(cur.right)

        # if cur.left is None and cur.right is None:
        #    if cur in visited:
        #        raise ValueError("wrong linking for %d" %cur.val)
        #    visited.add(cur)

    return


def size_check(r):
    q = Queue()
    q.put(r)
    visited = set()
    while not q.empty():
        cur = q.get()

        # if cur.rank is not None and cur.rank != int(math.floor(math.log(cur.N, 2))):
        #    raise ValueError("rank != log N", cur.rank, cur.N)

        if cur.left is None and cur.right is not None:
            assert cur.N == cur.right.N

        if cur.left is not None and cur.right is not None:
            # if cur.left.rank is None or cur.right.rank is None:
            #    left_rank = int(math.log(cur.left.N, 2))
            #    right_rank = int(math.log(cur.right.N, 2))
            #    if left_rank > right_rank:
            #        raise ValueError("left_rank: %d , right_rank: %d" % (left_rank, right_rank))
            # else:
            #    if cur.left.rank > cur.right.rank:
            #        raise ValueError("cur.left.rank: %d , cur.right.rank: %d" % (cur.left.rank, cur.right.rank))

            if cur.left.N + cur.right.N != cur.N:
                raise ValueError("Left.N + Right.N != Parent.N", cur.left.N + cur.right.N,  cur.N)
        if cur.right is not None and cur.left is None:
            if cur.N != cur.right.N:
                raise ValueError("wrong N value")

        if cur.left is not None:
            if cur.left in visited:
                raise ValueError("wrong linking for %d....." %cur.left)
            q.put(cur.left)
        if cur.right is not None:
            if cur.right in visited:
                raise ValueError("wrong linking for %d....." %cur.left)
            q.put(cur.right)

        if cur.left is None and cur.right is None:
            if cur in visited:
                raise ValueError("wrong linking for %d....." %cur.val)
            visited.add(cur)

    return


def level_check(r):
    # print("level check...")
    q = Queue()
    q.put(r)
    current_level = r.level
    while not q.empty():
        new_q = Queue()
        while not q.empty():
            cur = q.get()
            if cur.level is not None:
                # print(cur.level)
                assert cur.level == current_level

            if cur.left is not None:
                if cur.left.level is not None:
                    new_q.put(cur.left)
                else:
                    q.put(cur.left)

            if cur.right is not None:
                if cur.right.level is not None:
                    if cur.right.level is not None:
                        new_q.put(cur.right)
                    else:
                        q.put(cur.right)
        q = new_q
        current_level += 1

    return


def validate_local_rank_rank_tree(r):
    if type(r) == 2:
        q = Queue()
        q.put(r)
        current_rank = rank(r)
        initial_rank = rank(r)
        while not q.empty():
            new_q = Queue()
            while not q.empty():
                cur = q.get()
                assert rank(cur) == current_rank
                if cur.left is not None and cur.left.level is None and type(cur.left) == 2:
                    new_q.put(cur.left)
                if cur.right is not None and cur.right.level is None and type(cur.left) == 2:
                    new_q.put(cur.right)
            q = new_q
            current_rank -= 1

    return


def validate_children(n):
    q = Queue()
    q.put(n)
    visited = set()
    while not q.empty():
        cur = q.get()
        if cur.left is not None:
            assert cur.left.parent == cur
            if cur.left in visited:
                raise ValueError("wrong linking for .....", cur.left)
            q.put(cur.left)
        if cur.right is not None:
            assert cur.right.parent == cur
            if cur.right in visited:
                raise ValueError("wrong linking for .....", cur.right)
            q.put(cur.right)
    return


def validate_rank(node):
    if type(node) == 2:
        temp_r = node
        q = Queue()
        q.put(node)
        current_rank = rank(node)
        initial_rank = rank(node)
        while not q.empty():
            new_q = Queue()
            while not q.empty():
                cur = q.get()
                assert rank(cur) == current_rank or rank(cur) == current_rank - 1
                if cur.left is not None and cur.left.level is None and type(cur.left) == 2:
                    new_q.put(cur.left)
                if cur.right is not None and cur.right.level is None and type(cur.left) == 2:
                    if cur.N == 1:
                        temp = cur.right
                        while temp is not None:
                            assert temp.N == 1
                            temp = temp.right
                    else:
                        new_q.put(cur.right)
            q = new_q
            current_rank -= 1
    return


def validate_list(l, direction):
    # direction = True, ascending
    # direction = False, descending
    new_lt = []
    for cur in l:
        validate_children(cur)
        new_lt.append(cur.N)

    for i in range(len(new_lt) - 1):
        assert rank(l[i]) != rank(l[i + 1])
        if new_lt[i] > new_lt[i + 1] and direction:
            raise ValueError("value errors in l", new_lt)
        if new_lt[i] < new_lt[i + 1] and not direction:
            raise ValueError("value errors in l", new_lt)
