from LTV.LTNode import LTNode
from ST.ST_utils import toRoot
from utils.tree_utils import order
from queue import Queue
import math
from bitarray import bitarray
from LT.LT_utils import cal_node_edge_size
import sys


def query(LT, u, v):
    if u not in LT or v not in LT:
        return False
    return toRoot(LT[u]) == toRoot(LT[v])


def level_up(u, v, LT, adj_list_group_by_levels, level):
    adj_list_group_by_levels[u][level].remove(v)
    adj_list_group_by_levels[v][level].remove(u)

    if len(adj_list_group_by_levels[u][level]) == 0:
        LT[u].bitmap[level] = 0  # update bitmap
        update_bitmap(LT[u], level)

    if len(adj_list_group_by_levels[v][level]) == 0:
        LT[v].bitmap[level] = 0  # update bitmap
        update_bitmap(LT[v], level)

    if level + 1 not in adj_list_group_by_levels[u]:
        adj_list_group_by_levels[u][level + 1] = set()

    if level + 1 not in adj_list_group_by_levels[v]:
        adj_list_group_by_levels[v][level + 1] = set()

    adj_list_group_by_levels[u][level + 1].add(v)
    adj_list_group_by_levels[v][level + 1].add(u)

    # update bitmap
    if LT[u].bitmap[level + 1] == 0:
        LT[u].bitmap[level + 1] = 1
        update_bitmap(LT[u], level + 1)

    if LT[v].bitmap[level + 1] == 0:
        LT[v].bitmap[level + 1] = 1
        update_bitmap(LT[v], level + 1)

    return


def update_bitmap(node, level):  # level i
    # only looking at the sibling's bitmap[level] == 1
    cur = node
    p = node.parent
    while p is not None:
        if cur == p.left:
            sibling = p.right
        else:
            sibling = p.left

        if sibling is None:
            p.bitmap[level] = cur.bitmap[level]
        else:
            if sibling.bitmap[level] == 1:
                return
            else:
                p.bitmap[level] = cur.bitmap[level] | sibling.bitmap[level]
        cur = p
        p = p.parent
    return


def type(node):
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


def rank(node):
    if node.N <= 0:
        return -1
    return int(math.floor(math.log(node.N, 2)))


def remove_connecting_nodes(r):
    # given a root of a local tree, remove all connecting nodes and return a list of local rank trees
    # with ranks in an descending order

    if type(r) == 2:  # r is a local rank tree
        l = [pair_up(r.left, r.right)]
    else:
        if r.val is not None:
            l= [r]
        else:
            l = []
            cur = r
            while cur is not None:
                if cur.right is not None:
                    l.append(cur.right)
                else:
                    assert cur.left is None
                if cur.left is not None and type(cur.left) != 3:
                    l.append(cur.left)
                    break
                cur = cur.left
    return l


def remove_connecting_nodes_and_Rb(r, Rb):
    # given a root of a local tree, remove all connecting nodes and Rb
    # return a list of local rank trees with ranks in an descending order
    l = []
    cur = r
    while cur is not None:
        if cur.right != Rb:
            l.append(cur.right)
        if cur.left is not None and type(cur.left) != 3:
            if cur.left != Rb:
                l.append(cur.left)
            break
        cur = cur.left
    return l


def pair_up(c1, c2):  # c1.rank = c2.rank
    p = LTNode(None)  # make cur parent of left_child and right_child
    if c1.N < c2.N:
        left_child = c1
        right_child = c2
    else:
        left_child = c2
        right_child = c1
    #p.rank = left_child.rank + 1  # increases rank by 1
    p.N = left_child.N + right_child.N  # update size
    p.bitmap = left_child.bitmap | right_child.bitmap  # update bitmap

    # if type(left_child) == 2:
    #    left_child.level = None
    # if type(right_child) == 2:
    #    right_child.level = None

    p.left = left_child
    left_child.parent = p

    p.right = right_child
    right_child.parent = p
    p.type = 2
    return p


def merge_lists(l_u, l_v):
    # this function shows code for algorithm merge in the paper for readers
    # Note that merge_list1 is not used in the actual implementation because it is less efficient than merge_list

    if l_u is None or len(l_u) == 0:
        return l_v
    if l_v is None or len(l_v) == 0:
        return l_u

    # preprocess
    max_rank = max(rank(l_u[0]), rank(l_v[0]))
    S_u = []
    S_v = []
    l_u.reverse()
    l_v.reverse()
    for rk in range(max_rank, -1, -1):
        if len(l_u) > 0 and rank(l_u[-1]) == rk:
            S_u.append(l_u.pop())
        else:
            S_u.append(None)

        if len(l_v) > 0 and rank(l_v[-1]) == rk:
            S_v.append(l_v.pop())
        else:
            S_v.append(None)
    l = []

    carry = None
    while len(S_u) != 0 or len(S_v) != 0:  # pair up nodes with the same rank
        count = 0
        s = S_u.pop()
        v = S_v.pop()
        if s is None:
            count += 1
        if v is None:
            count += 1
        if carry is None:
            count += 1

        if count == 3:
            continue
        elif count == 2:
            if carry is not None:
                l.append(carry)
                carry = None
            elif s is not None:
                l.append(s)
            else:
                l.append(v)
        elif count == 1:
            if carry is None:
                carry = pair_up(s, v)
            elif s is None:
                carry = pair_up(carry, v)
            else:
                carry = pair_up(carry, s)
        elif count == 0:
            if s.N <= carry.N and s.N <= v.N:
                l.append(s)
                carry = pair_up(carry, v)
            elif v.N <= carry.N and v.N <= s.N:  # l_v[-1].N is the smallest
                l.append(v)
                carry = pair_up(carry, s)
            else:  # carry.N is the smallest
                l.append(carry)
                carry = pair_up(s, v)
    if carry is not None:
        l.append(carry)

    return l


def merge_lists1(l_u, l_v):
    # nodes in both l_u and l_v are sorted by ranks in a descending order
    # nodes in l are sorted by ranks in an ascending order

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

    return l


def construct(l, r):  # a list of local rank trees, with ranks in an ascending order
    # r is the root
    if len(l) == 0:  # l is empty, generated by the case where r has only one right child that is removed.
        r.left = None
        r.right = None
        r.N = 0
        r.bitmap = bitarray(64)
        r.bitmap.setall(0)
        r.type = None
        return r
    elif len(l) == 1:  # corner cases1!!!!!!!!!! multiple pointers to leaf nodes
        if l[0].level is None:
            r.N = l[0].N
            r.bitmap = l[0].bitmap.copy()
            r.left = l[0].left
            if l[0].left is not None:
                l[0].left.parent = r

            r.right = l[0].right
            if l[0].right is not None:
                l[0].right.parent = r

            if rank(l[0].left) == rank(l[0].right):
                r.type = 2
            else:
                r.type = 1
        else:
            r.left = None
            r.right = l[0]
            r.N = l[0].N
            r.bitmap = l[0].bitmap.copy()
            l[0].parent = r

        return r
    else:
        if len(l) == 2:
            cur = r
        else:
            cur = LTNode(None)  # a connecting (path) node

        left_child = l[0]
        right_child = l[1]

        cur.left = left_child
        cur.right = right_child
        left_child.parent = cur
        right_child.parent = cur

        cur.N = left_child.N + right_child.N
        cur.bitmap = left_child.bitmap | right_child.bitmap

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

            par.bitmap = par.left.bitmap | par.right.bitmap
            par.N = par.left.N + par.right.N

            cur = par
            i += 1
        r.type = 1
        return r


def insert(u, v, LT, adj_list_group_by_levels):
    #print("insert %d-%d" %(u, v))
    #if u == 95 and v == 102:
    #    print("check..")
    if u not in LT or v not in LT:
        if u not in LT and v not in LT:
            LT[u] = LTNode(u)  # a tree node
            # LT[u].rank = 0
            LT[u].N = 1
            adj_list_group_by_levels[u] = dict()
            adj_list_group_by_levels[u][0] = set()
            LT[u].level = 0

            LT[v] = LTNode(v)  # a tree node
            # LT[v].rank = 0
            LT[v].N = 1
            adj_list_group_by_levels[v] = dict()
            adj_list_group_by_levels[v][0] = set()
            LT[v].level = 0

            adj_list_group_by_levels[u][0].add(v)  # update adjacency list
            LT[u].bitmap[0] = 1
            adj_list_group_by_levels[v][0].add(u)  # update adjacency list
            LT[v].bitmap[0] = 1

            new_r = LTNode(None)  # a rank node, since left_child and right_child has the same rank
            new_r.left = LT[u]
            LT[u].parent = new_r
            new_r.right = LT[v]
            LT[v].parent = new_r
            new_r.N = 2
            # new_r.rank = 1
            new_r.bitmap[0] = 1
            new_r.level = -1
            new_r.type = 2

        elif u not in LT:
            LT[u] = LTNode(u)
            # LT[u].rank = 0
            LT[u].N = 1
            adj_list_group_by_levels[u] = dict()
            adj_list_group_by_levels[u][0] = set()
            LT[u].level = 0

            adj_list_group_by_levels[u][0].add(v)  # update adjacency list

            LT[u].bitmap[0] = 1
            update_bitmap(LT[u], 0)

            adj_list_group_by_levels[v][0].add(u)  # update adjacency list
            LT[v].bitmap[0] = 1
            update_bitmap(LT[v], 0)

            r_v = ancestor_at_i(LT[v], -1)  # root has level - 1
            l_v = remove_connecting_nodes(r_v)

            l = merge_lists([LT[u]], l_v)

            r_v = construct(l, r_v)
        else:
            LT[v] = LTNode(v)
            # LT[v].rank = 0
            LT[v].N = 1
            adj_list_group_by_levels[v] = dict()
            adj_list_group_by_levels[v][0] = set()
            LT[v].level = 0

            adj_list_group_by_levels[u][0].add(v)  # update adjacency list
            LT[u].bitmap[0] = 1
            update_bitmap(LT[u], 0)

            adj_list_group_by_levels[v][0].add(u)  # update adjacency list
            LT[v].bitmap[0] = 1
            update_bitmap(LT[v], 0)

            r_u = ancestor_at_i(LT[u], -1)
            l_u = remove_connecting_nodes(r_u)

            l = merge_lists(l_u, [LT[v]])

            r_u = construct(l, r_u)

    else:
        for i in adj_list_group_by_levels[u].keys():  # filter out redundant edges
            if v in adj_list_group_by_levels[u][i]:
                return

        adj_list_group_by_levels[v][0].add(u)
        adj_list_group_by_levels[u][0].add(v)

        LT[u].bitmap[0] = 1
        update_bitmap(LT[u], 0)
        LT[v].bitmap[0] = 1
        update_bitmap(LT[v], 0)

        r_u = ancestor_at_i(LT[u], -1)
        r_v = ancestor_at_i(LT[v], -1)
        if r_u != r_v:
            l_u = remove_connecting_nodes(r_u)
            l_v = remove_connecting_nodes(r_v)
            l = merge_lists(l_u, l_v)
            r_u = construct(l, r_u)
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
    return l


def remove(r, b):  # remove b from the local tree rooted at a, which involves several cases
    # type = 1: tree node in local tree
    # type = 2: node of a local rank tree
    # type = 3: connecting node
    if type(r) == 2:  # remove b from a local rank tree
        l = remove_from_local_rank_tree(r, b)  # ranks in an ascending order
        r = construct(l, r)
    else:  # remove b from a local tree
        Rb = b
        while type(Rb.parent) == 2:  #
            Rb = Rb.parent
        l = remove_connecting_nodes_and_Rb(r, Rb)  # remove connecting nodes and remove Rb from l
        if Rb == b:
            l.reverse()
            r = construct(l, r)
        else:
            lb = remove_from_local_rank_tree(Rb, b)
            lb.reverse()
            new_l = merge_lists(l, lb)
            r = construct(new_l, r)
    return r


def attach(r, b, level):

    if r is None:
        new_r = LTNode(None)
        new_r.N = b.N
        new_r.right = b
        b.parent = new_r
        new_r.bitmap = b.bitmap.copy()
        new_r.level = level - 1
        return new_r

    l_r = remove_connecting_nodes(r)
    l_b = [b]
    new_l = merge_lists(l_r, l_b)
    new_r = construct(new_l, r)

    return new_r


def merge(C, i):  # m a level-i node

    w = None
    par = None
    for c in C:
        assert c.level is not None

        if par is None:
            par = ancestor_at_i(c, i - 1)

        par = remove(par, c)

        if c.val is not None and len(C) > 1:
            c.level = i + 1  # promote leaf node

        if w is None:
            if c.val is not None and len(C) > 1:
                w = LTNode(None)
                w.N = c.N
                w.right = c
                c.parent = w
                w.bitmap = c.bitmap.copy()
                w.level = i
            else:
                w = c
        else:
            l_w = remove_connecting_nodes(w)

            l_c = remove_connecting_nodes(c)
            new_l = merge_lists(l_w, l_c)
            w = construct(new_l, w)

    return par, w


def delete_case1(C, P, LT, adj_list_group_by_levels, level):
    if len(P) == 0:
        return

    for u, v in P:  # promoting levels on edges
        if u == v:
            continue
        level_up(u, v, LT, adj_list_group_by_levels, level)

    if len(C) == 1:
        #assert next(iter(C)).val is None
        #assert C[0].val is None
        # there is only one cluster node in C, don't merge, which was not detailed in the paper.
        return

    par, w = merge(C, level)
    par = attach(par, w, level)
    return par


def delete_case2(sV, P, LT, adj_list_group_by_levels, level):

    for u, v in P:  # promoting levels on edges
        if u == v:
            continue
        level_up(u, v, LT, adj_list_group_by_levels, level)

    g = None
    par = None
    for s in sV:
        par = ancestor_at_i(s, level - 1)
        g = ancestor_at_i(s, level - 2)
        break

    if g is not None:
        g = remove(g, par)

    par, w = merge(sV, level)

    if level == 0 and w.N == 1:
        cur = w
        while cur is not None:
            temp = cur.right
            if cur.val is not None:
                del LT[cur.val]
            del cur
            cur = temp
    else:
        p_prime = LTNode(None)
        p_prime.right = w
        w.parent = p_prime
        p_prime.N = w.N
        p_prime.bitmap = w.bitmap.copy()
        p_prime.level = level - 1

    if g is not None:
        g = attach(g, par, par.level)
        g = attach(g, p_prime, p_prime.level)
        return g
    else:
        return par


def delete(u, v, LT, adj_list_group_by_levels):

    i = get_level(u, v, adj_list_group_by_levels)

    if i == -1:
        return

    # remove edges (u, v) and (v, u)
    adj_list_group_by_levels[u][i].remove(v)
    adj_list_group_by_levels[v][i].remove(u)

    if len(adj_list_group_by_levels[u][i]) == 0:  # update bitmap
        LT[u].bitmap[i] = 0
        update_bitmap(LT[u], i)

    if len(adj_list_group_by_levels[v][i]) == 0:  # update bitmap
        LT[v].bitmap[i] = 0
        update_bitmap(LT[v], i)

    while i >= 0:
        anc_u = ancestor_at_i(LT[u], i)
        stack_u = []  # DFS_u:  DFS starts at anc_u
        edges_u = []  # candidate edges to be visited by DFS_u
        stack_u.append(anc_u)
        C_u = set()
        C_u.add(anc_u)
        P_u = set()
        size_u = anc_u.N

        anc_v = ancestor_at_i(LT[v], i)
        stack_v = []  # DFS_v: DFS starts at anc_v
        edges_v = []  # candidate edges to be visited by DFS_v
        stack_v.append(anc_v)
        C_v = set()
        C_v.add(anc_v)
        P_v = set()
        size_v = anc_v.N

        # assert anc_v.level == anc_u.level == i

        while (len(stack_u) > 0 or len(edges_u) > 0) and (len(stack_v) > 0 or len(edges_v) > 0):
            if len(edges_u) == 0:
                search(stack_u, i, adj_list_group_by_levels, edges_u)
            else:
                vertex_u, adj_u = edges_u.pop()
                anc_adj_u = ancestor_at_i(LT[adj_u], i)
                # assert anc_adj_u.level == i
                if anc_adj_u in C_v:  # case 1
                    if size_u < size_v:
                        delete_case1(C_u, P_u, LT, adj_list_group_by_levels, i)
                    else:
                        delete_case1(C_v, P_v, LT, adj_list_group_by_levels, i)
                    return
                x, y = order(vertex_u, adj_u)
                P_u.add((x, y))
                if anc_adj_u not in C_u:
                    stack_u.append(anc_adj_u)
                    C_u.add(anc_adj_u)
                    size_u += anc_adj_u.N

            if len(edges_v) == 0:
                search(stack_v, i, adj_list_group_by_levels, edges_v)
            else:
                vertex_v, adj_v = edges_v.pop()
                anc_adj_v = ancestor_at_i(LT[adj_v], i)

                if anc_adj_v in C_u:  # case 1
                    if size_u < size_v:
                        delete_case1(C_u, P_u, LT, adj_list_group_by_levels, i)
                    else:
                        delete_case1(C_v, P_v, LT, adj_list_group_by_levels, i)
                    return
                x, y = order(vertex_v, adj_v)
                P_v.add((x, y))
                # assert anc_adj_v.level == i
                if anc_adj_v not in C_v:
                    stack_v.append(anc_adj_v)
                    C_v.add(anc_adj_v)
                    size_v += anc_adj_v.N

        if len(stack_u) == 0 and len(edges_u) == 0:

            while (len(stack_v) > 0 or len(edges_v) > 0) and size_v <= size_u:
                if len(edges_v) == 0:
                    search(stack_v, i, adj_list_group_by_levels, edges_v)
                else:
                    vertex_v, adj_v = edges_v.pop()
                    anc_adj_v = ancestor_at_i(LT[adj_v], i)
                    x, y = order(vertex_v, adj_v)
                    P_v.add((x, y))
                    # assert anc_adj_v.level == i
                    if anc_adj_v not in C_v:
                        stack_v.append(anc_adj_v)
                        C_v.add(anc_adj_v)
                        size_v += anc_adj_v.N

        if len(stack_v) == 0 and len(edges_v) == 0:
            while (len(stack_u) > 0 or len(edges_u) > 0) and size_u <= size_v:
                if len(edges_u) == 0:
                    search(stack_u, i, adj_list_group_by_levels, edges_u)
                else:
                    vertex_u, adj_u = edges_u.pop()
                    anc_adj_u = ancestor_at_i(LT[adj_u], i)
                    x, y = order(vertex_u, adj_u)
                    P_u.add((x, y))
                    # assert anc_adj_u.level == i
                    if anc_adj_u not in C_u:
                        stack_u.append(anc_adj_u)
                        C_u.add(anc_adj_u)
                        size_u += anc_adj_u.N

        if size_u <= size_v:
            delete_case2(C_u, P_u, LT, adj_list_group_by_levels, i)
        else:
            delete_case2(C_v, P_v, LT, adj_list_group_by_levels, i)

        i -= 1
    return


def search(stack, i, adj_list_group_by_levels, edges):
    # amortized cost O(h)
    if len(stack) == 0:
        return

    cur = stack.pop()
    if cur.val is not None:
        for adj_u in adj_list_group_by_levels[cur.val][i]:
            edges.append((cur.val, adj_u))

    if cur.left is not None and cur.left.bitmap[i] != 0:
        stack.append(cur.left)
    if cur.right is not None and cur.right.bitmap[i] != 0:
        stack.append(cur.right)

    return


def get_level(u, v, adjacency_level):
    for level, adj in adjacency_level[u].items():
        if v in adj:
            return level
    return -1


def ancestor_at_i(node, i):
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


def cal_total_memory_use(LTV, adj_group_by_levels):
    # get all root nodes
    root_node_set = set()
    for _, node in LTV.items():
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
            if node.left is not None:
                q.put(node.left)
            if node.right is not None:
                q.put(node.right)
    space_n += sys.getsizeof(LTV)
    for vertex, adj_list_dict in adj_group_by_levels.items():
        for level, vertice_set in adj_list_dict.items():
            if vertice_set is not None:
                space_e += sys.getsizeof(vertice_set)

    print("total memory size : %d bytes" % (space_n + space_e))
    return space_n, space_e


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