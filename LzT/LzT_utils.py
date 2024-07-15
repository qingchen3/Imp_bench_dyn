from LzT.LzTNode import LzTNode
from utils.tree_utils import order
import math
from utils import graph_utils
from timeit import default_timer as timer

alpha = 3  # LT's paper suggests that alpha > 2


def toRoot(node):
    r = node
    while r.parent is not None:
        r = r.parent
    return r


def query(LzT, u, v):
    if u not in LzT or v not in LzT:
        return False
    return toRoot(LzT[u]) == toRoot(LzT[v])


def tree_edge_level_up(u, v, LzT, adj_t_group_by_levels, level):
    adj_t_group_by_levels[u][level].remove(v)  # adjacency list for non-tree and tree edge
    adj_t_group_by_levels[v][level].remove(u)

    if len(adj_t_group_by_levels[u][level]) == 0:
        LzT[u].tree_bitmap[level] = 0  # update tree-edge bitmap
        update_bitmap(LzT[u], level, type)

    if len(adj_t_group_by_levels[v][level]) == 0:
        LzT[v].tree_bitmap[level] = 0
        update_bitmap(LzT[v], level, type)

    if level + 1 not in adj_t_group_by_levels[u]:
        adj_t_group_by_levels[u][level + 1] = set()

    if level + 1 not in adj_t_group_by_levels[v]:
        adj_t_group_by_levels[v][level + 1] = set()

    adj_t_group_by_levels[u][level + 1].add(v)
    adj_t_group_by_levels[v][level + 1].add(u)

    # update bitmap
    if LzT[u].tree_bitmap[level + 1] == 0:
        LzT[u].tree_bitmap[level + 1] = 1
        update_bitmap(LzT[u], level + 1, 1)

    if LzT[v].tree_bitmap[level + 1] == 0:
        LzT[v].tree_bitmap[level + 1] = 1
        update_bitmap(LzT[v], level + 1, 1)


def nontree_edge_level_up(u, v, LzT, adj_nt_group_by_levels, level):
    adj_nt_group_by_levels[u][level].remove(v)  # adjacency list for non-tree and tree edge
    adj_nt_group_by_levels[v][level].remove(u)

    if len(adj_nt_group_by_levels[u][level]) == 0:
        LzT[u].nontree_bitmap[level] = 0  # update tree-edge bitmap
        update_bitmap(LzT[u], level, 0)

    if len(adj_nt_group_by_levels[v][level]) == 0:
        LzT[v].nontree_bitmap[level] = 0
        update_bitmap(LzT[v], level, 0)

    if level + 1 not in adj_nt_group_by_levels[u]:
        adj_nt_group_by_levels[u][level + 1] = set()

    if level + 1 not in adj_nt_group_by_levels[v]:
        adj_nt_group_by_levels[v][level + 1] = set()

    adj_nt_group_by_levels[u][level + 1].add(v)
    adj_nt_group_by_levels[v][level + 1].add(u)

    # update bitmap
    if LzT[u].nontree_bitmap[level + 1] == 0:
        LzT[u].nontree_bitmap[level + 1] = 1
        update_bitmap(LzT[u], level + 1, 0)

    if LzT[v].nontree_bitmap[level + 1] == 0:
        LzT[v].nontree_bitmap[level + 1] = 1
        update_bitmap(LzT[v], level + 1, 0)


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
    # given a root of a local tree, remove all connecting nodes
    # return a list of local rank trees with ranks in a descending order

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
        if cur.right != Rb:
            if cur.right is not None:  # if there exists no bottom tree
                l.append(cur.right)
        if cur.left is not None and type(cur.left) != 3:
            if cur.left != Rb:
                l.append(cur.left)
            break
        cur = cur.left
    return l, 1


def pair_up(c1, c2):  # c1.rank = c2.rank
    p = LzTNode(None)  # make cur parent of left_child and right_child
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


def construct(l, r):
    # contruct a local tree
    # return l: a list of local rank trees, with ranks in an ascending order
    if len(l) == 0:  # l is empty, generated by the case where r has only one right child that is removed.
        r.left = None
        r.right = None
        r.N = 0
        r.nontree_bitmap.setall(0)
        r.tree_bitmap.setall(0)
        r.type = None
        return r, 0
    elif len(l) == 1:
        r.N = l[0].N
        r.left = None
        r.right = l[0]
        l[0].parent = r
        r.nontree_bitmap = l[0].nontree_bitmap.copy()
        r.tree_bitmap = l[0].tree_bitmap.copy()
        return r, 0
    else:
        if len(l) == 2:
            cur = r
        else:
            cur = LzTNode(None)  # a connecting (path) node
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
                par = LzTNode(None)  # a connecting (path) node
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


def construct_lazy_local_tree(l, r):  # a list of local rank trees, with ranks in an ascending order
    # r is the root
    if len(l) == 0:  # l is empty, generated by the case where r has only one right child that is removed.
        r.left = None
        r.right = None
        r.N = 0
        r.nontree_bitmap.setall(0)
        r.tree_bitmap.setall(0)
        r.type = None
        return r, 0
    elif len(l) == 1:
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
            cur = LzTNode(None)  # a connecting (path) node
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
                par = LzTNode(None)  # a connecting (path) node
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


def insert(u, v, LzT, adj_nt_group_by_levels, adj_t_group_by_levels):

    if u not in LzT or v not in LzT:
        if u not in LzT and v not in LzT:
            # create a lazy local tree with a buffer tree (as left child)

            LzT[u] = LzTNode(u)  # a tree node
            # LzT[u].rank = 0
            LzT[u].N = 1
            adj_t_group_by_levels[u] = dict()
            adj_t_group_by_levels[u][0] = set()
            LzT[u].level = 0

            LzT[v] = LzTNode(v)  # a tree node
            # LzT[v].rank = 0
            LzT[v].N = 1
            adj_t_group_by_levels[v] = dict()
            adj_t_group_by_levels[v][0] = set()
            LzT[v].level = 0

            adj_t_group_by_levels[u][0].add(v)  # update adjacency list
            LzT[u].tree_bitmap[0] = 1
            adj_t_group_by_levels[v][0].add(u)  # update adjacency list
            LzT[v].tree_bitmap[0] = 1

            r = LzTNode(None)  # root of the lazy local tree
            r.N = 0
            r.level = -1

            bfr_r = LzTNode(None)  # root of a buffer tree
            bfr_r.level = 0
            bfr_r.N = 0

            r.left = bfr_r
            bfr_r.parent = r

            attach_to_lazy_local_tree_as_a_child(r, LzT[u])
            attach_to_lazy_local_tree_as_a_child(r, LzT[v])

        elif u not in LzT:
            LzT[u] = LzTNode(u)
            # LzT[u].rank = 0
            LzT[u].N = 1
            adj_t_group_by_levels[u] = dict()
            adj_t_group_by_levels[u][0] = set()
            LzT[u].level = 0

            adj_t_group_by_levels[u][0].add(v)  # update adjacency list for tree edges
            LzT[u].tree_bitmap[0] = 1
            update_bitmap(LzT[u], 0, 1)

            adj_t_group_by_levels[v][0].add(u)  # update adjacency list for tree edges
            LzT[v].tree_bitmap[0] = 1
            update_bitmap(LzT[v], 0, 1)

            # add u into the buffer tree that contains v
            r_v = get_subtree_root(LzT[v], 0)  # r_v is the root of the tree (bottom tree or buffer tree) contains v
            root = ancester_at_level(r_v, -1)  # root of the whole lazy local tree

            bfr_r = root.left
            while type(bfr_r) == 3:
                bfr_r = bfr_r.left

            l, _ = remove_connecting_nodes_and_Rb(root, bfr_r)  # descending
            bfr_r, _ = merge(bfr_r, LzT[u])

            if bfr_r.N > 2 * (int(math.log(graph_utils.n))) ** alpha:  # buffer tree becomes bottom tree
                new_bfr_r = LzTNode(None)  # root of a new buffer tree
                new_bfr_r.level = 0
                new_bfr_r.N = 0
                new_l, _ = merge_lists(l, [bfr_r, new_bfr_r])
                construct_lazy_local_tree(new_l, root)
            else:
                l.append(bfr_r)
                l.reverse()
                construct_lazy_local_tree(l, root)
        else:
            LzT[v] = LzTNode(v)
            # LzT[v].rank = 0
            LzT[v].N = 1
            adj_t_group_by_levels[v] = dict()
            adj_t_group_by_levels[v][0] = set()

            LzT[v].level = 0

            adj_t_group_by_levels[u][0].add(v)  # update adjacency list for tree edges
            LzT[u].tree_bitmap[0] = 1
            update_bitmap(LzT[u], 0, 1)

            adj_t_group_by_levels[v][0].add(u)  # update adjacency list for tree edges
            LzT[v].tree_bitmap[0] = 1
            update_bitmap(LzT[v], 0, 1)

            # add v into the buffer tree that contains u
            r_u = get_subtree_root(LzT[u], 0)  # r_u is the root of the tree (bottom tree or buffer tree) contains u

            root = ancester_at_level(r_u, -1)  # root of the whole lazy local tree

            bfr_r = root.left
            while type(bfr_r) == 3:
                bfr_r = bfr_r.left

            l,_ = remove_connecting_nodes_and_Rb(root, bfr_r)  # descending
            bfr_r, _ = merge(bfr_r, LzT[v])
            if bfr_r.N > 2 * (int(math.log(graph_utils.n))) ** alpha:  # buffer tree becomes bottom tree
                new_bfr_r = LzTNode(None)
                new_bfr_r.level = 0
                new_bfr_r.N = 0
                new_l, _ = merge_lists(l, [bfr_r, new_bfr_r])
                construct_lazy_local_tree(new_l, root)
            else:
                l.append(bfr_r)
                l.reverse()
                construct_lazy_local_tree(l, root)
    else:
        if u in adj_nt_group_by_levels:
            for i in adj_nt_group_by_levels[u].keys():  # filter out redundant non-tree edges
                if v in adj_nt_group_by_levels[u][i]:
                    return

        if u in adj_t_group_by_levels:
            for i in adj_t_group_by_levels[u].keys():  # filter out redundant tree edges
                if v in adj_t_group_by_levels[u][i]:
                    return

        root_u = ancester_at_level(LzT[u], -1)  # root of the whole lazy local tree contains LzT[u]
        root_v = ancester_at_level(LzT[v], -1)  # root of the whole lazy local tree contains LzT[v]
        if root_u != root_v:
            # update tree edge bitmaps
            adj_t_group_by_levels[u][0].add(v)  # update adjacency list
            LzT[u].tree_bitmap[0] = 1
            update_bitmap(LzT[u], 0, 1)
            adj_t_group_by_levels[v][0].add(u)  # update adjacency list
            LzT[v].tree_bitmap[0] = 1
            update_bitmap(LzT[v], 0, 1)

            bfr_r_u = root_u.left
            while type(bfr_r_u) == 3:
                bfr_r_u = bfr_r_u.left
            l_u, _ = remove_connecting_nodes_and_Rb(root_u, bfr_r_u)

            bfr_r_v = root_v.left
            while type(bfr_r_v) == 3 :
                bfr_r_v = bfr_r_v.left
            l_v, _ = remove_connecting_nodes_and_Rb(root_v, bfr_r_v)

            new_l, _ = merge_lists(l_u, l_v)  # the list of bottom trees, ascending order

            # merge two buffer trees
            bfr_r_v_N = bfr_r_v.N
            bfr_r_u_N = bfr_r_u.N
            bfr_r, _ = merge(bfr_r_v, bfr_r_u)
            if bfr_r.N > 2 * (int(math.log(graph_utils.n))) ** alpha: # buffer tree becomes bottom tree
                new_bfr_r = LzTNode(None)
                new_bfr_r.level = 0
                new_bfr_r.N = 0
                new_l.reverse()
                new_l, _ = merge_lists(new_l, [bfr_r, new_bfr_r])
            else:
                new_l.insert(0, bfr_r)

            construct_lazy_local_tree(new_l, root_u)

        else:  # (u, v) is a non-tree edge

            # print(u, v , "is a non-tree edge")
            if u not in adj_nt_group_by_levels:
                adj_nt_group_by_levels[u] = dict()
                adj_nt_group_by_levels[u][0] = set()

            if v not in adj_nt_group_by_levels:
                adj_nt_group_by_levels[v] = dict()
                adj_nt_group_by_levels[v][0] = set()

            for i in adj_nt_group_by_levels[u].keys():  # fiLzTer out redundant edges
                if v in adj_nt_group_by_levels[u][i]:
                    return

            if v not in adj_nt_group_by_levels[u][0]:
                adj_nt_group_by_levels[u][0].add(v)  # update adjacency list

            if u not in adj_nt_group_by_levels[v][0]:
                adj_nt_group_by_levels[v][0].add(u)  # update adjacency list

            LzT[u].nontree_bitmap[0] = 1
            update_bitmap(LzT[u], 0, 0)
            LzT[v].nontree_bitmap[0] = 1
            update_bitmap(LzT[v], 0, 0)

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


def remove(r, b):  # remove b from the lazy local tree rooted at r
    x_b = get_subtree_root(b, b.level)  # x_b is the root of the tree (bottom tree or buffer tree) contains v
    buffer_tree_size = 2 * (int(math.log(graph_utils.n))) ** alpha
    OPS = 0
    if x_b.N <= buffer_tree_size:  # b is contained in the buffer tree rooted at x_b
        l, ops_num = remove_connecting_nodes_and_Rb(r, x_b)  # descending
        OPS += ops_num
        l_bfr, ops_num = remove_from_local_tree(x_b, b)  # a buffer tree is also a local tree, l_bfr ascending
        OPS += ops_num
        if len(l_bfr) == 0:  # the removed b is the root of the buffer tree
            # refresh the root of the buffer tree
            new_bfr_r = LzTNode(None)
            new_bfr_r.level = x_b.level
            new_bfr_r.N = 0
            l.append(new_bfr_r)
        else:
            x_b, ops_num = construct(l_bfr, x_b)
            OPS += ops_num
            l.append(x_b)
        l.reverse()
        OPS += 1 # count increase due to the reverse
        r, ops_num = construct_lazy_local_tree(l, r)
        OPS += ops_num
    else:  # b is contained in the bottom tree
        Rb = x_b
        while type(Rb.parent) == 2:
            Rb = Rb.parent
        l, ops_num = remove_connecting_nodes_and_Rb(r, Rb)  # descending
        OPS += ops_num
        bfr_r = l.pop()  # root of the buffer tree
        x_b_size = x_b.N  # bottom tree that contains b

        if Rb == x_b:
            if x_b == b:
                l.append(bfr_r)
                l.reverse()
                OPS += 1
                r, ops_num = construct_lazy_local_tree(l, r)
                OPS += ops_num
            else:
                l_x_b, ops_num = remove_from_local_tree(x_b, b)  # asceding
                OPS += ops_num
                if x_b_size - b.N <= buffer_tree_size:  # bottom tree becomes a buffer tree due to the deletion of b
                    l_x_b.reverse()  # descending
                    OPS += 1
                    if bfr_r.N + (x_b_size - b.N) > buffer_tree_size:  # l_x_b + bfr makes up a bottom tree
                        l_bfr_r, ops_num = remove_connecting_nodes(bfr_r)  # descending
                        OPS += ops_num
                        new_l_btm, ops_num = merge_lists(l_x_b, l_bfr_r)
                        OPS += ops_num
                        x_b, ops_num = construct(new_l_btm, x_b)
                        OPS += ops_num

                        new_bfr_r = LzTNode(None)
                        new_bfr_r.level = bfr_r.level
                        new_bfr_r.N = 0

                        new_l, ops_num = merge_lists(l, [x_b])
                        OPS += ops_num
                        new_l.insert(0, new_bfr_r)
                        OPS += 1
                        r, ops_num = construct_lazy_local_tree(new_l, r)
                        OPS += ops_num
                    else:
                        l_bfr_r, ops_num = remove_connecting_nodes(bfr_r)  # descending
                        OPS += ops_num
                        new_l_bfr, ops_num = merge_lists(l_x_b, l_bfr_r)
                        OPS += ops_num
                        bfr_r, ops_num = construct(new_l_bfr, bfr_r)
                        OPS += ops_num
                        l.append(bfr_r)
                        l.reverse()
                        OPS += 1
                        r, ops_num = construct_lazy_local_tree(l, r)
                        OPS += ops_num
                else:  # x_b remains to be a bottom tree
                    x_b, ops_num = construct(l_x_b, x_b)
                    OPS += ops_num
                    new_l, ops_num = merge_lists(l, [x_b])
                    OPS += ops_num
                    new_l.insert(0, bfr_r)
                    OPS += 1
                    r, ops_num = construct_lazy_local_tree(new_l, r)
                    OPS += ops_num
        else:
            l_Rb, ops_num = remove_from_local_rank_tree(Rb, x_b)  # ascending
            OPS += ops_num
            if x_b == b:
                l_Rb.reverse()  # descending
                OPS += 1
                new_l, ops_num = merge_lists(l, l_Rb)
                OPS += ops_num
                new_l.insert(0, bfr_r)
                OPS += 1
                r, ops_num = construct_lazy_local_tree(new_l, r)
                OPS += ops_num
            else:
                l_x_b, ops_num = remove_from_local_tree(x_b, b)  # ascending order
                OPS += ops_num
                if x_b_size - b.N <= buffer_tree_size:  # bottom tree becomes a buffer tree due to the deletion of b
                    l_x_b.reverse()  # descending order
                    OPS += 1
                    if bfr_r.N + (x_b_size - b.N) > buffer_tree_size: # l_x_b + bfr makes up a bottom tree
                        l_bfr_r, ops_num = remove_connecting_nodes(bfr_r)  # descending
                        OPS += ops_num
                        new_l_btm, ops_num = merge_lists(l_x_b, l_bfr_r)  # descending order for l_x_b and l_bfr_r
                        OPS += ops_num
                        x_b, ops_num = construct(new_l_btm, x_b)
                        OPS += ops_num
                        l_Rb.reverse()
                        OPS += 1
                        temp_l, ops_nump = merge_lists(l_Rb, [x_b])  # descending orders for [x_b, new_bfr_r]
                        OPS += 1
                        temp_l.reverse()
                        OPS += 1
                        new_l, ops_num = merge_lists(l, temp_l)
                        OPS += ops_num

                        new_bfr_r = LzTNode(None)
                        new_bfr_r.level = bfr_r.level
                        new_bfr_r.N = 0
                        new_l.insert(0, new_bfr_r)
                        OPS += 1
                        r, ops_num = construct_lazy_local_tree(new_l, r)
                        OPS += ops_num
                    else:
                        l_bfr_r, ops_num = remove_connecting_nodes(bfr_r)  # descending
                        OPS += ops_num
                        new_l_bfr, ops_num = merge_lists(l_x_b, l_bfr_r)
                        OPS += ops_num
                        bfr_r, ops_num = construct(new_l_bfr, bfr_r)
                        OPS += ops_num
                        l_Rb.reverse()
                        OPS += 1
                        new_l, ops_num = merge_lists(l, l_Rb)
                        OPS += ops_num
                        new_l.insert(0, bfr_r)
                        OPS += 1
                        r, ops_num = construct_lazy_local_tree(new_l, r)
                        OPS += ops_num
                else:
                    x_b, ops_num = construct(l_x_b, x_b)
                    OPS += ops_num
                    l_Rb.reverse()
                    OPS += 1
                    temp_l, ops_num = merge_lists(l_Rb, [x_b])
                    OPS += ops_num
                    temp_l.reverse()
                    OPS += 1
                    new_l, ops_num = merge_lists(l, temp_l)
                    OPS += 1
                    new_l.insert(0, bfr_r)
                    OPS += 1
                    r, ops_num = construct_lazy_local_tree(new_l, r)
                    OPS += ops_num
    return r, OPS


def remove_from_local_tree(r, b):  # remove b from the local tree rooted at r
    # generating a list with nodes' ranks ascending order

    # type = 1: tree node in local tree
    # type = 2: node of a local rank tree
    # type = 3: connecting node
    OPS = 0
    if type(r) == 2:  # remove b from a local rank tree
        l, ops_num = remove_from_local_rank_tree(r, b)  # ranks in an ascending order
        OPS += ops_num
    else:  # remove b from a local tree
        Rb = b
        while type(Rb.parent) == 2:  #
            Rb = Rb.parent
        l, ops_num = remove_connecting_nodes_and_Rb(r, Rb)  # remove connecting nodes and remove Rb from l
        OPS += ops_num
        if Rb != b:
            lb, ops_num = remove_from_local_rank_tree(Rb, b)
            OPS += ops_num
            lb.reverse()
            OPS += 1
            l, ops_num = merge_lists(l, lb)
            OPS += ops_num
        else:
            l.reverse()
            OPS += 1
    return l, OPS


def attach_to_lazy_local_tree_as_a_child(r, b):
    OPS = 0
    l_r, ops_num = remove_connecting_nodes(r)  # descending
    OPS += ops_num
    bfr_r = l_r.pop()

    buffer_tree_size = 2 * (int(math.log(graph_utils.n))) ** alpha

    if b.N <= buffer_tree_size:

        l_bfr_r, ops_num = remove_connecting_nodes(bfr_r)  # descending
        OPS += ops_num
        l_bfr, ops_num = merge_lists(l_bfr_r, [b])
        OPS += ops_num

        bfr_r, ops_num = construct(l_bfr, bfr_r)
        OPS += ops_num
        if bfr_r.N <= buffer_tree_size:  # the merged buffer tree is still a buffer tree
            l_r.append(bfr_r)
            l_r.reverse()
            OPS += 1
        else:
            l_r, ops_num = merge_lists(l_r, [bfr_r])
            OPS += ops_num
            new_bfr_r = LzTNode(None)
            new_bfr_r.level = bfr_r.level
            new_bfr_r.N = 0
            l_r.insert(0, new_bfr_r)
            OPS += 1
    else:
        l_r, ops_num = merge_lists(l_r, [b])
        OPS += ops_num
        l_r.insert(0, bfr_r)
        OPS += 1
    r, ops_num = construct_lazy_local_tree(l_r, r)
    OPS += ops_num
    return r, OPS


def merge_into_lazy_local_trees(r, b):  # merge b with the children of root r

    # assert r is not None
    OPS = 0

    l_r, ops_num = remove_connecting_nodes(r)  # descending
    bfr_r = l_r.pop()
    OPS += ops_num

    l_b, ops = remove_connecting_nodes(b)  # descending
    OPS += ops_num
    bfr_r_b = l_b.pop()  # buffer tree of c will be merged into bfr_r_Tv

    l_bfr_r, ops_num = remove_connecting_nodes(bfr_r)  # descending
    OPS += ops_num
    l_bfr_r_b, ops_num = remove_connecting_nodes(bfr_r_b)  # descending
    OPS += ops_num
    buffer_tree_size = 2 * (int(math.log(graph_utils.n))) ** alpha

    if bfr_r.N + bfr_r_b.N == 0:  # buffer trees for Tv and c are both empty
        l, ops_num = merge_lists(l_r, l_b)
        OPS += ops_num
        l.insert(0, bfr_r)
        OPS += 1
        r, ops_num = construct_lazy_local_tree(l, r)
        OPS += ops_num
    elif bfr_r.N + bfr_r_b.N <= buffer_tree_size:  # the merged buffer tree is still a buffer tree
        l_bfr, ops_num = merge_lists(l_bfr_r, l_bfr_r_b)
        OPS += ops_num
        bfr_r, ops_num = construct(l_bfr, bfr_r)
        OPS += ops_num
        l, ops_num = merge_lists(l_r, l_b)
        OPS += ops_num
        l.insert(0, bfr_r)
        OPS += 1
        r, ops_num = construct_lazy_local_tree(l, r)
        OPS += ops_num
    else:  # the merged buffer tree becomes a bottom tree
        l_bfr, ops_num = merge_lists(l_bfr_r, l_bfr_r_b)
        OPS += ops_num
        bfr_r, ops_num = construct(l_bfr, bfr_r)  # bfr_r_Tv becomes the root of a bottom tree
        OPS += ops_num

        new_bfr_r = LzTNode(None)
        new_bfr_r.level = bfr_r.level
        new_bfr_r.N = 0
        if len(l_r) < len(l_b):
            temp_l, ops_num = merge_lists(l_r, [bfr_r])
            OPS += ops_num
            temp_l.reverse()
            OPS += 1
            temp_l.append(new_bfr_r)
            l, ops_num = merge_lists(l_b, temp_l)
            OPS += ops_num
        else:
            temp_l, ops_num = merge_lists(l_b, [bfr_r])
            OPS += ops_num
            temp_l.reverse()
            OPS += 1
            temp_l.append(new_bfr_r)
            l, ops_num = merge_lists(l_r, temp_l)
            OPS += ops_num
        r, ops_num = construct_lazy_local_tree(l, r)
        OPS += ops_num
    return r, OPS


def merge(a, b):  # return a new tree with a as the root
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


def create_Tv(C, P, LzT, adj_t_group_by_levels, level):
    OPS = 0

    g = None

    Tv = LzTNode(None)
    Tv.level = level
    Tv.N = 0

    bfr_r = LzTNode(None)  # root of a buffer tree
    bfr_r.N = 0
    bfr_r.level = level + 1

    Tv.left = bfr_r
    bfr_r.parent = Tv

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

        TV, ops_num= merge_into_lazy_local_trees(Tv, c)
        OPS += ops_num

    if len(C) > 1:
        for u, v in P:  # promoting tree edges
            tree_edge_level_up(u, v, LzT, adj_t_group_by_levels, level)

    return g, par, Tv, OPS


def delete(u, v, LzT, adj_nt_group_by_levels, adj_t_group_by_levels):

    OPS = 0
    if edge_type(u, v, adj_t_group_by_levels) == 0:  # (u, v) is a non-tree edge
        i = get_level(u, v, adj_nt_group_by_levels)
        if i == -1:
            return 0
        adj_nt_group_by_levels[u][i].remove(v)
        adj_nt_group_by_levels[v][i].remove(u)

        if len(adj_nt_group_by_levels[u][i]) == 0:  # update bitmap
            LzT[u].nontree_bitmap[i] = 0
            update_bitmap(LzT[u], i, 0)

        if len(adj_nt_group_by_levels[v][i]) == 0:  # update bitmap
            LzT[v].nontree_bitmap[i] = 0
            update_bitmap(LzT[v], i, 0)

        return 0

    # remove edges (u, v)
    i = get_level(u, v, adj_t_group_by_levels)
    if i == -1:
        return 0
    # print("deleting tree edge %d-%d:" %(u, v), ", level:", i)

    adj_t_group_by_levels[u][i].remove(v)
    adj_t_group_by_levels[v][i].remove(u)

    if len(adj_t_group_by_levels[u][i]) == 0:  # update bitmap for tree edge
        LzT[u].tree_bitmap[i] = 0
        update_bitmap(LzT[u], i, 1)

    if len(adj_t_group_by_levels[v][i]) == 0:  # update bitmap for tree edge
        LzT[v].tree_bitmap[i] = 0
        update_bitmap(LzT[v], i, 1)

    my_start = 0
    count = 0
    while i >= 0:
        #print("current level:", i)
        anc_u = ancester_at_level(LzT[u], i)
        stack_u = []  # DFS_u:  DFS starts at anc_u
        edges_u = []  # candidate edges to be visited by DFS_u
        stack_u.append(anc_u)

        C_u = set()
        C_u.add(anc_u)
        P_u = set()
        size_u = anc_u.N

        anc_v = ancester_at_level(LzT[v], i)
        stack_v = []  # DFS_v: DFS starts at anc_v
        edges_v = []  # candidate edges to be visited by DFS_v
        stack_v.append(anc_v)
        C_v = set()
        C_v.add(anc_v)
        P_v = set()
        size_v = anc_v.N
        if u == 1 and v == 18:
            my_start = timer()
        while (len(stack_u) > 0 or len(edges_u) > 0) and (len(stack_v) > 0 or len(edges_v) > 0):
            if len(edges_u) == 0:
                search(stack_u, i, adj_t_group_by_levels, edges_u)
            else:
                vertex_u, adj_u = edges_u.pop()
                anc_adj_u = ancester_at_level(LzT[adj_u], i)
                count += 1
                # assert anc_adj_u.level == i
                # assert anc_adj_u not in C_v

                x, y = order(vertex_u, adj_u)
                if (x, y) not in P_u:
                    P_u.add((x, y))
                if anc_adj_u not in C_u:
                    stack_u.append(anc_adj_u)
                    C_u.add(anc_adj_u)
                    size_u += anc_adj_u.N

            if len(edges_v) == 0:
                search(stack_v, i, adj_t_group_by_levels, edges_v)
            else:
                vertex_v, adj_v = edges_v.pop()
                anc_adj_v = ancester_at_level(LzT[adj_v], i)
                count += 1
                # assert anc_adj_v.level == i
                # assert anc_adj_v not in C_u

                x, y = order(vertex_v, adj_v)
                if (x, y) not in P_v:
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
                    anc_adj_v = ancester_at_level(LzT[adj_v], i)
                    count += 1
                    x, y = order(vertex_v, adj_v)
                    if (x, y) not in P_v:
                        P_v.add((x, y))
                    # assert anc_adj_v.level == i
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
                    anc_adj_u = ancester_at_level(LzT[adj_u], i)
                    count += 1
                    x, y = order(vertex_u, adj_u)
                    if (x, y) not in P_u:
                        P_u.add((x, y))
                    # assert anc_adj_u.level == i
                    if anc_adj_u not in C_u:
                        stack_u.append(anc_adj_u)
                        C_u.add(anc_adj_u)
                        size_u += anc_adj_u.N

        if size_u <= size_v:  # nodes in C_u are made Tv
            C = C_u
            P = P_u
        else:  # nodes in C_v are made Tv
            C = C_v
            P = P_v
        g, par, Tv, ops_num = create_Tv(C, P, LzT, adj_t_group_by_levels, i)

        tx, ty, nontree_edges = search_replacemet_edge(Tv, adj_nt_group_by_levels, LzT, i)
        if tx is not None and ty is not None:  # (x, y) is a replacement edge, case 1 for deletion

            adj_nt_group_by_levels[tx][i].remove(ty)
            adj_nt_group_by_levels[ty][i].remove(tx)

            if len(adj_nt_group_by_levels[tx][i]) == 0:  # update bitmap
                LzT[tx].nontree_bitmap[i] = 0
                update_bitmap(LzT[tx], i, 0)

            if len(adj_nt_group_by_levels[ty][i]) == 0:  # update bitmap
                LzT[ty].nontree_bitmap[i] = 0
                update_bitmap(LzT[ty], i, 0)

            adj_t_group_by_levels[tx][i].add(ty)
            LzT[tx].tree_bitmap[i] = 1
            update_bitmap(LzT[tx], i, 1)

            adj_t_group_by_levels[ty][i].add(tx)
            LzT[ty].tree_bitmap[i] = 1
            update_bitmap(LzT[ty], i, 1)

            par, ops_num = attach_to_lazy_local_tree_as_a_child(par, Tv)
            OPS += ops_num
            if g is not None:
                g, ops_num = attach_to_lazy_local_tree_as_a_child(g, par)
                OPS += ops_num
            if nontree_edges is not None:
                for nu, nv in nontree_edges:
                    nontree_edge_level_up(nu, nv, LzT, adj_nt_group_by_levels, i)

            return count
        else:  # case 2 for deletion

            # root of a lazy local tree
            p_prime = LzTNode(None)
            p_prime.N = 0
            p_prime.level = i - 1

            # buffer tree
            bfr_r = LzTNode(None)
            bfr_r.level = i
            bfr_r.N = 0

            p_prime.left = bfr_r
            bfr_r.parent = p_prime

            buffer_tree_size = 2 * (int(math.log(graph_utils.n))) ** alpha
            # assert Tv.level == i

            if Tv.N <= buffer_tree_size:
                bfr_r.right = Tv
                Tv.parent = bfr_r
                bfr_r.N = Tv.N
                bfr_r.nontree_bitmap = Tv.nontree_bitmap
                bfr_r.tree_bitmap = Tv.tree_bitmap
            else:
                p_prime.right = Tv
                Tv.parent = p_prime

            p_prime.nontree_bitmap = Tv.nontree_bitmap
            p_prime.tree_bitmap = Tv.tree_bitmap
            p_prime.N = Tv.N
            # p_prime.type = 1

            if g is not None:
                g, ops_num = attach_to_lazy_local_tree_as_a_child(g, par)
                OPS += ops_num
                g, ops_num = attach_to_lazy_local_tree_as_a_child(g, p_prime)
                OPS += ops_num

            if nontree_edges is not None:
                for nu, nv in nontree_edges:
                    nontree_edge_level_up(nu, nv, LzT, adj_nt_group_by_levels, i)
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


def search_replacemet_edge(x, adj_nt_group_by_levels, LzT, i):
    stack = [x]
    nontree_edges = set()
    while len(stack) != 0:
        cur = stack.pop()
        if cur.val is not None:
            if cur.val not in adj_nt_group_by_levels or i not in adj_nt_group_by_levels[cur.val] or \
                    len(adj_nt_group_by_levels[cur.val][i]) == 0:
                continue
            for adj_nt_x in adj_nt_group_by_levels[cur.val][i]:
                # assert adj_nt_x in adj_nt_group_by_levels[cur.val][i]
                # assert cur.val in adj_nt_group_by_levels[adj_nt_x][i]
                if ancester_at_level(LzT[adj_nt_x], i) != ancester_at_level(LzT[cur.val], i):
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


def ancester_at_level(node, i):  # node.level > i. Levels decrease during the traversal to the root
    anc = node

    while anc is not None:
        if anc.level == i:  # tree node
            return anc
        anc = anc.parent
    return None


def get_level(u, v, adjacency_level):
    if u in adjacency_level:
        for level, adj in adjacency_level[u].items():
            if adj is not None and v in adj:
                return level
    return -1


def get_subtree_root(node, i):
    # returns the root of the tree (bottom tree or buffer tree) that contains node at level 1
    # node.level >= i
    # assert i != -1
    # assert node.level >= i
    if node.level > i:
        cur = ancester_at_level(node, i)
    else:
        cur = node
    tree_root = cur
    while cur.level == i or cur.level is None:
        if cur.level is not None:
            if cur.level < i:
                break
            if cur.level == i:  # tree node
                tree_root = cur
        cur = cur.parent
    return tree_root  # node itself is the root of a bottom tree or a buffer tree