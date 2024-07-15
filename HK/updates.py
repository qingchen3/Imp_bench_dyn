"""
This file is a python implementation of Henzinger-King's algorithm
Monika Rauch Henzinger and Valerie King. 1999. Randomized Fully Dynamic Graph
Algorithms with Polylogarithmic Time per Operation

To synchronize the usage of levels, spanning trees with lower levels includes those of higher level.
Level of newly inserted edges is max_level.
"""

import math
from HK.HK_Node import HK_Node
from utils.tree_utils import coding, smaller, sort
from utils.graph_utils import order
import utils.graph_utils as graph_utils
import sys, random, queue
from HK.NTE_Node import NTE_Node
from HK.NTE_utils import cal_nte_tree_node_size

global rebuild_bound
global added_edges
I = 0


def initialization():
    global rebuild_bound
    global added_edges
    max_level = graph_utils.max_level
    rebuild_bound = [0 * (max_level + 1)]
    added_edges = [0 * (max_level + 1)]
    bound = 5000
    for i in range(max_level, 0, -1):
        rebuild_bound[i] = bound
        if bound < sys.maxsize // 2:
            bound = bound * 2
    return


def sort1(tree_edge_pointer):
    maximum = 0
    minimum = 0
    codes = []
    for pointer in tree_edge_pointer:
        codes.append(coding(pointer))
    for i in range(1, len(tree_edge_pointer)):
        if smaller(codes[maximum], codes[i]):
            maximum = i
        if smaller(codes[i], codes[minimum]):
            minimum = i

    items = []
    for i in range(0, len(tree_edge_pointer)):
        if i == maximum or i == minimum:
            continue
        items.append(i)
    if len(items) == 1:
        return tree_edge_pointer[minimum], tree_edge_pointer[items[0]], tree_edge_pointer[maximum]
    elif smaller(codes[items[0]], codes[items[-1]]):
        return tree_edge_pointer[minimum], tree_edge_pointer[items[-1]], tree_edge_pointer[maximum]
    else:
        return tree_edge_pointer[minimum], tree_edge_pointer[items[0], tree_edge_pointer[maximum]]


def rotateRight(x, root):
    p = x.parent
    g = None
    if p.parent is not None:
        g = p.parent
        if g.left == p:
            g.left = x
        else:
            g.right = x
        x.parent = g
    else:
        x.parent = None

    p.weight = p.weight - x.weight

    # cut right branch of x
    x_right = x.right
    x_right_weight = 0
    if x_right is not None:
        x_right_weight = x_right.weight
    x.weight -= x_right_weight

    # attach right branch of x to p
    if x_right is not None:
        x_right.parent = p

    p.left = x_right
    p.weight += x_right_weight

    # attach p as the right branch of x
    x.right = p
    p.parent = x
    x.weight += p.weight

    if g is not None:
        return root
    else:
        return x


def rotateLeft(x, root):
    p = x.parent
    g = None
    if p.parent is not None:
        g = p.parent
        if g.left == p:
            g.left = x
        else:
            g.right = x
        x.parent = g
    else:
        x.parent = None

    p.weight = p.weight - x.weight

    # cut left branch of x
    x_left = x.left
    x_left_weight = 0
    if x_left is not None:
        x_left_weight = x_left.weight

    x.weight -= x_left_weight

    # attach left branch of x to p
    if x_left is not None:
        x_left.parent = p

    p.right = x_left
    p.weight += x_left_weight

    # attach p as the left branch of x
    x.left = p
    p.parent = x
    x.weight += p.weight

    if g is not None:
        return root
    else:
        return x


def rotate_to_root(x, root):
    c = 0
    while x != root:
        p = x.parent
        if x == p.left:
            root = rotateRight(x, root)
        else:
            root = rotateLeft(x, root)
        c += 1
    return x, c


def head_of_etr_tree(root):
    pointer = root
    while pointer.left is not None:
        pointer = pointer.left
    return pointer


def tail_of_etr_tree(root):
    pointer = root
    while pointer.right is not None:
        pointer = pointer.right
    return pointer


def split_before(root, current_node):
    node = HK_Node(-1, sys.maxsize)
    if current_node.left is None:
        current_node.left = node
        node.parent = current_node
    else:
        pointer = current_node.left
        while pointer.right is not None:
            pointer = pointer.right
        pointer.right = node
        node.parent = pointer

    root, c = rotate_to_root(node, root)
    left_branch = root.left
    left_branch.parent = None

    right_branch = root.right
    right_branch.parent = None

    return left_branch, right_branch


def split_after(root, current_node):
    node = HK_Node(-1, sys.maxsize)
    if current_node.right is None:
        current_node.right = node
        node.parent = current_node
    else:
        pointer = current_node.right
        while pointer.left is not None:
            pointer = pointer.left
        pointer.left = node
        node.parent = pointer

    root, c = rotate_to_root(node, root)

    left_branch = root.left
    left_branch.parent = None

    right_branch = root.right
    right_branch.parent = None

    return left_branch, right_branch


def predecessor(current_node):
    if current_node.left is not None:
        pred = current_node.left
        while pred.right is not None:
            pred = pred.right
        return pred
    else:
        if current_node.parent is None:
            return None
        else:
            parent_pointer = current_node.parent
            current_pointer = current_node
            while parent_pointer is not None:
                if current_pointer == parent_pointer.right:
                    return parent_pointer
                current_pointer = parent_pointer
                parent_pointer = parent_pointer.parent
            return None


def successor(current_node):
    if current_node.right is not None:
        succ = current_node.right
        while succ.left is not None:
            succ = succ.left
        return succ
    else:
        if current_node.parent is None:
            return None
        else:
            parent_pointer = current_node.parent
            current_pointer = current_node
            while parent_pointer is not None:
                if current_pointer == parent_pointer.left:
                    return parent_pointer
                current_pointer = parent_pointer
                parent_pointer = parent_pointer.parent
        return None


def rotate_to_leaf(current_node, root):
    while current_node.left is not None or current_node.right is not None:
        if current_node.left is not None and current_node.right is not None:
            if current_node.left.priority > current_node.right.priority:
                root = rotateRight(current_node.left, root)
            else:
                root = rotateLeft(current_node.right, root)
        elif current_node.left is not None:
            root = rotateRight(current_node.left, root)
        else:
            root = rotateLeft(current_node.right, root)

    return current_node, root


def merge(r1, r2):
    root = HK_Node(-1, sys.maxsize)
    root.weight = r1.weight + r2.weight
    root.left = r1
    root.right = r2
    r1.parent = root
    r2.parent = root

    current_node = root

    # rotate root to leaf
    current_node, root = rotate_to_leaf(current_node, root)

    p = current_node.parent
    if p.left == current_node:
        p.left = None
    else:
        p.right = None

    return root


def merge_update_pointer(r1, rightmost_r1, r2, leftmost_r2, tree_edges_pointer):
    if rightmost_r1 is None:
        rightmost_r1 = r1
        while rightmost_r1.right is not None:
            rightmost_r1 = rightmost_r1.right

    if leftmost_r2 is None:
        leftmost_r2 = r2
        while leftmost_r2.left is not None:
            leftmost_r2 = leftmost_r2.left

    if rightmost_r1.active:
        succ_leftmost_r2 = successor(leftmost_r2)
        if succ_leftmost_r2 is None:
            del r2
            return r1
        (u, v) = order(succ_leftmost_r2.val, leftmost_r2.val)
        tree_edges_pointer[(u, v)].remove(leftmost_r2)
        leftmost_r2, r2 = rotate_to_leaf(leftmost_r2, r2)
        p = leftmost_r2.parent
        if leftmost_r2 == p.left:
            p.left = None
        else:
            p.right = None
        del leftmost_r2
        if rightmost_r1 not in tree_edges_pointer[(u, v)]:
            tree_edges_pointer[(u, v)].append(rightmost_r1)
    else:
        pred_rightmost_r1 = predecessor(rightmost_r1)
        if pred_rightmost_r1 is None:
            del r1
            return r2

        (u, v) = order(pred_rightmost_r1.val, rightmost_r1.val)

        tree_edges_pointer[(u, v)].remove(rightmost_r1)
        rightmost_r1, r1 = rotate_to_leaf(rightmost_r1, r1)
        p = rightmost_r1.parent
        if rightmost_r1 == p.left:
            p.left = None
        else:
            p.right = None
        del rightmost_r1
        if leftmost_r2 not in tree_edges_pointer[(u, v)]:
            tree_edges_pointer[(u, v)].append(leftmost_r2)

    root = merge(r1, r2)
    return root


def reroot(root, u, active_occurrence_dict, tree_edges_pointers):
    head = head_of_etr_tree(root)
    if head.val == u:
        return root

    pred_u = predecessor(active_occurrence_dict[u])
    succ_u = successor(active_occurrence_dict[u])

    root_left_branch, root_right_branch = split_before(root, active_occurrence_dict[u])
    max_priority = sys.maxsize
    node = HK_Node(u, random.randint(1, max_priority))

    # attach node in the end of ETR-tree
    tail_of_left_branch = tail_of_etr_tree(root_left_branch)
    tail_of_left_branch.right = node
    node.parent = tail_of_left_branch

    # rotate node to the place,which does not violate the priority value
    p = tail_of_left_branch
    while p is not None and node.priority > p.priority:
        if node == p.right:
            root_left_branch = rotateLeft(node, root_left_branch)
        else:
            root_left_branch = rotateRight(node, root_left_branch)
        p = node.parent

    # update tree-edge pointers
    (a, b) = order(pred_u.val, u)
    if pred_u.val != succ_u.val:
        tree_edges_pointers[(a, b)].remove(active_occurrence_dict[u])
    tree_edges_pointers[(a, b)].append(node)

    root = merge_update_pointer(root_right_branch, None, root_left_branch, None, tree_edges_pointers)

    return root


def get_level(u, v, adj_groupby_level):
    for level, adj in adj_groupby_level.items():
        if (u, v) in adj:
            return level
    return -1


def insert_edge(u, v, act_occ_dict, tree_edges_pointers, non_tree_edges,
                tree_edges_group_by_levels, nontree_edges_group_by_levels):
    # initialize HKNode for HK if not exists
    global I
    pre_I = I
    max_level = 2 * graph_utils.max_level
    if not connected(u, v, 0, act_occ_dict):
        I += 1
        tree_edges_group_by_levels[0].add((u, v))
        insert_tree_edge(u, v, act_occ_dict, 0, tree_edges_pointers)
    else:  # a and b are connected
        if (u, v) not in non_tree_edges:
            I += 1
            nontree_edges_group_by_levels[0].add((u, v))
            insert_nontree_edge(u, v, act_occ_dict, 0, non_tree_edges, nontree_edges_group_by_levels)
    if pre_I != I:  # rebuild
        rebuild_level = leading_bit(pre_I ^ I)
        if rebuild_level > len(tree_edges_pointers) - 1:
            print("rebuild level is %d" % rebuild_level)
        #print("rebuild at level %d" %rebuild_level)
        rebuild1(rebuild_level, act_occ_dict, tree_edges_pointers, non_tree_edges, tree_edges_group_by_levels,
                 nontree_edges_group_by_levels)

    return


def insert_tree_edge(u, v, act_occ_dict, i, tree_edges_pointers):
    # for update tree_edges and non_tree_edges, rename (a, b) to (x, y) for one-time use
    max_priority = sys.maxsize
    if u not in act_occ_dict[i]:
        node = HK_Node(u, random.randint(1, max_priority))
        node.active = True
        act_occ_dict[i][u] = node

    if v not in act_occ_dict[i]:
        node = HK_Node(v, random.randint(1, max_priority))
        node.active = True
        act_occ_dict[i][v] = node

    root_u = find_root(act_occ_dict[i][u])
    root_v = find_root(act_occ_dict[i][v])

    root_u = reroot(root_u, u, act_occ_dict[i], tree_edges_pointers[i])
    root_v = reroot(root_v, v, act_occ_dict[i], tree_edges_pointers[i])

    tail_of_root_u = tail_of_etr_tree(root_u)
    head_of_root_v = head_of_etr_tree(root_v)
    max_priority = sys.maxsize
    # add a new occurrence of root_a to the end new ETR
    node = HK_Node(u, random.randint(1, max_priority))

    # update tree-edge-pointers
    tree_edges_pointers[i][(u, v)].append(tail_of_root_u)
    tree_edges_pointers[i][(u, v)].append(head_of_root_v)
    if root_v.left is not None or root_v.right is not None:
        tail_of_root_b = tail_of_etr_tree(root_v)
        tree_edges_pointers[i][(u, v)].append(tail_of_root_b)

    tree_edges_pointers[i][(u, v)].append(node)

    # attach node in the end of ETR-tree
    tail_of_root_b = tail_of_etr_tree(root_v)
    tail_of_root_b.right = node
    node.parent = tail_of_root_b
    ##
    # rotate node to the place,which does not violate the priority value
    p = tail_of_root_b
    while p is not None and node.priority > p.priority:
        if node == p.right:
            root_v = rotateLeft(node, root_v)
        else:
            root_v = rotateRight(node, root_v)
        p = node.parent

    root = merge(root_u, root_v)

    return root


def insert_nontree_edge(u, v, act_occ_dict, i, non_tree_edges, nontree_edges_group_by_levels):
    u_act_occ = act_occ_dict[i][u]
    v_act_occ = act_occ_dict[i][v]

    # add non-tree edge into the tree
    node = NTE_Node(u_act_occ, random.randint(1, sys.maxsize))
    non_tree_edges[(v, u)] = node
    add_nte_node(v_act_occ, node)

    node = NTE_Node(v_act_occ, random.randint(1, sys.maxsize))
    non_tree_edges[(u, v)] = node
    add_nte_node(u_act_occ, node)

    if u_act_occ.nte is None:
        u_act_occ.nte = set()
    u_act_occ.nte.add(v)
    pointer = u_act_occ
    while pointer is not None:
        pointer.weight += 1
        pointer = pointer.parent

    if v_act_occ.nte is None:
        v_act_occ.nte = set()
    v_act_occ.nte.add(u)
    pointer = v_act_occ
    while pointer is not None:
        pointer.weight += 1
        pointer = pointer.parent

    return


def rebuild1(i, act_occ_dict, tree_edges_pointers, non_tree_edges, tree_edges_group_by_levels,
             nontree_edges_group_by_levels):
    move_edges1(i, act_occ_dict, tree_edges_pointers, non_tree_edges, tree_edges_group_by_levels,
                nontree_edges_group_by_levels)
    return


def move_edges1(i, act_occ_dict, tree_edges_pointers, non_tree_edges, tree_edges_group_by_levels,
                nontree_edges_group_by_levels):

    for j in range(i, -1, -1):  # update ET-trees
        for k in range(0, j):
            for u, v in tree_edges_group_by_levels[k]:  # move tree edges to level i
                insert_tree_edge(u, v, act_occ_dict, j, tree_edges_pointers)

    for j in range(0, i):  # move edges
        for u, v in nontree_edges_group_by_levels[j]:  # move non-tree edges to level i
            delete_nontree_edge(u, v, act_occ_dict, j, non_tree_edges, nontree_edges_group_by_levels)
            insert_nontree_edge(u, v, act_occ_dict, i, non_tree_edges, nontree_edges_group_by_levels)
            nontree_edges_group_by_levels[i].add((u, v))
        nontree_edges_group_by_levels[j].clear()
        for u, v in tree_edges_group_by_levels[j]:  # move tree edges to level i
            tree_edges_group_by_levels[i].add((u, v))
        tree_edges_group_by_levels[j].clear()
    #act_occ_dict[j] = defaultdict()
    #tree_edges_pointers[j] = defaultdict(list)
    return


def delete_edge(u, v, act_occ_dict, tree_edges_pointers, non_tree_edges,
                tree_edges_group_by_levels, nontree_edges_group_by_levels):
    # HK
    if (u, v) in non_tree_edges:
        i = get_level(u, v, nontree_edges_group_by_levels)
        nontree_edges_group_by_levels[i].remove((order(u, v)))
        delete_nontree_edge(u, v, act_occ_dict, i, non_tree_edges, nontree_edges_group_by_levels)
    else:
        i = get_level(u, v, tree_edges_group_by_levels)
        if i >= 0:  # (u, v) exists
            tree_edges_group_by_levels[i].remove((u, v))
            delete_tree_edge(u, v, non_tree_edges, act_occ_dict, i, tree_edges_pointers,
                             tree_edges_group_by_levels, nontree_edges_group_by_levels)
    return


def delete_nontree_edge(u, v, act_occ_dict, i, non_tree_edges, nontree_edges_group_by_levels):
    u_act_occ = act_occ_dict[i][u]
    v_act_occ = act_occ_dict[i][v]

    u_act_occ.nte.remove(v)
    remove_nte_node(u_act_occ, non_tree_edges[(u, v)])
    # print("nte-tree rooted at %d" % u_act_occ.val)
    # print_treap_nte_tree(u_act_occ.nte_tree)

    v_act_occ.nte.remove(u)
    remove_nte_node(v_act_occ, non_tree_edges[(v, u)])

    # update weight
    pointer = act_occ_dict[i][u]
    while pointer is not None:
        pointer.weight -= 1
        pointer = pointer.parent

    # update weight
    pointer = act_occ_dict[i][v]
    while pointer is not None:
        pointer.weight -= 1
        pointer = pointer.parent

    del non_tree_edges[(u, v)]
    del non_tree_edges[(v, u)]

    return


def delete_tree_edge(u, v, non_tree_edges, act_occ_dict, i, tree_edges_pointers, tree_edges_group_by_levels,
                     nontree_edges_group_by_levels):
    ii = i
    while ii >= 0:
        remove_tree_edge_i(u, v, act_occ_dict, ii, tree_edges_pointers)
        # clean up tree_edges_pointers
        ii -= 1
    replace(u, v, i, act_occ_dict, non_tree_edges, tree_edges_pointers, tree_edges_group_by_levels,
            nontree_edges_group_by_levels)
    return


def replace(u, v, i, act_occ_dict, non_tree_edges, tree_edges_pointers, tree_edges_group_by_levels,
            nontree_edges_group_by_levels):
    # r_u_i is the root of the spanning tree at level i that contains u
    # r_v_i is the root of the spanning tree at level i that contains v

    if i < 0:
        return
    r_u_i = find_root(act_occ_dict[i][u])
    r_v_i = find_root(act_occ_dict[i][v])
    if r_u_i.weight < r_v_i.weight:
        T1 = r_u_i
    else:
        T1 = r_v_i

    log_n_square = int(math.pow(math.log(graph_utils.n, 2), 2))

    if T1.weight >= log_n_square:  # sample&test c*log(n)^2 times
        for _ in range(log_n_square):
            res = sample_test(T1)
            if res is not None:  # case 1, replacement edges e' is found, terminate
                x, y = res
                nontree_edges_group_by_levels[i].remove((order(x, y)))
                delete_nontree_edge(x, y, act_occ_dict, i, non_tree_edges, nontree_edges_group_by_levels)
                tree_edges_group_by_levels[i].add((order(x, y)))
                for j in range(0, i + 1):
                    insert_tree_edge(x, y, act_occ_dict, j, tree_edges_pointers)
                return
    S = set()
    for (x, y) in nontree_edges(T1):
        if not connected(x, y, 0, act_occ_dict):
            S.add((x, y))
    c_prime = 4
    log_n = int(math.log(graph_utils.n, 2))
    if len(S) > T1.weight / (15 * c_prime * log_n):  # case 2.1
        s_iter = iter(S)
        t_x, t_y = next(s_iter)
        nontree_edges_group_by_levels[i].remove((order(t_x, t_y)))
        delete_nontree_edge(t_x, t_y, act_occ_dict, i, non_tree_edges, nontree_edges_group_by_levels)
        tree_edges_group_by_levels[i].add((t_x, t_y))
        for j in range(0, i + 1):
            insert_tree_edge(t_x, t_y, act_occ_dict, j, tree_edges_pointers)
    elif len(S) > 0:  # case 2.2
        s_iter = iter(S)
        t_x, t_y = next(s_iter)
        nontree_edges_group_by_levels[i].remove((order(t_x, t_y)))
        delete_nontree_edge(t_x, t_y, act_occ_dict, i, non_tree_edges, nontree_edges_group_by_levels)
        tree_edges_group_by_levels[i].add((t_x, t_y))
        for j in range(0, i + 1):
            insert_tree_edge(t_x, t_y, act_occ_dict, j, tree_edges_pointers)
        try:
            for t_x, t_y in next(s_iter):
                nontree_edges_group_by_levels[i].remove((order(t_x, t_y)))
                delete_nontree_edge(t_x, t_y, act_occ_dict, i, non_tree_edges, nontree_edges_group_by_levels)
                nontree_edges_group_by_levels[i - 1].add((order(t_x, t_y)))
                insert_nontree_edge(t_x, t_y, act_occ_dict, i - 1, non_tree_edges, nontree_edges_group_by_levels)
        except StopIteration:
            pass
        '''
        if i == max_level:
            tree_edges_group_by_levels[i].add((t_x, t_y))
            insert_tree_edge(t_x, t_y, act_occ_dict, i, tree_edges_pointers)
            return
        elif 0 <= i < max_level:
            tree_edges_group_by_levels[i + 1].add((t_x, t_y))
            for j in range(i, max_level + 1):
                insert_tree_edge(t_x, t_y, act_occ_dict, j, tree_edges_pointers)
            try:
                for t_x, t_y in next(s_iter):
                    delete_nontree_edge(t_x, t_y, act_occ_dict, i, non_tree_edges, nontree_edges_group_by_levels)
                    insert_nontree_edge(t_x, t_y, act_occ_dict, i + 1, non_tree_edges, nontree_edges_group_by_levels)
            except StopIteration:
                pass

            rebuild(i + 1, act_occ_dict, tree_edges_pointers, non_tree_edges, tree_edges_group_by_levels,
                    nontree_edges_group_by_levels)
        else:
            raise ValueError("wrong value for level i")
        '''
        return

    else:  # case 2.3
        replace(u, v, i - 1, act_occ_dict, non_tree_edges, tree_edges_pointers, tree_edges_group_by_levels,
                nontree_edges_group_by_levels)
    return


def add_nte_node(act_occ, nte_node):
    # non-tree neighborhood for a node are stored in a randomized tree
    # add a node (a non-tree edge) into the tree rooted at r
    root_of_nte_tree = act_occ.nte_tree
    if root_of_nte_tree is None:
        act_occ.nte_tree = nte_node
        return

    p = root_of_nte_tree
    p.weight += 1
    coin = random.randint(1, 2)
    if coin == 1:
        while p.left is not None:
            p = p.left
            p.weight += 1
    else:
        while p.right is not None:
            p = p.right
            p.weight += 1

    nte_node.parent = p
    if p.left is None:
        p.left = nte_node
    else:
        p.right = nte_node
    # rotate node to the place,which does not violate the priority value
    while p is not None and nte_node.priority > p.priority:
        if nte_node == p.right:
            root_of_nte_tree = rotateLeft(nte_node, root_of_nte_tree)
        else:
            root_of_nte_tree = rotateRight(nte_node, root_of_nte_tree)
        p = nte_node.parent

    act_occ.nte_tree = root_of_nte_tree

    return


def remove_nte_node(act_occ, nte_node):
    # non-tree neighborhood for a node are stored in a randomized tree
    root_of_nte_tree = act_occ.nte_tree
    if root_of_nte_tree.left is None and root_of_nte_tree.right is None:
        act_occ.nte_tree = None
        return

    # rotate root to leaf
    nte_node, root_of_nte_tree = rotate_to_leaf(nte_node, root_of_nte_tree)

    p = nte_node.parent
    if p.left == nte_node:
        p.left = None
    else:
        p.right = None
    while p is not None:
        p.weight -= 1
        p = p.parent
    act_occ.nte_tree = root_of_nte_tree
    return


def sample_test(T1):  # use the same notation in the HKS's paper

    rnd = random.randint(1, T1.weight)
    cur, offset = locate_by_weight(T1, rnd)
    nte_node, _ = locate_by_weight(cur.nte_tree, offset)

    root1 = find_root(cur)
    root2 = find_root(nte_node.act_occ)
    if root1 != root2:
        # return cur.val, root1, nte_node.act_occ.val, root2
        return order(cur.val, nte_node.act_occ.val)
    else:
        return None


def locate_by_weight(r, rnd):
    # this function is used by sample_test, sampling on ET-tree
    # Each active occurrence ox has a weight w(ox), i.e., the number of non-tree edge incident to ox
    # Active occurrences in ET-tree are sorted by their position in the Euler Tour.
    # Assume o1, o2. ... ox ... are the ordered active occurrences in the Euler Tour
    # Given a random number rnd, the goal to find the active occurrence ox such that
    # sum(w(o1) + w(o2) + ... w(o_x-1)) < rnd <=  sum(w(o1) + w(o2) + ... w(o_x-1) + w(ox))
    # complexity: O(h) where h is the height of the tree. h = log n. Hence, O(log n).
    cur = r
    if cur.left is None:
        low = 0
    else:
        low = cur.left.weight
    high = low + node_weight(r)
    while rnd <= low or rnd > high:
        if rnd <= low:
            cur = cur.left
            if cur.left is None:
                low = 0
            else:
                low = cur.left.weight
            high = low + node_weight(cur)
        else:
            rnd -= high
            cur = cur.right
            if cur.left is None:
                low = 0
            else:
                low = cur.left.weight
            high = low + node_weight(cur)

    return cur, rnd - low


def node_weight(T):
    # given the root of the subtree T rooted at u
    # return how many non-tree edge incident to node u
    node_weight = T.weight
    if T.left is not None:
        node_weight -= T.left.weight
    if T.right is not None:
        node_weight -= T.right.weight
    return node_weight


def nontree_edges(root):
    # root of an ETR-tree
    if root.weight == 0:
        return []
    ntes = set()
    q = queue.Queue()
    q.put(root)
    while q.qsize() > 0:
        node = q.get()
        if node.nte is not None and len(node.nte) > 0:
            for endpoint in node.nte:
                ntes.add((order(node.val, endpoint)))
        if node.left is not None and node.left.weight > 0:
            q.put(node.left)
        if node.right is not None and node.right.weight > 0:
            q.put(node.right)

    return ntes


def rebuild(i, act_occ_dict, tree_edges_pointers, non_tree_edges, tree_edges_group_by_levels,
            nontree_edges_group_by_levels):
    max_level = 2 * graph_utils.max_level
    if i < 3:
        return
    sum_added_edges = 0
    for j in range(i, max_level + 1):
        sum_added_edges += added_edges[j]
    if sum_added_edges > rebuild_bound[i]:
        move_edges(i, act_occ_dict, tree_edges_pointers, non_tree_edges, tree_edges_group_by_levels,
                   nontree_edges_group_by_levels)
        for j in range(i, max_level + 1):
            added_edges[j] = 0
    return


def move_edges(i, act_occ_dict, tree_edges_pointers, non_tree_edges, tree_edges_group_by_levels,
               nontree_edges_group_by_levels):
    max_level = 2 * graph_utils.max_level
    for j in range(max_level, i + 1, -1):
        for u, v in nontree_edges_group_by_levels[j]:  # move non-tree edges
            delete_nontree_edge(u, v, act_occ_dict, j, non_tree_edges, nontree_edges_group_by_levels)
            insert_nontree_edge(u, v, act_occ_dict, i - 1, non_tree_edges, nontree_edges_group_by_levels)
        for u, v in tree_edges_group_by_levels[j]:  # move tree edges
            insert_tree_edge(u, v, act_occ_dict, i - 1, tree_edges_pointers)
    return


def remove_tree_edge_i(u, v, act_occ_dict, i, tree_edges_pointers):
    # this function remove tree edge at level i from ET-Tree
    root_u = find_root(act_occ_dict[i][u])
    root = root_u
    edge = order(u, v)
    first_pointer, last_pointer = sort(tree_edges_pointers[i][edge])
    s1, right_branch = split_after(root, first_pointer)
    s2, s3 = split_before(right_branch, last_pointer)
    merge_update_pointer(s1, first_pointer, s3, last_pointer, tree_edges_pointers[i])

    del tree_edges_pointers[i][edge]

    return


def find_root(node):
    p = node

    while p.parent is not None:
        p = p.parent
    return p


def connected(u, v, i, act_occ_dict):
    if u not in act_occ_dict[i] or v not in act_occ_dict[i]:
        return False
    root_u = find_root(act_occ_dict[i][u])
    root_v = find_root(act_occ_dict[i][v])
    return root_v == root_u


def query(u, v, act_occ_dict):
    return find_root(act_occ_dict[0][u]) == find_root(act_occ_dict[0][v])


def cal_node_edge_size(node):
    node_size = 0
    edge_size = 0
    node_size += sys.getsizeof(node.val)
    if node.parent is not None:
        edge_size += sys.getsizeof(node.parent)
    if node.left is not None:
        edge_size += sys.getsizeof(node.left)
    if node.right is not None:
        edge_size += sys.getsizeof(node.right)
    if node.nte is not None:
        edge_size += sys.getsizeof(node.nte)
    if node.active is not None:
        node_size += sys.getsizeof(node.active)
    if hasattr(node, 'weight'):
        node_size += sys.getsizeof(node.weight)
    if hasattr(node, 'size'):
        node_size += sys.getsizeof(node.size)
    node_size += sys.getsizeof(node.priority)

    if hasattr(node, 'nte_tree'):
        nte_tree_size = 0
        # calculate size of nte_tree
        if node.nte_tree is not None:
            q = queue.Queue()
            q.put(node.nte_tree)
            while not q.empty():
                cur = q.get()
                nte_tree_size += cal_nte_tree_node_size(cur)
                if cur.left is not None:
                    q.put(cur.left)
                if cur.right is not None:
                    q.put(cur.right)
        edge_size += nte_tree_size
    return node_size, edge_size


def cal_total_memory_use(act_occ_dict, tree_edges_pointers, non_tree_edges, tree_edges_group_by_levels,
                         nontree_edges_group_by_levels):
    # collect root nodes of all ET-trees
    root_node_set = set()
    act_occ_dict_size = 0
    for level, act_occ_dict_i in act_occ_dict.items():
        act_occ_dict_size += sys.getsizeof(level)  # calculate memory usage for act_dict
        act_occ_dict_size += sys.getsizeof(act_occ_dict_i)  # calculate memory usage for act_dict
        for key, node in act_occ_dict_i.items():
            while node.parent is not None:
                node = node.parent
            root_node_set.add(node)

    space_n = 0
    space_e = 0
    for root_node in root_node_set:
        q = queue.Queue()
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

    # print("overall node sizes : %d bytes" % node_size)
    # print("memory size for act_occ_dict: %d bytes" % act_occ_dict_size)
    # print("memory size for non-tree edges pointers: %d bytes" % sys.getsizeof(non_tree_edges))

    tree_edge_pointers_size = 0
    for et_pointers in tree_edges_pointers:
        for edge, edge_pointers_list in et_pointers.items():
            tree_edge_pointers_size += sys.getsizeof(edge)
            tree_edge_pointers_size += (sys.getsizeof(edge_pointers_list))
    # print("memory size for tree edge pointers: %d bytes" % tree_edge_pointers_size)
    non_tree_edge_pointers_size = sys.getsizeof(non_tree_edges)

    level_of_tree_edge_size = 0
    for level, edge_set in tree_edges_group_by_levels.items():
        level_of_tree_edge_size += sys.getsizeof(level)
        level_of_tree_edge_size += sys.getsizeof(edge_set)
    # print("memory size for level of tree edges: %d bytes" % level_of_tree_edge_size)

    level_of_nontree_edge_size = 0
    for level, edge_set in nontree_edges_group_by_levels.items():
        level_of_nontree_edge_size += sys.getsizeof(level)
        level_of_nontree_edge_size += sys.getsizeof(edge_set)
    # print("memory size for level of nontree edges: %d bytes" % level_of_nontree_edge_size)

    # print("memory size for nontree edge: %d bytes" % non_tree_edge_pointers_size)
    space_n += act_occ_dict_size
    space_e += (tree_edge_pointers_size + level_of_tree_edge_size + level_of_nontree_edge_size + non_tree_edge_pointers_size)
    # print("node size: %d, edge size: %d" %(node_size, edge_size))
    # print("total memory size : %d bytes" % (node_size + edge_size))
    return space_n, space_e


def leading_bit(n):
    b = 0
    pos = 0
    while n > 0:
        if n & 1 == 1:
            b = pos
        pos += 1
        n = n >> 1
    return b