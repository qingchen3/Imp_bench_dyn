"""
This file is an implementation of HDT algorithm
Poly-Logarithmic Deterministic Fully-Dynamic Graph Algorithms
"""

from utils.tree_utils import sort
from utils.graph_utils import order
import sys, random, queue
from HDT.HDT_Node import HDT_Node
from HK.updates import cal_node_edge_size, get_level


def insert_edge(u, v, act_occ_dict, tree_edges_pointers, tree_edges_group_by_levels, nontree_edges_group_by_levels):
    # initialize HKNode for HK if not exists
    if not connected(u, v, 0, act_occ_dict):
        tree_edges_group_by_levels[0].add((u, v))
        insert_tree_edge(u, v, act_occ_dict, 0, tree_edges_pointers)
    else:  # a and b are connected
        if not is_nontree_edge(u, v, nontree_edges_group_by_levels):
            nontree_edges_group_by_levels[0].add((u, v))
            insert_nontree_edge(u, v, act_occ_dict, 0)
    return


def insert_nontree_edge(u, v, act_occ_dict, i):
    # non_tree_edges.add((u, v))
    while i >= 0:
        u_act_occ = act_occ_dict[i][u]
        v_act_occ = act_occ_dict[i][v]
        if u_act_occ.nte is None:
            u_act_occ.nte = set()
        u_act_occ.nte.add(v)
        if v_act_occ.nte is None:
            v_act_occ.nte = set()
        v_act_occ.nte.add(u)

        i -= 1

    return


def insert_tree_edge(u, v, act_occ_dict, i, tree_edges_pointers):
    # this function add tree edge at level i into ET-Tree
    max_priority = sys.maxsize
    if u not in act_occ_dict[i]:
        node = HDT_Node(u, random.randint(1, max_priority))
        node.active = True
        node.size = 1
        act_occ_dict[i][u] = node

    if v not in act_occ_dict[i]:
        node = HDT_Node(v, random.randint(1, max_priority))
        node.active = True
        node.size = 1
        act_occ_dict[i][v] = node

    root_u = find_root(act_occ_dict[i][u])
    root_v = find_root(act_occ_dict[i][v])

    root_u = reroot(root_u, u, act_occ_dict[i], tree_edges_pointers[i])
    root_v = reroot(root_v, v, act_occ_dict[i], tree_edges_pointers[i])

    tail_of_root_u = tail_of_etr_tree(root_u)
    head_of_root_v = head_of_etr_tree(root_v)
    max_priority = sys.maxsize
    # add a new occurrence of root_a to the end new ETR
    node = HDT_Node(u, random.randint(1, max_priority))

    # update tree-edge-pointers
    tree_edges_pointers[i][(u, v)].append(tail_of_root_u)
    tree_edges_pointers[i][(u, v)].append(head_of_root_v)
    if root_v.left is not None or root_v.right is not None:
        tail_of_root_b = tail_of_etr_tree(root_v)
        tree_edges_pointers[i][(u, v)].append(tail_of_root_b)

    # tree_edges_pointers[(u, v)].append(node)
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

    merge(root_u, root_v)

    return


def delete_edge(u, v, act_occ_dict, tree_edges_pointers, tree_edges_group_by_levels, nontree_edges_group_by_levels):
    if is_nontree_edge(u, v, nontree_edges_group_by_levels):
        i = get_level(u, v, nontree_edges_group_by_levels)
        nontree_edges_group_by_levels[i].remove((u, v))
        delete_nontree_edge(u, v, act_occ_dict, i)
    else:
        i = get_level(u, v, tree_edges_group_by_levels)
        if i >= 0:  # (u, v) exists
            tree_edges_group_by_levels[i].remove((u, v))
            delete_tree_edge(u, v, act_occ_dict, i, tree_edges_pointers,
                             tree_edges_group_by_levels, nontree_edges_group_by_levels)
    return


def delete_nontree_edge(u, v, act_occ_dict, i):
    #non_tree_edges.remove((u, v))
    while i >= 0:
        u_act_occ = act_occ_dict[i][u]
        v_act_occ = act_occ_dict[i][v]

        u_act_occ.nte.remove(v)
        v_act_occ.nte.remove(u)

        i -= 1
    return


def delete_tree_edge(u, v, act_occ_dict, i, tree_edges_pointers,
                     tree_edges_group_by_levels, nontree_edges_group_by_levels):
    ii = i
    while ii >= 0:
        remove_tree_edge_i(u, v, act_occ_dict, ii, tree_edges_pointers)
        # clean up tree_edges_pointers
        ii -= 1
    replace(u, v, i, act_occ_dict, tree_edges_pointers,
            tree_edges_group_by_levels, nontree_edges_group_by_levels)
    return


def replace(u, v, i, act_occ_dict, tree_edges_pointers,
            tree_edges_group_by_levels, nontree_edges_group_by_levels):
    # r_u_i is the root of the spanning tree at level i that contains u
    # r_v_i is the root of the spanning tree at level i that contains v

    if i < 0:
        return
    r_u_i = find_root(act_occ_dict[i][u])
    r_v_i = find_root(act_occ_dict[i][v])
    if r_u_i.size <= r_v_i.size:
        T1 = r_u_i
    else:
        T1 = r_v_i

    euler_tour = []
    non_tree_edges_T1 = set()
    tree_nodes_T1 = set()
    tree_edges_T1 = set()
    inorder_euler_tour(T1, i, euler_tour, tree_edges_group_by_levels, tree_nodes_T1, tree_edges_T1)
    replacement = []
    for tree_node in tree_nodes_T1:
        current_node = act_occ_dict[i][tree_node]
        if current_node.nte is not None:
            for xx in current_node.nte:
                nte_x, nte_y = order(tree_node, xx)
                if get_level(nte_x, nte_y, nontree_edges_group_by_levels) == i:
                    if not connected(nte_x, nte_y, i, act_occ_dict):
                        replacement.append(order(nte_x, nte_y))
                    else:
                        non_tree_edges_T1.add((nte_x, nte_y))

    for te_x, te_y in tree_edges_T1:
        tree_edges_group_by_levels[i].remove((te_x, te_y))
        tree_edges_group_by_levels[i + 1].add((te_x, te_y))
        insert_tree_edge(te_x, te_y, act_occ_dict, i + 1, tree_edges_pointers)

    for (nte_x, nte_y) in non_tree_edges_T1:  # pushing non-tree edge from level i to i + 1
        nontree_edges_group_by_levels[i].remove((nte_x, nte_y))
        delete_nontree_edge(nte_x, nte_y, act_occ_dict, i)
        nontree_edges_group_by_levels[i + 1].add((nte_x, nte_y))
        insert_nontree_edge(nte_x, nte_y, act_occ_dict, i + 1)
    if len(replacement) != 0:
        replace_x, replace_y = replacement[0]
        nontree_edges_group_by_levels[i].remove((replace_x, replace_y))
        delete_nontree_edge(replace_x, replace_y, act_occ_dict, i)

        tree_edges_group_by_levels[i].add((replace_x, replace_y))
        insert_tree_edge(replace_x, replace_y, act_occ_dict, i, tree_edges_pointers)

        ii = i - 1
        while ii >= 0:
            insert_tree_edge(replace_x, replace_y, act_occ_dict, ii, tree_edges_pointers)
            ii -= 1
        return

    else:
        replace(u, v, i - 1, act_occ_dict, tree_edges_pointers,
                tree_edges_group_by_levels, nontree_edges_group_by_levels)

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


def inorder_euler_tour(root, i, euler_tour, tree_edges_group_by_levels, tree_nodes_i, tree_edges_i):
    if root is None:
        return

    if root.left is not None:
        inorder_euler_tour(root.left, i, euler_tour, tree_edges_group_by_levels, tree_nodes_i, tree_edges_i)

    euler_tour.append(root.val)
    tree_nodes_i.add(root.val)
    if len(euler_tour) > 1:
        x, y = order(euler_tour[-1], euler_tour[-2])
        if get_level(x, y, tree_edges_group_by_levels) == i:
            tree_edges_i.add((x, y))
    if root.right is not None:
        inorder_euler_tour(root.right, i, euler_tour, tree_edges_group_by_levels, tree_nodes_i, tree_edges_i)

    return


def inorder1(root, sequence, ntes, i, act_occ_dict, replacement, nontree_edges_group_by_levels, level_for_edges):
    if root is None:
        return

    if root.left is not None:
        inorder1(root.left, sequence, ntes, i, act_occ_dict, replacement, nontree_edges_group_by_levels,
                 level_for_edges)

    if root.active is True:
        for endpoint in root.nte:
            nte_x, nte_y = order(root.val, endpoint)
            if not connected(root.val, endpoint, i, act_occ_dict):
                replacement.append(order(root.val, endpoint))
            else:
                ntes.add((nte_x, nte_y))
    sequence.append(root.val)
    if root.right is not None:
        inorder1(root.right, sequence, ntes, i, act_occ_dict, replacement, nontree_edges_group_by_levels,
                 level_for_edges)

    return


def eulertour_to_edges(sequence, level_for_edges, i):
    edges = set()
    for idx in range(len(sequence) - 1):
        x = sequence[idx]
        y = sequence[idx + 1]
        x, y = order(x, y)
        if level_for_edges[(x, y)] == i:
            edges.add((x, y))
    return edges


def is_nontree_edge(u, v, adjacency_level_nt):
    for level, edge_set in adjacency_level_nt.items():
        if (u, v) in edge_set:
            return True
    return False


def get_nontree_edges(root):
    if root.size == 0:
        return []
    ntes = set()
    q = queue.Queue()
    q.put(root)
    while q.qsize() > 0:
        node = q.get()
        if len(node.nte) > 0:
            for endpoint in node.nte:
                ntes.add((order(node.val, endpoint)))
        if node.left is not None:
            q.put(node.left)
        if node.right is not None:
            q.put(node.right)
    return ntes


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

    p.size = p.size - x.size

    # cut right branch of x
    x_right = x.right
    x_right_size = 0
    if x_right is not None:
        x_right_size = x_right.size
    x.size -= x_right_size

    # attach right branch of x to p
    if x_right is not None:
        x_right.parent = p

    p.left = x_right
    p.size += x_right_size

    # attach p as the right branch of x
    x.right = p
    p.parent = x
    x.size += p.size

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

    p.size = p.size - x.size

    # cut left branch of x
    x_left = x.left
    x_left_size = 0
    if x_left is not None:
        x_left_size = x_left.size

    x.size -= x_left_size

    # attach left branch of x to p
    if x_left is not None:
        x_left.parent = p

    p.right = x_left
    p.size += x_left_size

    # attach p as the left branch of x
    x.left = p
    p.parent = x
    x.size += p.size

    if g is not None:
        return root
    else:
        return x


def rotate_to_root(x, root):
    while x != root:
        p = x.parent
        if x == p.left:
            root = rotateRight(x, root)
        else:
            root = rotateLeft(x, root)
    return x


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


def split_before(root, current_node):
    node = HDT_Node(-1, sys.maxsize)
    if current_node.left is None:
        current_node.left = node
        node.parent = current_node
    else:
        pointer = current_node.left
        while pointer.right is not None:
            pointer = pointer.right
        pointer.right = node
        node.parent = pointer

    root = rotate_to_root(node, root)
    left_branch = root.left
    left_branch.parent = None

    right_branch = root.right
    right_branch.parent = None

    return left_branch, right_branch


def split_after(root, current_node):
    node = HDT_Node(-1, sys.maxsize)
    if current_node.right is None:
        current_node.right = node
        node.parent = current_node
    else:
        pointer = current_node.right
        while pointer.left is not None:
            pointer = pointer.left
        pointer.left = node
        node.parent = pointer

    root = rotate_to_root(node, root)

    left_branch = root.left
    left_branch.parent = None

    right_branch = root.right
    right_branch.parent = None

    return left_branch, right_branch


def reroot(root, u, active_occurrence_dict, tree_edges_pointers):
    head = head_of_etr_tree(root)
    if head.val == u:
        return root
    max_priority = sys.maxsize
    pred_u = predecessor(active_occurrence_dict[u])
    succ_u = successor(active_occurrence_dict[u])

    root_left_branch, root_right_branch = split_before(root, active_occurrence_dict[u])

    node = HDT_Node(u, random.randint(1, max_priority))

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


def merge(r1, r2):
    root = HDT_Node(-1, sys.maxsize)
    root.size = r1.size + r2.size
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


def cal_total_memory_use(act_occ_dict, tree_edges_pointers, tree_edges_group_by_levels,
                         nontree_edges_group_by_levels):
    root_node_set = set()
    act_occ_dict_size = 0
    for level, node_dictionary in act_occ_dict.items():
        act_occ_dict_size += sys.getsizeof(level)
        act_occ_dict_size += sys.getsizeof(node_dictionary)
        for key, node in node_dictionary.items():
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
    space_n += act_occ_dict_size

    tree_edge_pointers_size = 0
    for tree_edge_pointer_dict in tree_edges_pointers:
        for edge, edge_pointers_list in tree_edge_pointer_dict.items():
            tree_edge_pointers_size += sys.getsizeof(edge)
            tree_edge_pointers_size += sys.getsizeof(edge_pointers_list)
    # print("memory size for tree edge pointers: %d bytes" % tree_edge_pointers_size)

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

    space_e += (tree_edge_pointers_size + level_of_tree_edge_size + level_of_nontree_edge_size)
    # print("node size: %d, edge size: %d" % (node_size, edge_size))
    # print("total memory size : %d bytes" % (node_size + edge_size))
    return space_n, space_e
