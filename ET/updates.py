from ET.ET_Node import ET_Node
from timeit import default_timer as timer
from queue import Queue

from utils.tree_utils import coding, smaller, sort
from utils.graph_utils import order
import sys, random


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


def find_root(node):
    p = node

    while p.parent is not None:
        p = p.parent
    return p


def find_root_with_distance(node):
    p = node
    d = 0
    while p.parent is not None:
        p = p.parent
        d += 1
    return p, d


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


def split_before(root, current_node):
    node = ET_Node(-1, sys.maxsize)
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
    node = ET_Node(-1, sys.maxsize)
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
    root = ET_Node(-1, sys.maxsize)
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


def merge_update_pointer(r1, rightmost_r1, r2, leftmost_r2, tree_edges_pointer):
    if rightmost_r1 is None:
        rightmost_r1 = r1
        while rightmost_r1.right is not None:
            rightmost_r1 = rightmost_r1.right
    #else:
        # update tree-edge pointer
        #t_rightmost_r1 = r1
        #while t_rightmost_r1.right is not None:
        #    t_rightmost_r1 = t_rightmost_r1.right
        #if t_rightmost_r1 != rightmost_r1:
        #    raise ValueError("Errors in pointer in r1 ")

    if leftmost_r2 is None:
        leftmost_r2 = r2
        while leftmost_r2.left is not None:
            leftmost_r2 = leftmost_r2.left
    #else:
        #t_leftmost_r2 = r2
        #while t_leftmost_r2.left is not None:
        #    t_leftmost_r2 = t_leftmost_r2.left
        #if t_leftmost_r2 != leftmost_r2:
        #    raise ValueError("Errors in pointer in r2")

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
    max_priority = sys.maxsize
    pred_u = predecessor(active_occurrence_dict[u])
    succ_u = successor(active_occurrence_dict[u])

    root_left_branch, root_right_branch = split_before(root, active_occurrence_dict[u])

    node = ET_Node(u, random.randint(1, max_priority))

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


def insert_tree_edge(a, root_a, b, root_b, tree_edges, active_occurrence_dict, tree_edges_pointers, max_priority):
    # for update tree_edges and non_tree_edges, rename (a, b) to (x, y) for one-time use
    (x, y) = order(a, b)
    if root_a.size < root_b.size: # root_b is the root of the smaller tree.
        root_a, root_b = root_b, root_a
        a, b = b, a

    tree_edges.add((x, y))
    if root_a is None:
        root_a = find_root(active_occurrence_dict[a])
    if root_b is None:
        root_b = find_root(active_occurrence_dict[b])

    root_a = reroot(root_a, a, active_occurrence_dict, tree_edges_pointers)
    root_b = reroot(root_b, b, active_occurrence_dict, tree_edges_pointers)

    tail_of_root_a = tail_of_etr_tree(root_a)
    head_of_root_b = head_of_etr_tree(root_b)

    # add a new occurrence of root_a to the end new ETR
    node = ET_Node(a, random.randint(1, max_priority))

    # update tree-edge-pointers
    tree_edges_pointers[(x, y)].append(tail_of_root_a)
    tree_edges_pointers[(x, y)].append(head_of_root_b)
    if root_b.left is not None or root_b.right is not None:
        tail_of_root_b = tail_of_etr_tree(root_b)
        tree_edges_pointers[(x, y)].append(tail_of_root_b)

    tree_edges_pointers[(x, y)].append(node)
    #print("appending new node takes %f" % (time.clock() - t))

    #t = time.clock()
    # attach node in the end of ETR-tree
    tail_of_root_b = tail_of_etr_tree(root_b)
    tail_of_root_b.right = node
    node.parent = tail_of_root_b
    ##
    # rotate node to the place,which does not violate the priority value
    p = tail_of_root_b
    while p is not None and node.priority > p.priority:
        if node == p.right:
            root_b = rotateLeft(node, root_b)
        else:
            root_b = rotateRight(node, root_b)
        p = node.parent

    root = merge(root_a, root_b) # attach smaller tree to larger tree

    return root


def insert_nontree_edge(u, v, active_occurrence_dict, non_tree_edges):

    active_occurrence_dict[u].nte.add(active_occurrence_dict[v])
    active_occurrence_dict[v].nte.add(active_occurrence_dict[u])
    non_tree_edges.add((u, v))


def delete_nontree_edge(u, v, active_occurrence_dict, non_tree_edges):

    active_occurrence_dict[u].nte.remove(active_occurrence_dict[v])
    active_occurrence_dict[v].nte.remove(active_occurrence_dict[u])

    non_tree_edges.remove((u, v))


def delete_tree_edge(u, v, tree_edges, non_tree_edges, active_occurrence_dict, tree_edges_pointers, max_priority):
    root_u = find_root(active_occurrence_dict[u])
    root = root_u
    edge = order(u, v)
    first_pointer, last_pointer = sort(tree_edges_pointers[edge])
    s1, right_branch = split_after(root, first_pointer)
    s2, s3 = split_before(right_branch, last_pointer)
    r2 = s2
    r1 = merge_update_pointer(s1, first_pointer, s3, last_pointer, tree_edges_pointers)

    # clean up tree_edges_pointers
    del tree_edges_pointers[edge]

    tree_edges.remove((u, v))

    start = timer()
    # find the replacement edge
    a, root_of_a, b, root_of_b = BFS_select(r1, r2)
    t = timer() - start

    # no replacement edges are found.
    if a is None and root_of_a is None and b is None and root_of_b is None:
        return t

    # delete non-tree edge (a, b)
    t_a, t_b = order(a, b)
    delete_nontree_edge(t_a, t_b, active_occurrence_dict, non_tree_edges)

    # insert replacement edge
    insert_tree_edge(a, root_of_a, b, root_of_b, tree_edges, active_occurrence_dict, tree_edges_pointers, max_priority)

    return t


def BFS_select(r1, r2):

    if r1.size < r2.size:
        r = r1
    else:
        r = r2

    minimal_dist = sys.maxsize
    q = Queue()
    q.put(r)
    n_rs = None
    r_s = None
    n_rl = None
    r_l = None
    while not q.empty():
        node = q.get()
        if len(node.nte) > 0:
            for nte in node.nte:
                rt, d = find_root_with_distance(nte)
                if rt.val == r.val:  # this non tree edge is included in the smaller tree.
                    continue
                if d < minimal_dist:
                    minimal_dist = d
                    n_rs = node.val
                    r_s = r
                    n_rl = nte.val
                    r_l = rt
        if node.left is not None:
            q.put(node.left)
        if node.right is not None:
            q.put(node.right)

    return n_rs, r_s, n_rl, r_l

