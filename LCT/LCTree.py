from LCT.MySplayTree import MyNode
import LCT.MySplayTree as splayTree
import sys


def make_tree(v):
    return MyNode(v)


def detach_child(node):  # node is the root of a splay tree
    if node.right is not None:
        r_ch = node.right
        r_ch.parent = None
        r_ch.lc_parent = node
    return


def access(node):

    splayTree.splay(node)

    detach_child(node)
    node.right = None

    while node.lc_parent is not None:
        par = node.lc_parent
        splayTree.splay(par)
        detach_child(par)
        par.right = None

        # attaching splay tree rooted at cur as the right child of par (root of another splay tree)
        par.right = node
        node.parent = par
        node.lc_parent = None

        splayTree.splay(node)

    return node


def cut(node):  # cut the node from the preferred path
    access(node)
    node.left.parent = None
    node.left = None
    return


def link(par, child):  # par is the parent of in the preferred path (not in the splay tree)
    access(child)
    access(par)
    par.parent = child
    child.left = par
    return


def insert(u, v, LCT):
    if u not in LCT:
        LCT[u] = make_tree(u)
    if v not in LCT:
        LCT[v] = make_tree(v)

    if traversal_top(LCT[u]) == traversal_top(LCT[v]):  # (u, v) is a tree edge
        if LCT[u].nte is None:
            LCT[u].nte = set()

        if LCT[v].nte is None:
            LCT[v].nte = set()

        LCT[u].nte.add(v)
        LCT[v].nte.add(u)

        return
    else:  # u is the parent of v in the preferred path
        par = LCT[u]
        ch = LCT[v]

        access(par)
        splayTree.splay(par)
        if par.right is not None:
            par_right = par.right
            par_right.lc_parent = par
            par_right.parent = None

        access(ch)
        splayTree.splay(ch)
        if ch.left is not None:
            ch_left = ch.left
            # ch_left.lc_parent = ch

            ch_left.parent = None
            ch.left = None

            # reverse the splay tree rooted at ch_left
            new_ch_left = reverse(ch_left, LCT)
            new_ch_left.parent = None
            new_ch_left.lc_parent = ch

        par.right = ch
        ch.parent = par

        splayTree.splay(ch)  # access(ch)

        return


def insert1(u, v, LCT):
    if u not in LCT:
        LCT[u] = make_tree(u)
    if v not in LCT:
        LCT[v] = make_tree(v)

    if traversal_top(LCT[u]) == traversal_top(LCT[v]):  # (u, v) is a tree edge
        if LCT[u].nte is None:
            LCT[u].nte = set()

        if LCT[v].nte is None:
            LCT[v].nte = set()

        LCT[u].nte.add(v)
        LCT[v].nte.add(u)

        return
    else:  # u is the parent of v in the preferred path
        upper_node = LCT[u]
        lower_node = LCT[v]

        access(lower_node)
        splayTree.splay(lower_node)

        if lower_node.left is not None:
            node_left = lower_node.left

            node_left.parent = None
            lower_node.left = None

            # reverse the splay tree rooted at ch_left
            new_ch_left = reverse(node_left, LCT)
            new_ch_left.parent = None
            new_ch_left.lc_parent = lower_node

        access(upper_node)
        splayTree.splay(upper_node)
        if upper_node.right is not None:
            node_right = upper_node.right

            node_right.parent = None
            upper_node.right = None
            node_right.lc_parent = upper_node

        upper_node.right = lower_node
        lower_node.parent = upper_node

        return


def delete(u, v, LCT):

    if LCT[u].nte is not None and LCT[v].nte is not None and (u in LCT[v].nte or v in LCT[u].nte):
        # (u, v) is a non-tree edge
        LCT[v].nte.remove(u)
        LCT[u].nte.remove(v)
        return
    # (u, v) or (v, u) is a tree edge
    par = None
    ch = None
    if splayTree.predecessor(LCT[u]) == LCT[v]:  # v is the parent of u in the preferred path
        # cut(LCT[u])
        par = LCT[v]
        ch = LCT[u]

        # access(par)
        splayTree.splay(par)  # ch is the root of the link-cut tree

        # if par.right is not None:
        par.right.parent = None
        # !!!!!!!!!!!!!!!!!! par.parent = None  # par = ch.left
        assert par.right.lc_parent is None
        par.right = None

        '''
        access(ch)
        splayTree.splay(ch)  # ch is the root of the link-cut tree

        ch.left.parent = None
        # !!!!!!!!!!!!!!!!!! par.parent = None  # par = ch.left
        assert ch.left.lc_parent is None
        ch.left = None
        '''
    elif splayTree.predecessor(LCT[v]) == LCT[u]:  # u is the parent of v in the preferred path
        # cut(LCT[v])
        par = LCT[u]
        ch = LCT[v]

        # access(par)
        splayTree.splay(par)  # ch is the root of the link-cut tree

        # if par.right is not None:
        par.right.parent = None
        # !!!!!!!!!!!!!!!!!! par.parent = None  # par = ch.left
        assert par.right.lc_parent is None
        par.right = None

        '''
        access(ch)
        splayTree.splay(ch)  # ch is the root of the link-cut tree

        # !!!!!!!!!!!!!!!!!!!! par.parent = None  # par = ch.left
        ch.left.parent = None
        assert ch.left.lc_parent is None
        ch.left = None
        '''
    else:  # (u, v) or (v, u) is an edge linking two splay trees
        splayTree.splay(LCT[u])
        if LCT[u].lc_parent == LCT[v]:  # v is the parent of u
            par = LCT[v]
            ch = LCT[u]
            LCT[u].lc_parent = None

        else:
            splayTree.splay(LCT[v])
            if LCT[v].lc_parent != LCT[u]:  # u is the parent of v
                raise ValueError("edge (%d, %d) does not exist" %(u, v))
            par = LCT[u]
            ch = LCT[v]

            LCT[v].lc_parent = None

    # delete (u, v) or (v, u) and search for a replacement edge
    # access(ch)
    # splayTree.splay(ch)  # ch is the root of the link-cut tree

    # par.parent = None  # par = ch.left
    # assert par.lc_parent is None
    # ch.left = None

    # search subtree rooted at ch for a replacement edge

    for val, lower_node in LCT.items():
        if minimum(lower_node) == ch:  # search the subtree rooted at ch
            if lower_node.nte is not None:
                for t_x in lower_node.nte:
                    if minimum(LCT[t_x]) != ch:  # (val, t_x) is a replacement
                        upper_node = LCT[t_x]  # attach

                        upper_node.nte.remove(val)
                        lower_node.nte.remove(t_x)

                        # attach lower_node as a child of upper_node in the preferred path
                        access(lower_node)
                        splayTree.splay(lower_node)

                        if lower_node.left is not None:
                            node_left = lower_node.left

                            node_left.parent = None
                            lower_node.left = None

                            # reverse the splay tree rooted at ch_left
                            new_ch_left = reverse(node_left, LCT)
                            new_ch_left.parent = None
                            new_ch_left.lc_parent = lower_node

                        access(upper_node)
                        splayTree.splay(upper_node)
                        if upper_node.right is not None:
                            node_right = upper_node.right

                            node_right.parent = None
                            upper_node.right = None
                            node_right.lc_parent = upper_node

                        upper_node.right = lower_node
                        lower_node.parent = upper_node

                        splayTree.splay(lower_node)  # access(node)

                        '''
                        print("find a replacemnet edge (%d, %d)" %(val, t_x))
                        access(node)
                        splayTree.splay(node)

                        access(LCT[t_x])
                        splayTree.splay(LCT[t_x])
                        assert LCT[t_x].right is None

                        # attach node to LCT[t_x]
                        node.parent = LCT[t_x]
                        LCT[t_x].right = node

                        LCT[t_x].nte.remove(node.val)
                        node.nte.remove(t_x)

                        LCT[t_x].lc_parent = node.lc_parent
                        node.lc_parent = None
                        '''
                        return

    # there is no replacement edge
    # access(par)

    return


def reverse(root, LCT):  # reverse one path (and store nodes (keyed by depth) on the path in a balanced binary tree)
    node_list = list()
    inorder_traversal(root, node_list)
    new_root = reconstruct(LCT, node_list, 0, len(node_list) - 1)
    reverse_list = list()
    inorder_traversal(new_root, reverse_list)
    return new_root


def reconstruct(LCT, node_list, low, high):
    if low > high:
        return None
    mid = (low + high + 1) // 2

    root = LCT[node_list[mid]]
    right_ch = reconstruct(LCT, node_list, low, mid - 1)  # sort in descending order
    left_ch = reconstruct(LCT, node_list, mid + 1, high)  # sort in descending order

    root.left = left_ch
    root.right = right_ch
    if left_ch is not None:
        left_ch.parent = root
    if right_ch is not None:
        right_ch.parent = root

    return root


def inorder_traversal(node, l):
    if node is None:
        return
    inorder_traversal(node.left, l)
    l.append(node.val)
    inorder_traversal(node.right, l)
    return


def traversal_top(node):
    cur = node
    while cur.parent is not None or cur.lc_parent is not None:
        if cur.parent is not None:
            cur = cur.parent
        elif cur.lc_parent is not None:
            cur = cur.lc_parent
    return cur


def conn(u, v, LCT):
    return traversal_top(LCT[u]) == traversal_top(LCT[v])


def minimum(node):
    top = traversal_top(node)
    return splayTree.leftmost(top)


def cal_node_edge_size(node):
    node_size = 0
    edge_size = 0
    if node.parent is not None:
        edge_size += sys.getsizeof(node.parent)
    if node.left is not None:
        edge_size += sys.getsizeof(node.left)
    if node.right is not None:
        edge_size += sys.getsizeof(node.right)
    if node.nte is not None:
        edge_size += sys.getsizeof(node.nte)
    if node.lc_parent is not None:
        edge_size += sys.getsizeof(node.lc_parent)

    node_size += sys.getsizeof(node.val)

    return node_size, edge_size


def cal_total_memory_use(LCT):
    print("memory size for LCT dictionary: %d bytes" % sys.getsizeof(LCT))
    space_n = 0
    space_e = 0
    for key, node in LCT.items():
        n_size, e_size = cal_node_edge_size(node)
        space_n += n_size
        space_e += e_size
    space_n += sys.getsizeof(LCT)
    print("total memory size : %d bytes" % (space_n + space_e))
    return space_n, space_e

