from Dtree.DTNode import DTNode
from queue import Queue
import sys


# implementation of reroot algorithm in the paper.
def reroot(n_w):

    if n_w.parent is None:
        return n_w

    ch = n_w
    cur = ch.parent
    n_w.parent = None

    while cur is not None:
        g = cur.parent

        cur.parent = ch
        cur.children.remove(ch)
        if ch.children is None:
            ch.children = set()
        ch.children.add(cur)

        ch = cur
        cur = g

    while ch.parent is not None:
        # update the size-attributes of ch and parent of ch.
        ch.size -= ch.parent.size
        ch.parent.size += ch.size
        ch = ch.parent
    return n_w


def link(n_u, r_u, n_v):
    n_v.parent = n_u
    if n_u.children is None:
        n_u.children = set()
    n_u.children.add(n_v)

    c = n_u
    new_root = None
    while c is not None:
        c.size += n_v.size

        if c.size > (r_u.size + n_v.size) // 2 and new_root is None and c.parent is not None:
            new_root = c

        c = c.parent
    if new_root is not None:
        r_u = reroot(new_root)
    return r_u


def insert_nte_simple(n_u, n_v):
    if n_u.nte is None:
        n_u.nte = set()
    if n_v.nte is None:
        n_v.nte = set()
    n_u.nte.add(n_v)
    n_v.nte.add(n_u)


def insert_nte(r, n_u, dist_u, n_v, dist_v):  # inserting a non tree edge

    if n_u.nte is not None and n_v.nte is not None and n_v in n_u.nte and n_u in n_v.nte:
        return

    if abs(dist_u - dist_v) < 2:  # no changes to BFS spanning tree
        if n_u.nte is None:
            n_u.nte = set()
        if n_v.nte is None:
            n_v.nte = set()
        n_u.nte.add(n_v)
        n_v.nte.add(n_u)
        return r
    else:
        if dist_u < dist_v:
            h = n_v
            l = n_u
        else:
            h = n_u
            l = n_v
        delta = abs(dist_u - dist_v) - 2
        c = h
        for i in range(1, delta):
            c = c.parent

        if c.parent.nte is None:
            c.parent.nte = set()
        if c.nte is None:
            c.nte = set()

        c.parent.nte.add(c)
        c.nte.add(c.parent)
        unlink(c)

        return link(l, r, reroot(h))


def insert_edge(u, v, id2node):
    if u not in id2node:
        id2node[u] = DTNode(u)
    if v not in id2node:
        id2node[v] = DTNode(v)

    root_u, distance_u = find_root(id2node[u])
    root_v, distance_v = find_root(id2node[v])

    if root_u.val != root_v.val:
        insert_te(id2node[u], id2node[v], root_u, root_v)
        return
    else:  # a and b are connected
        if not (id2node[u].parent == id2node[v] or id2node[v].parent == id2node[u]) and \
                (id2node[u].nte is None or id2node[v].nte is None
                 or not (id2node[u] in id2node[v].nte and id2node[v] in id2node[u].nte)):
            # (u, v) is a new  non tree edge
            insert_nte(root_u, id2node[u], distance_u, id2node[v], distance_v)
        return


def delete_edge(u, v, id2node):
    if id2node[u].parent == id2node[v] or id2node[v].parent == id2node[u]:
        delete_te(id2node[u], id2node[v])
    elif id2node[u].nte is not None and id2node[v].nte is not None and \
            (id2node[u] in id2node[v].nte or id2node[v] in id2node[u].nte):
        delete_nte(id2node[u], id2node[v])


def insert_te(n_u, n_v, r_u, r_v):

    # T1 includes v, T2 includes u
    if r_v.size < r_u.size:
        return link(n_u, r_u, reroot(n_v))
    else:
        return link(n_v, r_v, reroot(n_u))


def insert_te_simple(n_u, n_v, r_u, r_v):

    # T1 includes v, T2 includes u
    if r_v.size < r_u.size:
        r_v = reroot(n_v)
        if n_u.children is None:
            n_u.children = set()
        n_u.children.add(r_v)
        r_v.parent = n_u
        c = n_u
        while c is not None:
            c.size += r_v.size
            c = c.parent
        return r_u
    else:
        r_u = reroot(n_u)
        if n_v.children is None:
            n_v.children = set()
        n_v.children.add(r_u)
        r_u.parent = n_v
        c = n_v
        while c is not None:
            c.size += r_u.size
            c = c.parent
        return r_v


def delete_nte(node_u, node_v):
    node_u.nte.remove(node_v)
    node_v.nte.remove(node_u)


def unlink(n_v):
    # n_v is a non-root node
    c = n_v
    while c.parent is not None:
        c = c.parent
        c.size -= n_v.size
    n_v.parent.children.remove(n_v)
    n_v.parent = None

    return n_v, c  # return n_v and the root


def delete_te_simple(n_u, n_v):
    # determine parent and child
    if n_u.parent == n_v:
        ch = n_u
    else:
        ch = n_v

    ch, root = unlink(ch)

    if ch.size < root.size:  # BFS is conducted on the smaller tree to find replacement edge.
        r_s = ch
        r_l = root
    else:
        r_s = root
        r_l = ch
    small_size = r_s.size
    n_rs, n_rl, beta = BFS_select_simple(r_s)

    if n_rs is None and n_rl is None:
        return r_s, r_l, small_size, beta
    else:
        n_rs.nte.remove(n_rl)
        n_rl.nte.remove(n_rs)

        return insert_te_simple(n_rs, n_rl, r_s, r_l), None, small_size, beta


def BFS_select_simple(r):
    # traverse the smaller tree to find all neighbors
    q = Queue()
    q.put(r)
    beta = 0
    while not q.empty(): # BFS
        node = q.get()
        if len(node.nte) > 0:
            for nte in node.nte:
                rt, _ = find_root(nte)
                beta += 1
                if rt.val == r.val:  # this non tree edge is included in the smaller tree.
                    continue
                return nte, node, beta
        for c_node in node.children:
            q.put(c_node)

    return None, None, beta


def delete_te(n_u, n_v):
    # determine parent and child
    if n_u.parent == n_v:
        ch = n_u
    else:
        ch = n_v

    ch, root = unlink(ch)

    if ch.size < root.size:  # BFS is conducted on the smaller tree to find replacement edge.
        r_s = ch
        r_l = root
    else:
        r_s = root
        r_l = ch

    small_size = r_s.size
    n_rs, n_rl, new_r, beta = BFS_select(r_s)

    if n_rs is None and n_rl is None:
        if new_r is not None:  # in smaller tree, new_r is the new root, reroot it.
            r_s = reroot(new_r)
        return r_s, r_l, small_size, beta

    else:
        n_rs.nte.remove(n_rl)
        n_rl.nte.remove(n_rs)

        return insert_te(n_rs, n_rl, r_s, r_l), None, small_size, beta


def BFS_select(r):
    # traverse the smaller tree to find all neighbors
    q = Queue()
    q.put(r)
    new_root = None  # new root for smaller tree if new_r is not None
    S = r.size  # size of smaller tree
    minimum_dist = sys.maxsize
    n_rs = None
    n_rl = None
    beta = 0
    while not q.empty():
        new_q = Queue()
        while not q.empty():
            node = q.get()
            if S > node.size > S // 2 and new_root is None:  # new root
                new_root = node
            if node.nte is not None and len(node.nte) > 0:
                for nte in node.nte:
                    rt, dist = find_root(nte)
                    beta += 1
                    if rt.val == r.val:  # this non tree edge is included in the smaller tree.
                        continue

                    if dist < minimum_dist:
                        minimum_dist = dist
                        n_rl = nte  # in larger tree
                        n_rs = node  # in smaller tree
            if node.children is not None:
                for c_node in node.children:
                    q.put(c_node)
        q = new_q

    return n_rs, n_rl, new_root, beta


def query(n_u, n_v):

    d_u = None
    while n_u.parent is not None:
        d_u = n_u
        n_u = n_u.parent

    if d_u is not None and d_u.size > n_u.size // 2:
        n_u = reroot(d_u)

    d_v = None
    while n_v.parent is not None:
        d_v = n_v
        n_v = n_v.parent

    if d_v is not None and d_v.size > n_v.size // 2:
        n_v = reroot(d_v)

    return n_u.val == n_v.val


def query_simple(n_u, n_v):

    while n_u.parent is not None:
        n_u = n_u.parent

    while n_v.parent is not None:
        n_v = n_v.parent
    return n_u.val == n_v.val


def find_root(node):
    dist = 0
    while node.parent is not None:
        node = node.parent
        dist += 1
    return node, dist


def toRoot(node):
    dist = 0
    while node.parent is not None:
        node = node.parent
        dist += 1
    return dist


def cal_node_edge_size(node):
    node_size = 0
    edge_size = 0
    if node.parent is not None:
        edge_size += sys.getsizeof(node.parent)
    if node.children is not None:
        edge_size += sys.getsizeof(node.children)
    if node.nte is not None:
        edge_size += sys.getsizeof(node.nte)
    node_size += sys.getsizeof(node.val)
    node_size += sys.getsizeof(node.size)
    return node_size, edge_size


def cal_total_memory_use(Dtree):
    #print("memory size for Dtree dictionary: %d bytes" % sys.getsizeof(Dtree))
    space_n = 0
    space_e = 0
    for key, node in Dtree.items():
        n_size, e_size = cal_node_edge_size(node)
        space_n += n_size
        space_e += e_size
    space_n += sys.getsizeof(Dtree)
    print("total memory size : %d bytes" % (space_n + space_e))
    return space_n, space_e