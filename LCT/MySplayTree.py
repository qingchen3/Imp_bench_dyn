import sys
from timeit import default_timer as timer


class MyNode:
    def __init__(self, v):
        self.val = v
        self.parent = None
        self.left = None
        self.right = None
        self.nte = None
        self.lc_parent = None  # pointing to the parent node in link cut tree (not in the splay tree!)


def right_rotate(x):  # zig rotation
    y = x.left
    y.parent = x.parent
    if x.parent is None:
        pass
    elif x == x.parent.left:
        x.parent.left = y
    else:
        x.parent.right = y

    x.left = y.right
    if y.right is not None:
        y.right.parent = x

    x.parent = y
    y.right = x

    # update pointers of link-cut tree
    if x.lc_parent is not None:
        y.lc_parent = x.lc_parent
        x.lc_parent = None

    return


def left_rotate(x):  # zag roration
    y = x.right

    y.parent = x.parent
    if x.parent is None:
        pass
    elif x == x.parent.left:
        x.parent.left = y
    else:
        x.parent.right = y

    x.right = y.left
    if y.left is not None:
        y.left.parent = x

    y.left = x
    x.parent = y

    # update pointers of link-cut tree
    if x.lc_parent is not None:
        y.lc_parent = x.lc_parent
        x.lc_parent = None

    return


def splay(x):
    while x.parent is not None:
        if x.parent.parent is None and x.parent.left == x:
            right_rotate(x.parent)  # zig rotation

        elif x.parent.parent is None and x.parent.right == x:
            left_rotate(x.parent)  # zag rotation

        elif x.parent.parent is not None:
            if x.parent == x.parent.parent.left and x == x.parent.left:  # zig-zig rotation
                right_rotate(x.parent.parent)
                right_rotate(x.parent)

            elif x.parent == x.parent.parent.right and x == x.parent.right:  # zag-zag rotation
                left_rotate(x.parent.parent)
                left_rotate(x.parent)

            elif x.parent == x.parent.parent.left and x == x.parent.right:  # zig-zag rotation
                left_rotate(x.parent)
                right_rotate(x.parent)

            elif x.parent == x.parent.parent.right and x == x.parent.left:  # zag-zig rotation
                right_rotate(x.parent)
                left_rotate(x.parent)

        else:
            raise ValueError("invalid tree structure")
    return x


def insert(x, root):  # inserting nodes
    cur = root
    while cur is not None:
        if cur.val < x.val:
            if cur.right is None:
                cur.right = x
                x.parent = cur
                break
            else:
                cur = cur.right
        else:
            if cur.left is None:
                cur.left = x
                x.parent = cur
                break
            else:
                cur = cur.left
    if root is None:
        root = x
    else:
        root = splay(x)
    return root


def insert1(x, root):  # inserting nodes
    cur = root
    par = None
    while cur is not None:
        par = cur
        if cur.val < x.val:
            cur = cur.right
        else:
            cur = cur.left
    if root is None:
        root = x
    else:
        if x.val > par.val:
            par.right = x
        else:
            par.left = x
        x.parent = par
        root = splay(x)
    return root


def rightmost(root):
    cur = root
    while cur.right is not None:
        cur = cur.right
    return cur


def leftmost(root):
    cur = root
    while cur.left is not None:
        cur = cur.left
    return cur


# merge two splay trees, assuming nodes in the splay tree rooted at root1 are "less" than
# all nodes in splay tree rooted in root2.
# Neither of root1 and root2 is None
def merge(root1, root2):
    r = rightmost(root1)
    r = splay(r)
    r.right = root2
    root2.parent = r
    return r


def delete():
    return


def split():
    return


def successor(x):
    if x.right is not None:
        return leftmost(x.right)

    y = x.parent
    while y is not None and x == y.right:
        x = y
        y = y.parent
    return y


def predecessor(x):
    if x.left is not None:
        return rightmost(x.left)

    y = x.parent
    while y is not None and x == y.left:
        x = y
        y = y.parent
    return y


def print_helper(currPtr, indent, last):
    if currPtr is not None and currPtr.parent is None:
        print("Root: %d" % currPtr.val)
    if currPtr != None:
        sys.stdout.write(indent)
        if last:
            sys.stdout.write("R----")
            indent += "     "
        else:
            sys.stdout.write("L----")
            indent += "|    "

        print(currPtr.val)

        print_helper(currPtr.left, indent, False)
        print_helper(currPtr.right, indent, True)
    return


if __name__ == '__main__':
    print("testing ...")