import sys


def cal_nte_tree_node_size(node):
    tree_size = 0
    if node.parent is not None:
        tree_size += sys.getsizeof(node.parent)
    if node.left is not None:
        tree_size += sys.getsizeof(node.left)
    if node.right is not None:
        tree_size += sys.getsizeof(node.right)
    if node.act_occ is not None:
        tree_size += sys.getsizeof(node.act_occ)
    tree_size += sys.getsizeof(node.weight)
    tree_size += sys.getsizeof(node.priority)
    return tree_size