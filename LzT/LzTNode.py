from bitarray import bitarray


class LzTNode:

    def __init__(self, v):
        self.parent = None
        self.left = None
        self.right = None
        self.val = v
        self.N = None  # size: number of descending **leaf** nodes
        self.type = None  # 1: tree node, 2: rank node (node of a local rank tree) 3: path (connecting) nodes
        self.level = None
        self.tree_bitmap = bitarray(64)
        self.tree_bitmap.setall(0)
        self.nontree_bitmap = bitarray(64)  # bitmap with 40 digits for non-tree edges
        self.nontree_bitmap.setall(0)

