from bitarray import bitarray


class LTNode:

    def __init__(self, v):
        self.parent = None
        self.left = None
        self.right = None
        self.val = v
        self.N = None
        self.type = None  # 1: tree node, 2: rank node (node of a local rank tree) 3: path (connecting) nodes
        self.level = None
        # self.rank = None
        self.bitmap = bitarray(64)  # bitmap with 40 digits
        self.bitmap.setall(0)
