class HK_Node:

    def __init__(self, val, priority):
        self.parent = None
        self.left = None
        self.right = None
        self.val = val
        self.active = False
        self.nte = None
        self.nte_tree = None  # randomize search tree that stores nontree neighbors
        self.weight = 0
        self.priority = priority