class HDT_Node:

    def __init__(self, val, priority):
        self.parent = None
        self.left = None
        self.right = None
        self.val = val
        self.active = False
        self.nte = None
        self.size = 0
        self.priority = priority