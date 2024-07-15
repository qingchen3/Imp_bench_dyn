class NTE_Node:

    def __init__(self, act_occ, priority):
        self.parent = None
        self.left = None
        self.right = None
        self.act_occ = act_occ
        self.weight = 1
        self.priority = priority


