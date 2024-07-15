class STNode:

    def __init__(self, v):
        self.parent = None
        self.children = None
        self.val = v
        self.N = 1
        self.level = 0
        if self.val is None:
            self.N = 0
