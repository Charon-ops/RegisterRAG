class DocNode:
    def __init__(self, id: int, index: int) -> None:
        self.id = id
        self.index = index
        self.height = 1
        self.parent = None
        self.left = None
        self.right = None
