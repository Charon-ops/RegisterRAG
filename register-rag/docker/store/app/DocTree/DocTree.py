import threading

from .DocNode import DocNode


class DocTree:
    _instance = {}
    _lock = threading.Lock()

    def __new__(cls, doc_id: int):
        with cls._lock:
            if doc_id not in cls._instance:
                cls._instance[doc_id] = super().__new__(cls)
            return cls._instance[doc_id]

    def __init__(self, doc_id: int) -> None:
        self.lock = threading.Lock()
        self.doc_id = doc_id
        self.root = None

    def insert(self, id: int, index: int) -> None:
        assert id is not None and index is not None, "id and index should not be None"
        with self.lock:
            self.root = self._insert(id, index, self.root)

    def delete(self, index: int) -> None:
        if index is None:
            return
        with self.lock:
            self.root = self._delete(index, self.root)

    def _insert(self, id: int, index: int, node: DocNode = None) -> DocNode:
        if node is None:
            node = DocNode(id, index)
            return node

        if index < node.index:
            node.left = self._insert(id, index, node.left)
            node.left.parent = node
        else:
            node.right = self._insert(id, index, node.right)
            node.right.parent = node

        node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))

        balance = self.get_balance(node)

        # 左左
        if balance > 1 and index < node.left.index:
            return self.rotate_right(node)

        # 右右
        if balance < -1 and index > node.right.index:
            return self.rotate_left(node)

        # 左右
        if balance > 1 and index > node.left.index:
            node.left = self.rotate_left(node.left)
            return self.rotate_right(node)

        # 右左
        if balance < -1 and id < node.right.index:
            node.right = self.rotate_right(node.right)
            return self.rotate_left(node)

        return node

    def _delete(self, index: int, node: DocNode = None) -> DocNode:
        if not node:
            return node

        if index < node.index:
            node.left = self._delete(index, node.left)
        elif index > node.index:
            node.right = self._delete(index, node.right)
        else:
            if node.left is None:
                temp = node.right
                node = None
                return temp
            elif node.right is None:
                temp = node.left
                node = None
                return temp
            temp = self.get_min(node.right)
            node.id = temp.id
            node.index = temp.index
            node.right = self._delete(temp.index, node.right)
            if not node.right:
                node.right.parent = node
            node.height = 1 + max(
                self.get_height(node.left), self.get_height(node.right)
            )
            balance = self.get_balance(node)
            if balance > 1 and self.get_balance(node.left) >= 0:
                return self.rotate_right(node)
            if balance < -1 and self.get_balance(node.right) <= 0:
                return self.rotate_left(node)
            if balance > 1 and self.get_balance(node.left) < 0:
                node.left = self.rotate_left(node.left)
                return self.rotate_right(node)
            if balance < -1 and self.get_balance(node.right) > 0:
                node.right = self.rotate_right(node.right)
                return self.rotate_left(node)
        return node

    def get_height(self, node: DocNode) -> int:
        if not node:
            return 0
        return node.height

    def get_balance(self, node: DocNode) -> int:
        if not node:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)

    def rotate_left(self, z: DocNode) -> DocNode:
        y = z.right
        T2 = y.left

        y.left = z
        z.right = T2

        y.parent = z.parent
        z.parent = y
        if T2:
            T2.parent = z

        if y.parent:
            if y.parent.left == z:
                y.parent.left = y
            elif y.parent.right == z:
                y.parent.right = y

        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))

        return y

    def rotate_right(self, z: DocNode) -> DocNode:
        y = z.left
        T3 = y.right

        y.right = z
        z.left = T3

        y.parent = z.parent
        z.parent = y
        if T3:
            T3.parent = z

        if y.parent:
            if y.parent.left == z:
                y.parent.left = y
            elif y.parent.right == z:
                y.parent.right = y

        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))

        return y

    def get_predecessor(self, node: DocNode) -> DocNode:
        # 如果有左子树，前驱在左子树的最右边
        if node.left:
            return self.get_max(node.left)
        # 否则向上找到第一个是其右孩子的祖先
        current = node
        while current.parent and current.parent.left == current:
            current = current.parent
        return current.parent

    def get_successor(self, node: DocNode) -> DocNode:
        # 如果有右子树，后继在右子树的最左边
        if node.right:
            return self.get_min(node.right)
        # 否则向上找到第一个是其左孩子的祖先
        current = node
        while current.parent and current.parent.right == current:
            current = current.parent
        return current.parent

    def get_min(self, node: DocNode) -> DocNode:
        current = node
        while current.left:
            current = current.left
        return current

    def get_max(self, node: DocNode) -> DocNode:
        current = node
        while current.right:
            current = current.right
        return current

    def find_by_index(self, index: int, node: DocNode = None) -> DocNode:
        if not self.root:
            return None
        if node is None:
            node = self.root
        if index == node.index:
            return node
        if index < node.index:
            return self.find_by_index(index, node.left)
        else:
            return self.find_by_index(index, node.right)
