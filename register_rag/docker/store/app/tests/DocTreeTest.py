import sys

sys.path.append(f"{sys.path[0]}/../")

from DocTree import DocTree

avl_tree = DocTree(1)
avl_tree.insert(1, 7)
avl_tree.insert(2, 5)
avl_tree.insert(3, 8)
avl_tree.insert(4, 6)
avl_tree.insert(5, 4)
avl_tree.insert(6, 3)

node = avl_tree.find_by_index(5)
print(node.id)
print(avl_tree.get_successor(node).index)
if avl_tree.get_predecessor(node):
    print(avl_tree.get_predecessor(node).index)
else:
    print(f"Node {node.id} has no predecessor")
