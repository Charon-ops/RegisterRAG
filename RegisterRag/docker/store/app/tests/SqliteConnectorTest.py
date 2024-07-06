import sys
import os

sys.path.append(f"{sys.path[0]}/../")

from SqlConnector import SqliteConnector
from DocTree import DocTree

connector = SqliteConnector(
    os.path.join(os.path.dirname(__file__), "..", "data", "test.db")
)

doc_id = connector.add_documents(
    ["This is a test document"], [[0.1, 0.2, 0.3, 0.4, 0.5]], [0], doc_name="test"
)

connector.add_documents(
    ["This is another test document"], [[0.1, 0.2, 0.3, 0.4, 0.5]], [1], doc_name="test"
)

connector.add_documents(
    ["This is the third test document"],
    [[0.1, 0.2, 0.3, 0.4, 0.5]],
    [2],
    doc_name="test",
)

connector.delete_by_id(1)

avl1 = DocTree(doc_id)
avl2 = connector.read_tree(doc_id)

if avl1 is avl2:
    print("The trees are the same")
