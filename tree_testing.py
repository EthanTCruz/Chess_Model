class Node:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)
# create the nodes
a = Node(1)
b = Node(2)
c = Node(3)
d = Node(4)
e = Node(5)
f = Node(6)
g = Node(7)
h = Node(8)

# build the tree structure
a.add_child(b)
a.add_child(c)
a.add_child(d)
a.add_child(e)
b.add_child(f)
b.add_child(g)
e.add_child(h)

def dfs(node, target_value):
    if node.value == target_value:
        return node
    for child in node.children:
        result = dfs(child, target_value)
        if result is not None:
            return result
    return None

result = dfs(a, 6)
if result is not None:
    print("Found node with value:", result.value)
else:
    print("Node not found")
