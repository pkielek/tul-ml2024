from graphviz import Digraph

def plot_tree(root, name):
    def add_nodes_edges(node, graph, node_id=0):
        """
        Recursive helper function to add nodes and edges to the graph.
        """
        if node is None:
            return node_id
        
        # Add the current node
        graph.node(str(node_id), label=str(node.condition))
        current_id = node_id
        node_id += 1
        
        # Add left child
        if node.left is not None:
            graph.edge(str(current_id), str(node_id))
            node_id = add_nodes_edges(node.left, graph, node_id)
        
        # Add right child
        if node.right is not None:
            graph.edge(str(current_id), str(node_id))
            node_id = add_nodes_edges(node.right, graph, node_id)
        
        return node_id

    # Create a Digraph object
    dot = Digraph(filename=name, directory="", format="png")
    dot.attr(rankdir="TB")  # Top to Bottom

    # Add nodes and edges starting from the root
    add_nodes_edges(root, dot)

    # Render and display the tree
    dot.render(f"./trees/visualized_trees/{name}", view=False)
