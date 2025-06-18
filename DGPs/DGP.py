import matplotlib.pyplot as plt
import networkx as nx

class DGP:
    def __init__(self, effects):
        self.effects = effects
        
        self.G = nx.MultiDiGraph()

        self.pos = {
            'U': (0, 1),
            'X': (0, 0), 
            'Z': (1, 1),
            'W': (1, -1), 
            'Y': (3, 0)
        }

        # Adding edges to the graph
        for edge, effect in self.effects.items():
            if effect != 0:
                self.G.add_edge(*edge)

        # Initialize edge_labels based on existing edges
        self.edge_labels = {edge: effect if effect != 0 else None for edge, effect in self.effects.items() if self.G.has_edge(*edge)}

    def generate_data(self, num_samples):
        raise NotImplementedError("This function should not be called in the superclass.")

    def plot_graph(self, axis=None, node_sz=999):
        if axis is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            plt.figure(figsize=(6, 4))
        else:
            ax = axis
        # Draw nodes and node labels
        nx.draw_networkx_nodes(self.G, self.pos, ax=ax, node_color="#CCCCCC", node_size=node_sz)
        nx.draw_networkx_labels(self.G, self.pos, ax=ax, font_size=10)

        # Draw edges and edge labels
        nx.draw_networkx_edges(self.G, self.pos, ax=ax, edge_color="black", node_size=node_sz, width=1)
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=self.edge_labels, ax=ax)

        if axis is None:
            plt.show()