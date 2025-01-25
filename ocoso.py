import matplotlib.pyplot as plt
import networkx as nx

# Create a hexagonal lattice graph with specific positions
G = nx.Graph()
positions = {
    r"$R_1$": (1, 0),
    r"$R_2$": (0.5, -0.866),
    r"$R_3$": (-0.5, -0.866),
    r"$R_4$": (-1, 0),
    r"$R_5$": (-0.5, 0.866),
    r"$R_6$": (0.5, 0.866),
    r"$R_0$": (0, 0),
}
edges = [
    (r"$R_0$", r"$R_1$"),
    (r"$R_0$", r"$R_2$"),
    (r"$R_0$", r"$R_3$"),
    (r"$R_0$", r"$R_4$"),
    (r"$R_0$", r"$R_5$"),
    (r"$R_0$", r"$R_6$"),
    (r"$R_1$", r"$R_2$"),
    (r"$R_2$", r"$R_3$"),
    (r"$R_3$", r"$R_4$"),
    (r"$R_4$", r"$R_5$"),
    (r"$R_5$", r"$R_6$"),
    (r"$R_6$", r"$R_1$"),
]
G.add_edges_from(edges)

labels = {node: node for node in G.nodes}
labels = {
    r"$R_0$": r"$R_0$" "\n" r"$(m, n)$",
    r"$R_1$": r"$R_1$" "\n" r"$(m+2, n)$",
    r"$R_2$": r"$R_2$" "\n" r"$(m+1, n-1)$",
    r"$R_3$": r"$R_3$" "\n" r"$(m-1, n-1)$",
    r"$R_4$": r"$R_4$" "\n" r"$(m-2, n)$",
    r"$R_5$": r"$R_5$" "\n" r"$(m-1, n+1)$",
    r"$R_6$": r"$R_6$" "\n" r"$(m+1, n+1)$",
}

fig = plt.figure(figsize=(8, 8))

nx.draw(
    G,
    pos=positions,
    with_labels=True,
    labels=labels,
    node_color="lightgray",
    edge_color="black",
    node_size=900,
    font_size=10,
    font_weight="bold",
    font_family="serif",
)
plt.annotate(
    "",
    xy=(0.25, 0.433),
    xytext=(0, 0),
    arrowprops=dict(arrowstyle="->", color="r", lw=1.5),
)
plt.annotate(
    "", xy=(0.5, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="r", lw=1.5)
)


plt.title("Hexagonal Lattice with Custom Labels and Shapes")
plt.axis("equal")
plt.savefig("Lattice.png")
plt.show()
