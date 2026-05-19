import networkx as nx
from funciones_caminantes import step_traditional, step_degree_biased, step_inverse_degree, step_node2vec

G = nx.karate_club_graph()
START = 0
STEPS = 1

def run_walk(name, fn, use_prev=False, p=1.0, q=1.0):
    print(f"\n── {name} ──")
    current, prev = START, None
    path = [current]
    for _ in range(STEPS):
        if use_prev:
            nxt = fn(current, G, prev=prev, p=p, q=q)
        else:
            nxt = fn(current, G)
        if nxt is None:
            print("  Sin vecinos, caminata terminada.")
            break
        prev, current = current, nxt
        path.append(current)
    print(f"  Camino: {path}")

run_walk("Traditional",    step_traditional)
run_walk("Degree biased",  step_degree_biased)
run_walk("Inverse degree", step_inverse_degree)
run_walk("Node2vec p=2 q=0.5", step_node2vec, use_prev=True, p=2.0, q=0.5)
run_walk("Node2vec p=0.5 q=2", step_node2vec, use_prev=True, p=0.5, q=2.0)