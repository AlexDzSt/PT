"""
test_funciones.py
──────────────────
Tests con pytest para las funciones de probabilidad de salto y de
caminata definidas en caminantes.py (el módulo que efectivamente usan
grafica_cobertura.py y vis_caminantes.py).

Ejecutar con:
    pytest test_funciones.py -v
"""

import math

import networkx as nx
import pytest

from caminantes import (
    jump_prob_traditional,
    jump_prob_degree_biased,
    jump_prob_inverse_degree,
    jump_prob_node2vec,
    walk,
    step_probabilities,
)

ALL_STRATEGIES = [
    jump_prob_traditional,
    jump_prob_degree_biased,
    jump_prob_inverse_degree,
    jump_prob_node2vec,
]


@pytest.fixture
def G():
    return nx.karate_club_graph()


# ─────────────────────────────────────────────────────────────────────────
#  Las probabilidades de salto deben ser una distribución válida
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("jump_fn", ALL_STRATEGIES)
def test_probabilidades_suman_uno(G, jump_fn):
    current = 0
    probs = jump_fn(G, source=current, current=current)
    assert probs, "se esperaban probabilidades para un nodo con vecinos"
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-9)


@pytest.mark.parametrize("jump_fn", ALL_STRATEGIES)
def test_probabilidades_no_negativas(G, jump_fn):
    current = 0
    probs = jump_fn(G, source=current, current=current)
    assert all(p >= 0 for p in probs.values())


@pytest.mark.parametrize("jump_fn", ALL_STRATEGIES)
def test_solo_vecinos_reales(G, jump_fn):
    current = 0
    probs = jump_fn(G, source=current, current=current)
    assert set(probs.keys()) == set(G.neighbors(current))


@pytest.mark.parametrize("jump_fn", ALL_STRATEGIES)
def test_nodo_aislado_devuelve_vacio(jump_fn):
    G = nx.Graph()
    G.add_node(0)  # sin vecinos
    probs = jump_fn(G, source=0, current=0)
    assert probs == {}


# ─────────────────────────────────────────────────────────────────────────
#  Casos específicos de cada estrategia
# ─────────────────────────────────────────────────────────────────────────

def test_traditional_es_uniforme(G):
    current = 0
    probs = jump_prob_traditional(G, source=current, current=current)
    n = len(probs)
    assert all(math.isclose(p, 1 / n) for p in probs.values())


def test_degree_biased_favorece_alto_grado():
    G = nx.star_graph(5)
    probs = jump_prob_degree_biased(G, source=1, current=1)
    assert probs == {0: 1.0}


def test_inverse_degree_favorece_bajo_grado():
    G = nx.star_graph(5)
    probs = jump_prob_inverse_degree(G, source=0, current=0)
    n = len(probs)
    assert all(math.isclose(p, 1 / n) for p in probs.values())


def test_node2vec_p_q_uno_equivale_a_uniforme(G):
    # Con p = q = 1, node2vec debe comportarse como el caminante tradicional
    current, source = 5, 0
    probs_n2v = jump_prob_node2vec(G, source=source, current=current, p=1.0, q=1.0)
    probs_trad = jump_prob_traditional(G, source=source, current=current)
    assert probs_n2v.keys() == probs_trad.keys()
    for node in probs_n2v:
        assert math.isclose(probs_n2v[node], probs_trad[node])


def test_node2vec_p_grande_desincentiva_volver(G):
    # p grande → volver al nodo origen debe ser mucho menos probable que con p pequeño
    current, source = 5, 0
    if source not in G.neighbors(current):
        pytest.skip("el nodo fuente no es vecino directo en este caso")
    probs_p_alto = jump_prob_node2vec(G, source=source, current=current, p=100.0, q=1.0)
    probs_p_bajo = jump_prob_node2vec(G, source=source, current=current, p=0.01, q=1.0)
    assert probs_p_alto[source] < probs_p_bajo[source]


# ─────────────────────────────────────────────────────────────────────────
#  walk() y step_probabilities()
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("jump_fn", ALL_STRATEGIES)
def test_walk_respeta_max_steps(G, jump_fn):
    path = walk(G, start=0, jump_fn=jump_fn, stop_prob=0.0, max_steps=50)
    # +1 porque el nodo inicial también cuenta
    assert len(path) <= 51


@pytest.mark.parametrize("jump_fn", ALL_STRATEGIES)
def test_walk_empieza_en_start(G, jump_fn):
    path = walk(G, start=3, jump_fn=jump_fn, stop_prob=0.0, max_steps=10)
    assert path[0] == 3


def test_walk_pasos_consecutivos_son_vecinos(G):
    path = walk(G, start=0, jump_fn=jump_prob_traditional, stop_prob=0.0, max_steps=30)
    for a, b in zip(path, path[1:]):
        assert b in G.neighbors(a)


def test_step_probabilities_nodo_actual(G):
    path = [0, 2, 5]
    probs = step_probabilities(G, path, jump_prob_traditional)
    esperado = jump_prob_traditional(G, source=2, current=5)
    assert probs == esperado


def test_step_probabilities_camino_vacio(G):
    assert step_probabilities(G, [], jump_prob_traditional) == {}
