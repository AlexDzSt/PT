# Random Walk Lab

Este Proyecto Terminal está asociado el Proyecto de Ciencia Básica y de Frontera  titulado "Modelos de reconexión para la autoorganización de redes complejas de gran escala" (CBF-2025-G-1812)

Laboratorio en Python para simular, visualizar y comparar **caminatas aleatorias (random walks)** sobre grafos, implementando cuatro estrategias de salto distintas — incluyendo una variante inspirada en **node2vec**. El proyecto incluye una aplicación de escritorio interactiva (Tkinter + Matplotlib), utilidades de análisis, scripts de generación de gráficas y una suite de pruebas con `pytest`.

## Contenido del repositorio

| Archivo / carpeta | Descripción |
|---|---|
| `caminantes.py` | Núcleo del proyecto: funciones de probabilidad de salto (`jump_prob_*`), el motor de la caminata (`walk`) y el cálculo de probabilidades para un paso dado (`step_probabilities`). Es el módulo que consumen tanto el simulador visual como el script de gráficas de cobertura. |
| `funciones_caminantes.py` | Implementación alternativa/independiente de las mismas estrategias, expresada como funciones `step_*` que devuelven directamente el **siguiente nodo** (en vez de una distribución de probabilidad). Útil como referencia didáctica o para integraciones más simples. |
| `datos_caminata.py` | Funciones puras (sin dependencias de visualización) para calcular métricas sobre una caminata ya ejecutada: longitud, nodos únicos, cobertura y la curva de cobertura paso a paso. |
| `vis_caminantes.py` | **Simulador interactivo** con interfaz gráfica (Tkinter). Permite elegir grafo, estrategia y parámetros, ejecutar caminatas paso a paso o completas, visualizar probabilidades de salto y comparar las 4 estrategias simultáneamente. Ver la sección [Simulador `vis_caminantes.py`](#simulador-vis_caminantespy) más abajo. |
| `grafica_cobertura.py` | Script de análisis estadístico: ejecuta muchas caminatas independientes por estrategia sobre un grafo (por defecto, el club de karate de Zachary) y grafica la **cobertura media ± desviación estándar** en función de la longitud relativa de la caminata. |
| `test_funciones.py` | Suite de pruebas (`pytest`) que valida las funciones de `caminantes.py`: que las probabilidades sumen 1, que sean no negativas, que solo apunten a vecinos reales, etc. |
| `resultadosExperimentos/` | Resultados crudos de experimentos a mayor escala (reglas `R1`, `R2`, `R3`) y el script `crearGraficas.py`, que procesa esos datos para producir gráficas de diámetro, distribución de grados, mapas de calor y clustering/modularidad. |
| `requirements.txt` | Dependencias del proyecto. |

## Estrategias de caminata implementadas

Todas se basan en la probabilidad de moverse desde el nodo actual `v` (con vecino candidato `u`, o `x` en el caso de node2vec):

1. **Traditional RW (caminata tradicional)**
   `P(v→u) = 1 / deg(v)` — distribución uniforme entre todos los vecinos.
2. **Degree-Biased RW (sesgada por grado)**
   `P(v→u) = deg(u) / Σ deg(w)` para `w` vecino de `v` — favorece saltar hacia nodos de alto grado (*hubs*).
3. **Inverse-Degree RW (grado inverso)**
   `P(v→u) = (1/deg(u)) / Σ (1/deg(w))` — favorece nodos de bajo grado (periferia de la red).
4. **Node2Vec RW**
   Caminata de **Markov de segundo orden**: la probabilidad de ir del nodo actual `t` (anterior) a través de `v` hacia un candidato `x` depende de un factor `α`:
   - `α = 1/p` si `x` es el nodo previo `t` (controla la probabilidad de "volver atrás").
   - `α = 1` si `x` es vecino común de `t` y `v` (vecindad local/triángulo).
   - `α = 1/q` en cualquier otro caso (exploración hacia nodos más lejanos).

   `P(v→x | t) = α(t,x) / Z`, con `Z` la suma normalizadora. `p` controla el retorno y `q` controla el balance exploración local (BFS) vs. exploración distante (DFS), tal como en el algoritmo node2vec original.

## Instalación

```bash
git clone https://github.com/AlexDzSt/random-walk-lab.git
cd random-walk-lab
python -m venv .venv
source .venv/bin/activate      # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Dependencias principales:** `networkx`, `matplotlib`, `numpy`, `pytest`. La interfaz gráfica usa `tkinter`, que viene incluido en la mayoría de las distribuciones estándar de Python (en Linux puede requerir instalar el paquete del sistema `python3-tk`).

## Uso rápido

```bash
# Abrir el simulador interactivo
python vis_caminantes.py

# Generar la gráfica de cobertura vs. longitud (comparando las 4 estrategias)
python grafica_cobertura.py

# Ejecutar la suite de pruebas
pytest test_funciones.py -v
```

---

## Simulador `vis_caminantes.py`

Es una aplicación de escritorio construida con **Tkinter** (interfaz) y **Matplotlib** (gráficas embebidas, vía `FigureCanvasTkAgg`), apoyada en **NetworkX** para el manejo de grafos. Su propósito es permitir explorar de forma visual e interactiva cómo se comporta cada estrategia de caminata aleatoria sobre distintas topologías de red.

### Estructura general de la ventana

Al iniciar (`App.__init__` → `_load_graph("karate")`), la aplicación abre una ventana dividida en dos zonas:

- **Panel izquierdo (controles)**, scrollable, con:
  - Selector de **grafo** (`GRAFO`).
  - Selector de **estrategia** de caminata (`ESTRATEGIA`).
  - Campos de **parámetros**: nodo de inicio, probabilidad de parada, `p` y `q` (solo activos si la estrategia es Node2Vec), y número máximo de pasos.
  - Casilla para mostrar u ocultar las probabilidades sobre las aristas del grafo.
  - Botones de acción: **Ejecutar caminata**, **Paso a paso**, **Comparar todas**, **Limpiar**.
  - Panel de **estadísticas** en vivo del grafo y de la caminata activa.
  - Panel con la **fórmula matemática** de la estrategia seleccionada.
- **Panel derecho**, organizado en cuatro pestañas (`ttk.Notebook`):
  1. **Red** — dibujo del grafo con la caminata resaltada.
  2. **Probabilidades** — gráfica de barras con `P(salto)` desde el nodo actual.
  3. **Comparar Estrategias** — cuadrícula 2×4 que ejecuta y compara las 4 estrategias a la vez.
  4. **Log** — bitácora textual de todas las acciones y caminatas realizadas.

### Grafos disponibles

El diccionario `App.GRAPHS` define siete topologías seleccionables, generadas con NetworkX (o a mano, en el caso del grafo de ejemplo):

- **Zachary's Karate Club** (`nx.karate_club_graph`) — grafo por defecto al abrir la app.
- **Petersen Graph** (`nx.petersen_graph`).
- **Barbell (5-5)** (`nx.barbell_graph(5, 5)`).
- **Erdős–Rényi (30, 0.15)** — grafo aleatorio (`nx.erdos_renyi_graph`, semilla fija 42).
- **Barabási–Albert (30, 2)** — grafo libre de escala (`nx.barabasi_albert_graph`, semilla fija 42).
- **Grid 5×5** (`nx.grid_2d_graph`, con nodos re-etiquetados a enteros).
- **Grafo Ejemplo** — un grafo pequeño hecho a mano en `build_custom_graph()` (4 nodos, 4 aristas), pensado como plantilla para que el usuario defina su propia topología de prueba.

Al cambiar de grafo se recalcula el layout con `nx.spring_layout` (semilla 42, para que la disposición sea reproducible) y se reinicia el estado de la caminata activa.

### Flujo de ejecución de una caminata

El simulador mantiene dos variables de estado clave:

- `self.current_path`: la secuencia de nodos visitados por la caminata activa.
- `self.current_probs`: las probabilidades de salto **desde el nodo actual** (`path[-1]`) hacia sus vecinos. Esta invariante se mantiene actualizada tras cada acción, de modo que la pestaña "Probabilidades" siempre responde a la pregunta "¿a dónde puedo ir desde aquí?".

Las tres formas de generar/avanzar una caminata son:

1. **▶ Ejecutar caminata** (`_run_walk`): valida los parámetros, ejecuta la función `walk(...)` de `caminantes.py` de principio a fin (deteniéndose por probabilidad de parada o al alcanzar el máximo de pasos) y actualiza de golpe todas las vistas: red, probabilidades, estadísticas y log.

2. **⊕ Paso a paso** (`_step_walk`): avanza un único paso sobre la caminata activa (la inicia si no existe aún). Calcula las probabilidades desde el nodo actual usando la función de la estrategia elegida, elige el siguiente nodo con `random.choices` ponderado por esas probabilidades, y refresca las vistas. Permite observar de forma incremental cómo decide cada estrategia en cada punto de la red.

3. **⊞ Comparar todas** (`_run_compare`): ejecuta las **4 estrategias** con exactamente los mismos parámetros (nodo de inicio, probabilidad de parada, `p`, `q`, máximo de pasos) y construye una figura de 2×4 subgráficas:
   - **Fila superior**: el grafo con la caminata de cada estrategia resaltada (nodo inicial en amarillo, nodo final en rojo, intensidad de color proporcional a la frecuencia de visitas).
   - **Fila inferior**: barras con la probabilidad de salto desde el nodo final de cada caminata (para Node2Vec se indica también el nodo "fuente" de segundo orden).

   Además llena una tabla de métricas (longitud, nodos únicos visitados, cobertura porcentual y nodo final) para las cuatro estrategias, y salta automáticamente a la pestaña de comparación.

El botón **✕ Limpiar** reinicia el estado de la caminata (sin cambiar el grafo ni la estrategia) y vuelve a dibujar las vistas vacías.

### Dibujo de la red (`draw_network`)

Esta función centraliza el renderizado del grafo en la pestaña "Red":

- Colorea el **nodo de inicio** en amarillo (`#F9A825`) y el **nodo actual** (último de la caminata) en rojo (`#EF5350`), sobre-escribiendo cualquier otro color.
- Los **nodos visitados** se pintan con el color asociado a la estrategia activa (definido en `STRATEGY_COLORS`), con intensidad y tamaño proporcionales a cuántas veces fueron visitados (usando `Counter(path)`).
- Las **aristas recorridas** por la caminata se resaltan con el color de la estrategia y mayor grosor; el resto queda en gris tenue.
- Si la casilla "Mostrar probs en red" está activa, se superponen etiquetas numéricas de probabilidad sobre las aristas que salen del nodo actual hacia sus vecinos.
- Se agrega una leyenda indicando el significado de cada color.

### Pestaña de probabilidades

Muestra un gráfico de barras (`_refresh_probs`) con la probabilidad de salto desde el nodo actual hacia cada uno de sus vecinos, usando el color de la estrategia activa y anotando el valor numérico sobre cada barra. Si no hay caminata activa, se muestra un mensaje de marcador de posición.

### Panel de estadísticas y fórmulas

- **Estadísticas** (`_update_stats`): número de nodos y aristas del grafo, densidad, grado medio, longitud de la caminata activa, nodos únicos visitados, porcentaje de cobertura y nodo actual.
- **Fórmula activa** (`_update_formula`): muestra en texto la expresión matemática de la estrategia seleccionada (las mismas descritas en la sección [Estrategias de caminata implementadas](#estrategias-de-caminata-implementadas)), actualizándose automáticamente al cambiar de estrategia.

### Log

La pestaña "Log" acumula un historial con formato y colores por tipo de evento (carga de grafo, inicio/fin de caminata, cada paso individual, resultados de comparación), útil para revisar en detalle lo ocurrido durante la sesión de exploración.

### Validación de parámetros

Antes de cualquier ejecución, `_get_params()` valida que:
- el nodo de inicio exista en el grafo actual;
- la probabilidad de parada esté estrictamente en `(0, 1)`;
- `p` y `q` sean números positivos (relevantes solo para Node2Vec);
- el máximo de pasos sea un entero ≥ 1.

Si hay errores, se muestra un cuadro de diálogo (`messagebox.showerror`) detallando todos los problemas encontrados y no se ejecuta la caminata.

### Relación con `caminantes.py`

`vis_caminantes.py` no reimplementa la lógica de las caminatas: importa directamente `jump_prob_traditional`, `jump_prob_degree_biased`, `jump_prob_inverse_degree`, `jump_prob_node2vec`, `walk` y `step_probabilities` desde `caminantes.py`. Esto mantiene una única fuente de verdad para las estrategias, compartida también por `grafica_cobertura.py` y validada por `test_funciones.py`.