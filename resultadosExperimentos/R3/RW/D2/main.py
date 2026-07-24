
import math
import sys
import networkx as nx
from simulation import Simulation
from complexNetwork import ComplexNetwork
from event import Event
from paquete import Paquete
import config

class Main:#hereda de la clase object, hay que recordar que en Python todo es un objeto

	def __init__(self):
		#--------ANILLO------------------
		self.__nodes=config.NODOS_ANILLO # Número de nodos del anillo
		self.__diametro=20               # Se asume que el radio del anillo es 10 unidades
		#--------MALLA-------------------
		self.__rows=config.ROWS          # Filas de la malla
		self.__columns=config.COLUMNS    # Columnas de la malla
		self.__diagonal=self.diag()      # Longitud de la diagonal principal de la malla
		#--------------------------------
		self.__ciclos=config.CICLOS      # Número de ciclos de simulación
		#self.__tamEnlace=self.__diagonal # Longitud máxima del enlace dinámico, se puede ajustar con self.__diagonal para la malla y self.__diametro para el anillo
		self.__grafo=config.RED          # Topología sobre la que se desarrollará la simulación
		self.__graph = None              # Grafo NetworkX sobre el que se desarrollará la simulación
		self.__coordinador=1             # Identificador del nodo coordinador de la simulación
		self.__nodosTotales=-1           # Número de nodos total del grafo
	
	#---MALLA----------------------------

	def createGrid(self):
		self.__graph = nx.grid_2d_graph(self.__rows,self.__columns,periodic=False)
		self.__graph = nx.convert_node_labels_to_integers(self.__graph,first_label=1,ordering="sorted")
		self.__tamEnlace=self.__diagonal / config.LONG_ENLACE
		nx.write_adjlist(self.__graph,"graph.adjlist")

	def diag(self):
		n1=[0,0]
		n2=[self.__rows-1,self.__columns-1]
		d=math.sqrt(math.pow(n2[0]-n1[0],2)+math.pow(n2[1]-n1[1],2))
		return d

	#---ANILLO----------------------------

	def createRing(self):
		self.__graph = nx.circulant_graph(self.__nodes, [1])
		self.__graph = nx.convert_node_labels_to_integers(self.__graph,first_label=1,ordering="sorted")
		self.__tamEnlace=self.__diametro / config.LONG_ENLACE
		nx.write_adjlist(self.__graph,"graph.adjlist")

	#---GETTERS y SETTERS------------------

	@property
	def nodes(self):
		return self.__nodes

	@property
	def rows(self): 
		return self.__rows

	@property
	def columns(self):
		return self.__columns

	@property
	def ciclos(self):
		return self.__ciclos

	@property
	def tamEnlace(self):
		return self.__tamEnlace

	@property
	def grafo(self):
		return self.__grafo

	@grafo.setter
	def grafo(self,value):
		self.__grafo=value

	@property
	def graph(self):
		return self.__graph

	@property
	def coordinador(self):
		return self.__coordinador

	@property
	def nodosTotales(self):
		return self.__nodosTotales

	@nodosTotales.setter
	def nodosTotales(self,value):
		self.__nodosTotales=value

#---MAIN()-----------------------
if __name__ == "__main__":
	#Se crea una instancia de la clase Main
	main=Main()
	#Se selecciona el grafo con el que se trabajará en la simulación
	#1.- Malla
	#3.- Anillo
	if(main.grafo==1):
		main.createGrid()
		main.nodosTotales=main.rows*main.columns
	elif(main.grafo==3): 
		main.createRing()
		main.nodosTotales=main.nodes
	#se muestra el grafo en una imágen:
	#plt.cla()
	#nx.draw_networkx(main.graph)
	#plt.show()
	#Se construye una instancia de la clase Simulation
	experiment = Simulation(sys.maxsize,main)
	experiment.readAdjacencyListNetworkX()
	#Se asocia un pareja proceso/modelo con cada nodo de la gráfica
	for i in range(1,len(experiment.graph)+1):
		#parámetros del constructor de la clase ComplexNetwork
		#el primer parámetro es una referencia al objeto principal de la simulación
		#el segundo parámetro es el número de enlaces dinámicos del nodo
		#el tercer parámetro es el número de conexiones soportadas por el nodo 
		#el cuarto parámetro es el número de paquetes exploradores que enviará el nodo en fase de exploración
		#el quinto parámetro indica el algoritmo de encaminamiento que el nodo ejecutará en la fase de exploración
		#opciones de algoritmo de encaminamiento: "COMPASS-ROUTING", "RANDOM-WALK", "SHORTEST-PATH"
		#el sexto parámetro indica el número de la regla de recableado que el nodo ejecutará en la fase de negociación
		#el septimo parámetro es el número de ejecucipon para ubicar archivo de log donde se registrarán los eventos del nodo
  		#opciones de reglas de recableado: R1=>1, R2=>2, R3=>3
		m = ComplexNetwork(main,config.ENLACES_DINAMICOS,main.nodosTotales/config.DIV_CONEXIONES,config.EXPLORADORES,config.ROUTING,config.REGLA)#,sys.argv[1])
		experiment.setModel(m, i)
	#inserta un evento semilla en la agenda y arranca
	#para comenzar la simulacion el coordinador arranca un PIF en donde avisará a todos que hagan su fase de EXPOLRACIÓN
	new_package=Paquete(0,0)
	new_package.diametro=nx.diameter(main.graph)
	seed=Event("PIF-EXPLORACION", 0.0, main.coordinador,main.coordinador,new_package)
	experiment.init(seed)
	
	import time
	from pathlib import Path
	
	start_time = time.perf_counter()
	experiment.run()
	elapsed_time = time.perf_counter() - start_time
	
	total_mensajes = experiment.engine.total_events
	event_counts = experiment.engine.event_counts
	
	event_types = [
		"PIF-EXPLORACION",
		"PACKAGE",
		"ACK",
		"PIF-NEGOCIACION",
		"SOLICITUD-CONEXION",
		"ACEPTO-CONEXION",
		"DESCONEXION",
		"DESCONEXION-RECIBIDA",
		"RECHAZO-CONEXION",
		"PIF-CONEXION"
	]
	
	if len(sys.argv) > 1:
		hoja_dir = Path(sys.argv[1]).parent
	else:
		hoja_dir = Path(".")
		
	metrics_file = hoja_dir / "metricas_rendimiento.csv"
	with open(metrics_file, "w", encoding="utf-8") as mf:
		header = ["Tiempo_Ejecucion_Segundos", "Mensajes_Intercambiados"] + event_types
		mf.write(",".join(header) + "\n")
		
		row = [f"{elapsed_time:.6f}", str(total_mensajes)]
		for et in event_types:
			row.append(str(event_counts.get(et, 0)))
		mf.write(",".join(row) + "\n")