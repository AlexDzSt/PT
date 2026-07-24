""" An instance of "Simulation" represents an experiment in which a distributed
algorithm (or several) is (are) executed on top of a network.  """

from process import Process                # to build the table of active entities
from simulator import Simulator            # the definition of the engine
import re                                  # regular expressions library
#---------------------------------------------
class Simulation:                          # Descends from "Object" (default)
	""" Simulation attributes: "engine", "graph", "weights" & "table", 
	contains constructor and getters in python3-style, as well as the 
	methods "setModel()", "insertEvent()" & "run()". """


	def __init__(self, maxtime, main):
		""" Builds an instance of "Simulation", including its events engine, the 
		underlying network, the weights of the links & the table of processes. """
		self.__numnodes = 0  # <-- Contador
		self.__engine = Simulator(maxtime)
		self.__graph = []
		self.__weights = []
		self.__table  = [[]]          # la entrada 0 se deja vacia
		self.__main = main #obtiene una referencia al objeto main

	def readAdjacencyListNetworkX(self):

		nodes =list(self.__main.graph.nodes())
		nodes.sort()
		#print("nodos ordenados",nodes)
		for node in nodes:
			vecinos=self.__main.graph.neighbors(node)
			#print("nodo:",node,"vecinos:",[n for n in self.__main.graph.neighbors(node)])
			neighbors = []
			nweight = []
			for f in vecinos:
				neighbors.append(int(f))
				nweight.append(1)#hay que arreglar esto para que acepte grafos ponderados-------------------
			self.__numnodes+=1    
			self.__graph.append(neighbors)
			self.__weights.append(nweight)
		for i,row in enumerate(self.__graph):
			newprocess = Process(row, self.__weights[i], self.__engine, i+1)
			self.__table.append(newprocess)
 
	@property
	def engine(self):
		return self.__engine

	@property
	def graph(self):
		return self.__graph

	@property
	def table(self):
		return self.__table

	@property
	def numnodes(self): 
		return self.__numnodes
		
	def setModel(self, model, id, port=0):
		""" asocia al proceso con el modelo que debe ejecutar y viceversa """
		process = self.__table[id]
		process.setModel(model, port)
		
	def init(self, event):
		""" inserta un evento semilla en la agenda """
		self.__engine.insertEvent(event)

	def run(self):	
		""" arranca el motor de simulacion """
		while self.__engine.isOn():
			nextevent = self.__engine.returnEvent()
			target = nextevent.target 
			time = nextevent.time
			port = nextevent.port
			nextprocess = self.__table[target]
			nextprocess.setTime(time, port)
			nextprocess.receive(nextevent, port)
