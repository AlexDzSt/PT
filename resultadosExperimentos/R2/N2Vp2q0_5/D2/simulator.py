""" An instance of "Simulator" represents the engine that moves the experiment
enabling the communication between processes, enforcing the causal order of the
exchanged events. """

import heapq

#---------------------------------------------
class Simulator:                           # Descends from "Object" (default)
	""" Simulator attributes: "clock" & "agenda", 
	contains constructor and getters in python3-style, as well as the 
	methods "insertEvent()", "returnEvent()" & "isOn()". """
	
	def __init__(self, lastmoment):
		""" Builds an instance of "Simulator", sets the initial value of clock 
		and uses a binary heap for the event agenda. A monotonic counter
		ensures FIFO ordering for events at the same time. """
		self.__clock = 0.0
		self.__agenda = []       # heap of (time, counter, event)
		self.__counter = 0       # monotonic counter for tie-breaking
		self.__total_events = 0  # contador de mensajes/eventos totales
		self.__event_counts = {} # diccionario {nombre_evento: cantidad}
		
	@property
	def clock(self):
		""" Invoke as x.clock, without "()". """
		return self.__clock

	@property
	def agenda(self):
		""" Invoke as x.agenda, without "()". """
		return self.__agenda

	@property
	def total_events(self):
		""" Devuelve la cantidad de eventos procesados (mensajes intercambiados). """
		return self.__total_events

	@property
	def event_counts(self):
		""" Devuelve la cantidad de eventos procesados agrupados por su nombre. """
		return self.__event_counts
	
	def insertEvent(self, event):
		""" Inserts an event in the agenda using a binary heap for O(log n)
		insertion. The monotonic counter ensures FIFO ordering for events
		at the same time, matching the original linear insertion behavior. """
		self.__counter += 1
		heapq.heappush(self.__agenda, (event.time, self.__counter, event))
	
	def returnEvent(self):
		""" Returns and removes the event with the smallest time from the agenda. """
		time, counter, event = heapq.heappop(self.__agenda)
		self.__total_events += 1
		event_name = event.name
		self.__event_counts[event_name] = self.__event_counts.get(event_name, 0) + 1
		return event

	def isOn(self):
		""" True, provided that there are events remaining in the agenda. """
		return len(self.__agenda) > 0
