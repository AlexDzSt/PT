
import random
import math
import networkx as nx

class Encaminamiento:

    def __init__(self):
        self._coord_cache_grid = {}   # {nodeId: (x, y)}
        self._coord_cache_ring = {}   # {nodeId: (x, y)}
        self._sp_cache = {}           # {source: {dest: path}}

    #---ANILLO-------------------------------

    def distanciaAnillo(self,node1,node2,main):
        """Returns the euclidean distance between two nodes"""
        n1 = self.getCoordinatesAnillo(node1,main)
        n2 = self.getCoordinatesAnillo(node2,main)
        d = math.sqrt(math.pow(n2[0]-n1[0],2)+math.pow(n2[1]-n1[1],2))
        return d

    def getCoordinatesAnillo(self,nodeId,main):
        if nodeId not in self._coord_cache_ring:
            #transformación de coordenadas polares a rectangulares:
            angulo=math.radians((360/main.nodes)*nodeId)
            x = 10*math.cos(angulo)
            y = 10*math.sin(angulo)
            self._coord_cache_ring[nodeId] = (x,y)
        return self._coord_cache_ring[nodeId]

    #---MALLA----------------------------------

    def distancia(self,node1,node2,main):
        n1 = self.getCoordinates(node1,main)
        n2 = self.getCoordinates(node2,main)
        d = math.sqrt(math.pow(n2[0]-n1[0],2)+math.pow(n2[1]-n1[1],2))
        return d

    def getCoordinates(self,nodeId,main):
        if nodeId not in self._coord_cache_grid:
            x = (nodeId-1)%main.rows
            y = (nodeId-1)//main.columns
            self._coord_cache_grid[nodeId] = (x,y)
        return self._coord_cache_grid[nodeId]

    #---COMPASS ROUTING--------------------------

    def angulo(self,destino,origen,nodo,main):
        #print("or",origen,"des",destino,"vecino",nodo)
        if main.grafo==1:#si trabajo en una malla:
            p0 = self.getCoordinates(origen,main)
            p1 = self.getCoordinates(destino,main)
            p2 = self.getCoordinates(nodo,main)
        elif main.grafo==3:#si trabajo en un anillo:
            p0 = self.getCoordinatesAnillo(origen,main)
            p1 = self.getCoordinatesAnillo(destino,main)
            p2 = self.getCoordinatesAnillo(nodo,main)
        #print("origen",p0)
        #print("destino",p1)
        #print("vecino",p2)
        u1 = p1[0]-p0[0]
        #print("u1",u1)
        u2 = p1[1]-p0[1]
        #print("u2",u2)
        v1 = p2[0]-p0[0]
        #print("v1",v1)
        v2 = p2[1]-p0[1]
        #print("v2",v2)
        #calculo del angulo entre vectores mediante producto punto
        div = (u1*v1+u2*v2)/((math.sqrt((math.pow(u1,2) + math.pow(u2,2))))*(math.sqrt((math.pow(v1,2) + math.pow(v2,2)))))
        try:
            angC=math.degrees(math.acos(div))
        except ValueError as error:
            if div>1.0:
                #print("mas grande",div)
                div=1.0
            elif div < -1.0:
                #print("mas pequeño",div)
                div=-1.0
            angC=math.degrees(math.acos(div))
        #print("angC",angC)
        return angC

    def Compass_Routing(self,identificador,vecinos,destino,main):
        w = random.choice(vecinos)
        ang = 360
        #print("origen",identificador,"destino",destino,"vecinos",vecinos)
        if destino in vecinos:
            w = destino
        else:
            for vecino in vecinos:     
                ang2 = self.angulo(destino,identificador,vecino,main) 
                if ang2 <= ang:
                    w = vecino
                    ang = ang2
        #print("angulo seleccionado",ang)
        #print("vecino seleccionado",w)
        return w 

    #---SHORTEST PATH (DIJKSTRA)----------------

    def shortestPath(self,grafo,fuente,destino):
        if fuente not in self._sp_cache:
            self._sp_cache[fuente] = dict(nx.single_source_shortest_path(grafo, fuente))
        return list(self._sp_cache[fuente][destino])

    def clearShortestPathCache(self):
        """Limpia la caché de rutas más cortas (necesario entre ciclos cuando la topología cambia)."""
        self._sp_cache.clear()

    #---RANDOM WALK-----------------------------

    def Random_Walk(self,identificador,vecinos):
        w=random.choice(vecinos)
        return w

    def Random_Walk_Degree(self,identificador,vecinos,grafo):
        """Caminata sesgada por grado.
        P(u) = deg(u) / sum(deg(v) for v in vecinos)
        Favorece saltar hacia nodos de mayor conectividad (hubs).
        """
        pesos = [grafo.degree(u) for u in vecinos]
        total = sum(pesos)
        if total == 0:
            return random.choice(vecinos)
        return random.choices(vecinos, weights=pesos, k=1)[0]

    def Random_Walk_Inverse(self,identificador,vecinos,grafo):
        """Caminata sesgada por grado inverso.
        P(u) = (1/deg(u)) / sum(1/deg(v) for v in vecinos)
        Favorece saltar hacia nodos de menor conectividad (periféricos).
        """
        pesos = [1.0 / grafo.degree(u) for u in vecinos]
        total = sum(pesos)
        if total == 0:
            return random.choice(vecinos)
        return random.choices(vecinos, weights=pesos, k=1)[0]

    def Random_Walk_Node2Vec(self,identificador,vecinos,grafo,prev=None,p=1.0,q=1.0):
        """Caminata node2vec.
        Los parámetros p y q controlan el sesgo de retorno y de exploración:
            alpha(x) = 1/p  si x == prev          (retorno al nodo anterior)
                     = 1    si x es vecino de prev  (triángulo local / BFS)
                     = 1/q  en otro caso            (exploración DFS)
        p > 1 desincentiva volver atrás.
        q < 1 favorece exploración hacia nodos lejanos (DFS).
        q > 1 favorece exploración local (BFS).
        """
        if prev is None:
            return random.choice(vecinos)
        vecinos_prev = set(grafo.neighbors(prev))
        pesos = []
        for x in vecinos:
            if x == prev:
                alpha = 1.0 / p
            elif x in vecinos_prev:
                alpha = 1.0
            else:
                alpha = 1.0 / q
            pesos.append(alpha)
        return random.choices(vecinos, weights=pesos, k=1)[0]

    #---EXTRAS----------------------------------

    def generaDestino(self,identificador,main,vecinos):
        """genera un nodo destino de manera aleatoria (que no esté en mis vecinos)"""
        d = False
        while(d == False):
            dest=-1
            if main.grafo==3:#si es un anillo con el que estoy trabajando:
                dest = random.randint(1,main.nodes)
            elif main.grafo==1:#si es una malla con la que estoy trabajando:
                dest = random.randint(1,main.rows*main.columns)
            if dest != identificador and dest not in vecinos:
                d = True
        return dest