import numpy as np
import operator
from collections import Counter
import copy

class Reglas:
    def __init__(self,regla,f_n,paquetes):
        self.r=regla
        self.f_n=f_n 
        self.paquetes=paquetes
        self.candidato=-1
         
    def seleccionar_candidato(self):
        if self.r==1:
            self.candidato=self.regla_1()
        elif self.r==2:
            self.candidato=self.regla_2()
        elif self.r==3:
            self.candidato=self.regla_3()
        return self.candidato

    def regla_1(self):
        self.f_n = sorted(self.f_n.items(), key=operator.itemgetter(1), reverse=True)# Ordeno el diccionario f_n de mayor a menor frecuencia
        self.f_n = dict(self.f_n)#convierto la lista de tuplas en un diccionario nuevamente, pero ya ordenado
        claves=list(self.f_n.keys())#obtengo las claves del diccionario f_n, en este caso son los id´s de los nodos visitados
        clave=claves[0]
        return clave

    def regla_2(self):
        claves=list(self.f_n.keys())#obtengo las claves del diccionario f_n, en este caso son los id´s de los nodos visitados
        clave=claves[0]
        return clave
    
    def regla_3(self):
        llaves=list(self.f_n.keys())
        probabilidades=list(self.f_n.values())
        probabilidades=probabilidades/np.sum(probabilidades)
        clave=np.random.choice(llaves,p=probabilidades)
        return clave
 
        long = len(self.vector_distancia)
        popularidad, distancia = self.matrices(long)
        self.popularidad = popularidad
        self.distancia = distancia
        probabilidad=self.nodos_seleccionados(popularidad, distancia)
        if len(probabilidad)==0:
            return -1
        probabilidades = np.array(list(probabilidad.values()), dtype=float)
        suma = np.sum(probabilidades)
        probabilidades = probabilidades/suma
        probabilidad_normalizada = dict(zip(probabilidad.keys(), probabilidades))
        #Elegir un nodo candidato
        nodos = list(probabilidad_normalizada.keys())
        proba = list(probabilidad_normalizada.values())
        candidato = np.random.choice(nodos, p=proba)
        return candidato