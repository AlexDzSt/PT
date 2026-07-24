
from event import Event
from model import Model
from encaminamiento import Encaminamiento
import random
from paquete import Paquete
from reglas import Reglas
from enlace import Enlace
import sys
import operator
import networkx as nx

class ComplexNetwork(Model):
    """La clase ComplexNetwork desciende de la clase Model e implementa los metodos init() y  
        receive()"que en la clase madre se definen como abstractos"""

    def __init__(self,main,numEnlacesDinamicos,maximasConexiones,num_paquetes,algoritmoEncaminamiento,regla):#constructor de la clase ComplexNetwork
        self.__main=main
        self.__numEnlacesDinamicos=numEnlacesDinamicos
        self.__maximoConexionesPermitidas=maximasConexiones
        self.__num_paquetes=num_paquetes
        self.__encaminamiento=Encaminamiento()
        self.__algoritmoEncaminamiento=algoritmoEncaminamiento
        self.__regla=regla
        
    def init(self):
        self.contadorCiclos=1#contador que indica el ciclo en el que se encuentra la simulación
        self.frecEnlace = self.__num_paquetes*1#umbral de frecuencia de uso de enlace dinámico
        if(self.__regla==1):
            self.frecNodo = self.__num_paquetes*0.5#umbral de frecuencia de visita a nodos, sobre la cual se considera a un nodo en la regla de recableado
        else:
            self.frecNodo = self.__num_paquetes*0
        self.paquetes=[]#lista de paquetes exploradores enviados, servirá para buscar rutas hacia mis posibles candidatos a conexión
        self.paquetesRegreso=0#contador de paquetes exploradores que han regresado a mi, después de explorar la red
        self.enlacesDinamicos=[]#lista de enlaces dinámicos que tengo
        self.enlacesDinamicosDict={}#diccionario para búsquedas O(1) de enlaces por id
        self.vecinosConectadosDinamicos=[]#lista de id´s de los nodos con los que estoy conectado mediante mis enlaces dinámicos
        self.f_n = {} #Diccionario de frecuencia de visita a los nodos que no son mis vecinos, mediante mis paquetes exploradores
        self.f_e = {} #Diccionario de frecuencia de uso de mis enlaces dinámicos
        self.listaNodosSolicitados=set()#lista de id´s de nodos solicitados para conectarme con ellos en un ciclo en particular
        self.numeroSolicitudes=0#número de solicitudes hechas por mi en un ciclo de recableado
        self.conexionesAceptadas=0#numero de conexiones que he aceptado en la simulación
        self.neighborsPendientes=[]#lista de vecinos que voy a agregar en la etapa de recableado, se da en los nodos destino (los que aceptan solicitudes)
        self.neighborsPendientesEliminacion=[]#lista de vecinos que voy a eliminar en la etapa de recableado
        self.solicitudesPendientes=[]#lista de solicitudes que tengo que atender en la etapa de recableado, se da en los nodos origen (los que mandan solicitudes)
        self.creaLongitudEnlacesDinamicos()#se establece la longitud de cada uno de mis enlaces dinámicos
        self.diametroGrafo=-1#diametro del grafo en cada ciclo de simulación (para uso en Random-Walk)
        #para los 3 PIF que completan un ciclo:
        self.visited=False
        self.father=-1
        self.count=1

    def receive(self, event):
        
        #print("soy",self.id,"recibo",event.name,"de",event.source,"en el tiempo",event.time)
        
        #PIF fase de EXPLORACION-------------------------------
        if(event.name=="PIF-EXPLORACION"):
            self.count-=1
            if(self.visited==False):
                self.diametroGrafo=event.package.diametro
                #print("soy",self.id,"recibo",event.name,"por primera vez, viene de",event.source,"en el tiempo",event.time,"el diametro del grafo es",self.diametroGrafo)
                self.visited=True
                self.father=event.source
                conjuntoVecinos=self.neighbors+self.vecinosConectadosDinamicos
                for i in conjuntoVecinos:
                    if i != self.father:
                        newevent=Event("PIF-EXPLORACION",self.clock+1.0,i,self.id,event.package)
                        self.transmit(newevent)
                        self.count+=1
                #después de propagar mi PIF-EXPLORACION, empiezo mi fase de EXPLORACIÓN
                new_package=Paquete(self.id,self.paquetesRegreso)
                nextStep=-1
                neighborsTodos=self.neighbors+self.vecinosConectadosDinamicos
                if(self.__algoritmoEncaminamiento=="COMPASS-ROUTING"):
                    new_package.destino=self.__encaminamiento.generaDestino(self.id,self.__main,neighborsTodos)
                    nextStep = self.__encaminamiento.Compass_Routing(self.id,neighborsTodos,new_package.destino,self.__main)
                    #print("soy",self.id,"genero el paquete",self.paquetesRegreso,"que va a dirigido a",new_package.destino,"en el tiempo",(event.time+1.0),"lo mando hacia",nextStep)
                    new_package.ruta.append(self.id)#agrego a la ruta del paquete mi id 
                elif(self.__algoritmoEncaminamiento=="RANDOM-WALK"):
                    new_package.distanciaMaxima=random.randint(2,self.diametroGrafo)#la distancia maxima esta comprendida entre [2,diametroGrafo]
                    nextStep = self.__encaminamiento.Random_Walk(self.id,neighborsTodos)
                    #print("soy",self.id,"genero el paquete",self.paquetesRegreso,"que va a distancia",new_package.distanciaMaxima,"en el tiempo",(event.time+1.0),"lo mando hacia",nextStep,"el diametro de la red es:",event.package.diametro) 
                    new_package.ruta.append(self.id)#agrego a la ruta del paquete mi id
                elif(self.__algoritmoEncaminamiento=="SHORTEST-PATH"):
                    new_package.destino=self.__encaminamiento.generaDestino(self.id,self.__main,neighborsTodos)
                    new_package.ruta=self.__encaminamiento.shortestPath(self.__main.graph,self.id,new_package.destino)[:]
                    new_package.rutaAuxiliar=new_package.ruta[:]
                    new_package.ruta.pop(0)#elimino el primer elemento de la lista, que en este caso es mi id
                    nextStep = new_package.ruta[0]
                    new_package.ruta.pop(0)#elimino el primer elemento de la lista, que en este caso es mi vecino directo
                    #print("soy",self.id,"genero el paquete",self.paquetesRegreso,"que va a dirigido a",new_package.destino,"en el tiempo",(event.time+1.0),"lo mando hacia",nextStep,"ruta",new_package.rutaAuxiliar)
                self.paquetes.append(new_package)
                newevent = Event("PACKAGE", event.time + 1.0, nextStep, self.id, new_package)            
                self.transmit(newevent)
                    
            if self.count == 0 and self.paquetesRegreso==self.__num_paquetes:
                if(self.father!=self.id):
                    newevent = Event("PIF-EXPLORACION", self.clock+1.0, self.father, self.id, None)
                    self.transmit(newevent)
                    #print ("soy",self.id,"TERMINO y envio",newevent.name,"hacia mi papa",newevent.target,"en el tiempo",newevent.time)
                    self.reiniciaPIF()
                else:
                    self.reiniciaPIF()
                    #print("soy",self.id,"COORDINADOR y declaro terminada la fase de EXPLORACION, ahora inicio el PIF-NEGOCIACION ciclo:",self.contadorCiclos)
                    newevent = Event("PIF-NEGOCIACION", self.clock+1.0, self.id, self.id, None)
                    self.transmit(newevent)

        elif(event.name=="PACKAGE"):
            if((event.package.distanciaMaxima > 1 and self.__algoritmoEncaminamiento=="RANDOM-WALK") or (event.package.destino!=self.id and self.__algoritmoEncaminamiento=="COMPASS-ROUTING")):
                auxNeighbors=self.neighbors+self.vecinosConectadosDinamicos#Creo una copia de mi lista de vecinos
                for n in event.package.ruta:#Descarto de la copia a todos los vecinos por los que ya paso el paquete
                    if n in auxNeighbors:
                        auxNeighbors.remove(n)
                #if(self.__algoritmoEncaminamiento=="COMPASS-ROUTING"):
                    #print("soy",self.id,"el paquete",event.package.idPaquete,"tiene origen:",event.package.sourceID,"destino:",event.package.destino,"auxNeighbors:",auxNeighbors,"vs",self.neighbors,"ruta",event.package.ruta)
                #elif(self.__algoritmoEncaminamiento=="RANDOM-WALK"):
                    #print("soy",self.id,"el paquete",event.package.idPaquete,"va a distancia",event.package.distanciaMaxima,"origen:",event.package.sourceID,",auxNeighbors:",auxNeighbors,"vs",self.neighbors,"ruta",event.package.ruta)
                if(len(auxNeighbors)==0):#si ya no tengo a quien enviarle el paquete, por que son nodos ya visitados, mando ack, fase de BACKTRACK
                    event.package.ruta.append(self.id)#agrego a la ruta del paquete mi id
                    #print("soy",self.id,"ya no tengo a quien enviarle el paquete, ya llego a su destino, la ruta por la que paso es",event.package.ruta)
                    event.package.rutaAuxiliar=event.package.ruta[:]#guardo la ruta original necesaria para llegar al destino
                    event.package.ruta.pop(len(event.package.ruta)-1)#elimino el ultimo id de la ruta, que en este caso es mi id, para no enviarme ACK a mi mismo
                    event.package.ruta.reverse()#invierto la ruta del mensaje
                    nextStep=event.package.ruta[0]
                    event.package.ruta.pop(0)#elimino el primer elemento de la lista
                    newevent = Event("ACK", event.time + 1.0, nextStep, self.id, event.package)            
                    self.transmit(newevent)
                    #print("soy",self.id,"mando ACK a",event.package.sourceID,"mediante",nextStep,"en el tiempo",newevent.time,"ruta faltante hasta ahora",newevent.package.ruta)
                else:
                    nextStep=-1
                    event.package.ruta.append(self.id)#agrego a la ruta del paquete mi id
                    if(self.__algoritmoEncaminamiento=="COMPASS-ROUTING"):
                        nextStep = self.__encaminamiento.Compass_Routing(self.id,auxNeighbors,event.package.destino,self.__main)
                        #print("soy",self.id,"reenvio paquete",event.package.idPaquete,"proveniente de",event.package.sourceID,"a",nextStep,", su destino es",event.package.destino,"en el tiempo",(event.time+1.0),"ruta hasta ahora",event.package.ruta)
                    elif(self.__algoritmoEncaminamiento=="RANDOM-WALK"):
                        nextStep = self.__encaminamiento.Random_Walk(self.id,auxNeighbors)
                        event.package.distanciaMaxima-=1
                        #print("soy",self.id,"reenvio paquete",event.package.idPaquete,"proveniente de",event.package.sourceID,"faltan",event.package.distanciaMaxima,"pasos, en el tiempo",(event.time+1.0),"lo mando hacia",nextStep,"ruta hasta ahora",event.package.ruta)   
                    newevent = Event("PACKAGE", event.time + 1.0, nextStep, self.id, event.package)            
                    self.transmit(newevent)
            else:
                if(event.package.destino!=self.id and self.__algoritmoEncaminamiento=="SHORTEST-PATH"):
                    nextStep=event.package.ruta[0]
                    event.package.ruta.pop(0)
                    #print("soy",self.id,"reenvio paquete",event.package.idPaquete,"proveniente de",event.package.sourceID,"a",nextStep,", su destino es",event.package.destino,"en el tiempo",(event.time+1.0),"ruta",event.package.ruta)
                    newevent = Event("PACKAGE", event.time + 1.0, nextStep, self.id, event.package)            
                    self.transmit(newevent)
                else:#si el paquete ya llegó a su destino
                    if(self.__algoritmoEncaminamiento!="SHORTEST-PATH"):
                        event.package.ruta.append(self.id)#agrego a la ruta del paquete mi id
                        #print("soy",self.id,"ya llegó el paquete",event.package.idPaquete,"proveniente de",event.package.sourceID,"a su destino, la ruta por la que paso es",event.package.ruta)
                    #else:
                        #print("soy",self.id,"ya llegó el paquete",event.package.idPaquete,"proveniente de",event.package.sourceID,"a su destino, la ruta por la que paso es",event.package.rutaAuxiliar)
                    if(self.__algoritmoEncaminamiento!="SHORTEST-PATH"):
                        event.package.rutaAuxiliar=event.package.ruta[:]#guardo la ruta original necesaria para llegar al destino
                    else:
                        event.package.ruta=event.package.rutaAuxiliar[:]
                    event.package.ruta.pop(len(event.package.ruta)-1)#elimino el último id de la ruta, que en este caso es mi id, para no enviarme ACK a mi mismo
                    event.package.ruta.reverse()#invierto la ruta del mensaje
                    nextStep=event.package.ruta[0]
                    event.package.ruta.pop(0)#elimino el primer elemento de la lista
                    newevent = Event("ACK", event.time + 1.0, nextStep, self.id, event.package)            
                    self.transmit(newevent)
                    #print("soy",self.id,"mando ACK a",event.package.sourceID,"mediante",nextStep,"en el tiempo",newevent.time,"ruta faltante hasta ahora",newevent.package.ruta)
        
        elif(event.name=="ACK"):
            if(len(event.package.ruta) > 0):#si todavía falta recorrer nodos en la ruta    
                nextStep=event.package.ruta[0]
                event.package.ruta.pop(0)#elimino el primer elemento de la lista
                newevent = Event("ACK", event.time + 1.0, nextStep, self.id, event.package)            
                self.transmit(newevent)
                #print("soy",self.id,"reenvio ACK del paquete",event.package.idPaquete,"con destino",event.package.sourceID,"mediante",nextStep,"en el tiempo",newevent.time,"ruta faltante hasta ahora",newevent.package.ruta)
            else:
                self.paquetesRegreso+=1#incremento mi número de paquetes recibidos
                #verifico si se incrementa la frecuencia de uso de uno de mis enlaces dinámicos
                for enlace in self.enlacesDinamicos:
                    if(event.source==enlace.idConectado):
                        self.f_e[enlace.idEnlace]+=1
                        break
                nodos=self.f_n.keys()#obtengo las claves de mi diccionario de f_n, las cuales representan los nodos visitados
                ruta = []
                conjuntoVecinos=set(self.neighbors+self.vecinosConectadosDinamicos)
                for i in range(2,len(event.package.rutaAuxiliar)):#recorro cada uno de los nodos de la ruta, excepto mi id y mi vecino directo
                    """NOTA: es necesario poner la condicion event.package.rutaAuxiliar[i] not in self.neighbors por que el algoritmo de encaminamiento
                        puede hacer que los paquetes pasen por mis vecinos directos, y no debo considerarlos en f_n"""
                    if(event.package.rutaAuxiliar[i] not in conjuntoVecinos):
                        ruta.append(event.package.rutaAuxiliar[i])
                        if(event.package.rutaAuxiliar[i] not in nodos):#si el nodo no está en las claves, lo agrego con valor 1
                            self.f_n[event.package.rutaAuxiliar[i]]=1
                        else:#si ya se encuentra el nodo en las claves del diccionario, lo incremento en 1 unidad
                            self.f_n[event.package.rutaAuxiliar[i]]+=1
                #print("soy",self.id,"ya llegó de regreso el paquete",event.package.idPaquete,"la ruta usada original fue",event.package.rutaAuxiliar,"tiempo",event.time,"f_n:",self.f_n)
                if(self.paquetesRegreso<self.__num_paquetes):
                    new_package=Paquete(self.id,self.paquetesRegreso)
                    nextStep=-1
                    neighborsTodos=self.neighbors+self.vecinosConectadosDinamicos
                    if(self.__algoritmoEncaminamiento=="COMPASS-ROUTING"):
                        new_package.destino=self.__encaminamiento.generaDestino(self.id,self.__main,neighborsTodos)
                        nextStep = self.__encaminamiento.Compass_Routing(self.id,neighborsTodos,new_package.destino,self.__main)
                        #print("soy",self.id,"genero el paquete",self.paquetesRegreso,"que va a dirigido a",new_package.destino,"en el tiempo",(event.time+1.0),"lo mando hacia",nextStep)
                        new_package.ruta.append(self.id)#agrego a la ruta del paquete mi id 
                    elif(self.__algoritmoEncaminamiento=="RANDOM-WALK"):
                        new_package.distanciaMaxima=random.randint(2,self.diametroGrafo)#la distancia máxima esta comprendida entre [2,diametroGrafo]
                        nextStep = self.__encaminamiento.Random_Walk(self.id,neighborsTodos)
                        #print("soy",self.id,"genero el paquete",self.paquetesRegreso,"que va a distancia",new_package.distanciaMaxima,"en el tiempo",(event.time+1.0),"lo mando hacia",nextStep) 
                        new_package.ruta.append(self.id)#agrego a la ruta del paquete mi id
                    elif(self.__algoritmoEncaminamiento=="SHORTEST-PATH"):
                        new_package.destino=self.__encaminamiento.generaDestino(self.id,self.__main,neighborsTodos)
                        new_package.ruta=self.__encaminamiento.shortestPath(self.__main.graph,self.id,new_package.destino)[:]
                        new_package.rutaAuxiliar=new_package.ruta[:]
                        new_package.ruta.pop(0)#elimino el primer elemento de la lista, que en este caso es mi id
                        nextStep = new_package.ruta[0]
                        new_package.ruta.pop(0)#elimino el primer elemento de la lista, que en este caso es mi vecino directo
                        #print("soy",self.id,"genero el paquete",self.paquetesRegreso,"que va a dirigido a",new_package.destino,"en el tiempo",(event.time+1.0),"lo mando hacia",nextStep,"ruta",new_package.rutaAuxiliar)
                    self.paquetes.append(new_package)
                    newevent = Event("PACKAGE", event.time + 1.0, nextStep, self.id, new_package)            
                    self.transmit(newevent)
                """Esta seccion de código es por si todos mis vecinos ya me mandaron el PIF-EXPLORACION, pero yo aún no terminaba 
                    de recibir todos mis ACK´s, entonces me espero a recibir todos mis ACK´s para mandar PIF-EXPLORACION a mi padre"""
                if self.count == 0 and self.paquetesRegreso==self.__num_paquetes:
                    if(self.father!=self.id):
                        newevent = Event("PIF-EXPLORACION", self.clock+1.0, self.father, self.id, None)
                        self.transmit(newevent)
                        #print ("soy",self.id,"TERMINO y envio",newevent.name,"hacia mi papa",newevent.target,"en el tiempo",newevent.time)
                        self.reiniciaPIF()
                    else:
                        self.reiniciaPIF()
                        #print("soy",self.id,"COORDINADOR y declaro terminada la fase de EXPLORACION, ahora inicio el PIF-NEGOCIACION ciclo:",self.contadorCiclos)
                        newevent = Event("PIF-NEGOCIACION", self.clock+1.0, self.id, self.id, None)
                        self.transmit(newevent)

        #PIF fase de NEGOCIACION--------------------------------
        elif(event.name=="PIF-NEGOCIACION"):
            self.count-=1
            if(self.visited==False):
                #print("soy",self.id,"recibo",event.name,"por primera vez, viene de",event.source,"en el tiempo",event.time)
                self.visited=True
                self.father=event.source
                conjuntoVecinos=self.neighbors+self.vecinosConectadosDinamicos
                for i in conjuntoVecinos:
                    if i != self.father:
                        newevent=Event("PIF-NEGOCIACION",self.clock+1.0,i,self.id,None)
                        self.transmit(newevent)
                        self.count+=1
                #después de propagar mi PIF-NEGOCIACION, empiezo mi fase de NEGOCIACIÓN
                hayEnlacesLibres=False#booleano que se activará si alguno de mis enlaces dinámicos está libre
                if(len(self.f_n)>0):#si existen candidatos a reconexión en f_n:
                    for enlace in self.enlacesDinamicos:#recorro mi lista de enlaces dinámicos en busca de enlaces libres
                        if(enlace.libre==True):#si el enlace en cuestión está libre, mando solicitudes de CABLEADO
                            hayEnlacesLibres=True
                            clave=-1
                            #selecciono al candidato a conexión
                            rule = Reglas(self.__regla,self.f_n, self.paquetes)
                            clave = rule.seleccionar_candidato()
                                
                            distancia=-1
                            if(self.__main.grafo!=3):#si no estoy trabajando en un anillo:
                                distancia=self.__encaminamiento.distancia(self.id,clave,self.__main)
                            else:#si estoy trabajando en un anillo:
                                distancia=self.__encaminamiento.distanciaAnillo(self.id,clave,self.__main)
                            if(distancia<=enlace.longitud):#el candidato seleccionado cumple con la restricción de distancia del enlace
                                #if(clave in self.neighbors or clave in self.vecinosConectadosDinamicos):
                                    #print("no puede ser ese nodo esta en mis vecinos")
                                new_package=Paquete(self.id,0)#a este paquete se le puede agregar lo que sea
                                #busco entre las rutas de mis paquetes aquellas que me permiten llegar al destino
                                rutasCandidatas=[]#lista de todas las rutas que pasan por el nodo, de ellas buscare la de menor longitud
                                for i in self.paquetes:#recorro mi lista de paquetes
                                    ruta=[]
                                    if (clave in i.rutaAuxiliar):#si el nodo se encuentra en la ruta del i-ésimo paquete
                                        for j in range(0,len(i.rutaAuxiliar)):#recorro toda la ruta 
                                            ruta.append(i.rutaAuxiliar[j])#agrego los nodos de la ruta hasta que llegue al destino
                                            if(i.rutaAuxiliar[j]==clave):
                                                break
                                        rutasCandidatas.append(ruta)#agrego la ruta a la lista de rutas candidatas
                                #busco la ruta de menor longitud que me permite llegar al destino
                                minimaLongitud=sys.maxsize
                                for i in rutasCandidatas:
                                    if(len(i)<minimaLongitud):
                                        new_package.ruta=i[:]
                                        minimaLongitud=len(i)
                                new_package.rutaAuxiliar=new_package.ruta[:]#copio la ruta en rutaAuxiliar
                                del new_package.ruta[0]#elimino mi id de la ruta principal
                                nextStep=new_package.ruta[0]
                                self.listaNodosSolicitados.add(clave)
                                self.numeroSolicitudes+=1
                                del new_package.ruta[0]#elimino el id al que le voy a enviar el mensaje de la ruta principal
                                new_package.idEnlace=enlace.idEnlace#agrego al paquete el id del enlace dinamico que esta haciendo la solicitud
                                #envío la solicitud al nodo con el que me quiero conectar mediante la ruta de menor longitud encontrada previamente
                                newevent = Event("SOLICITUD-CONEXION", event.time + 1.0, nextStep, self.id, new_package)            
                                self.transmit(newevent)
                                #if(self.__main.grafo!=3):
                                    #print("soy",self.id,"mando",newevent.name,"a",clave,"en el tiempo",newevent.time,"mediante mi enlace dinamico numero",enlace.idEnlace,"la distancia del enlace es",self.__encaminamiento.distancia(self.id,clave,self.__main),"la ruta a seguir es",new_package.rutaAuxiliar,"ahorita envio el mensaje a",nextStep)
                                #else:
                                    #print("soy",self.id,"mando",newevent.name,"a",clave,"en el tiempo",newevent.time,"mediante mi enlace dinamico numero",enlace.idEnlace,"la distancia del enlace es",self.__encaminamiento.distanciaAnillo(self.id,clave,self.__main),"la ruta a seguir es",new_package.rutaAuxiliar,"ahorita envio el mensaje a",nextStep)
                                #print("soy",self.id,"elimino a",clave,"de mi f_n")

                                del self.f_n[clave]#elimino al nodo de la lista de frecuencias para que no se considere en futuros ciclos
                                #print("soy",self.id,"mi f_n queda asi:",self.f_n)
                                if(self.__regla==2):#si ejecuto r2 rompo el ciclo
                                    break
                    if(not(hayEnlacesLibres)):#si no hay enlaces libres:
                        #print("soy",self.id,"ya no tengo enlaces libres, entro a recablear")   
                        #Busco el enlace dinámico que menos haya usado.
                        self.f_e = sorted(self.f_e.items(), key=operator.itemgetter(1))# Ordeno el diccionario f_e de menor a mayor frecuencia
                        #NOTA: la función sorted() usada en la linea anterior, regresa una lista de tuplas
                        self.f_e = dict(self.f_e)#convierto la lista de tuplas en un diccionario nuevamente, pero ya ordenado
                        #print("soy",self.id,"mi f_e ordenado de menor a mayor es:",self.f_e)
                        clavesF_e=list(self.f_e.keys())#obtengo las claves del diccionario previamente ordenado de frecuencias de enlaces dinamicos, en forma de lista
                        if(self.f_e[clavesF_e[0]]<self.frecEnlace):#si la frecuencia de uso del enlace es menor que el umbral self.frecEnlace, entonces busco el recableado
                            #print("soy",self.id,"el enlace a reconectar es el que tiene id",clavesF_e[0])
                            enlaceReconectar=self.enlacesDinamicosDict[clavesF_e[0]]
                            clave=-1
                            #selecciono al candidato a conexión
                            rule = Reglas(self.__regla,self.f_n, self.paquetes)
                            clave = rule.seleccionar_candidato()   
                                    
                            nodoConexion=-1
                            #reviso si se puede hacer el recableado:
                            distancia=-1
                            if(self.__main.grafo!=3):#si no estoy trabajando en un anillo:
                                distancia=self.__encaminamiento.distancia(self.id,clave,self.__main)
                            else:#si estoy trabajando en un anillo:
                                distancia=self.__encaminamiento.distanciaAnillo(self.id,clave,self.__main)
                            if(self.f_n[clave]>=self.frecNodo and distancia<=enlaceReconectar.longitud):
                                nodoConexion=clave
                            if(nodoConexion!=-1):
                                #print("soy",self.id,"elegi a",nodoConexion,"como mi candidato a CONEXION")
                                new_package=Paquete(self.id,0)#a este paquete se le puede agregar lo que sea
                                #busco entre las rutas de mis paquetes aquellas que me permiten llegar al destino
                                rutasCandidatas=[]#lista de todas las rutas que pasan por el nodo, de ellas buscare la de menor longitud
                                for i in self.paquetes:#recorro mi lista de paquetes
                                    ruta=[]
                                    if (nodoConexion in i.rutaAuxiliar):#si el nodo se encuentra en la ruta del i-esimo paquete
                                        for j in range(0,len(i.rutaAuxiliar)):#recorro toda la ruta 
                                            ruta.append(i.rutaAuxiliar[j])#agrego los nodos de la ruta hasta que llegue al destino
                                            if(i.rutaAuxiliar[j]==nodoConexion):
                                                break
                                        rutasCandidatas.append(ruta)#agrego la ruta a la lista de rutas candidatas
                                #busco la ruta de menor longitud que me permite llegar al destino
                                minimaLongitud=sys.maxsize
                                for i in rutasCandidatas:
                                    if(len(i)<minimaLongitud):
                                        new_package.ruta=i[:]
                                        minimaLongitud=len(i)
                                new_package.rutaAuxiliar=new_package.ruta[:]#copio la ruta en rutaAuxiliar
                                del new_package.ruta[0]#elimino mi id de la ruta principal
                                nextStep=new_package.ruta[0]
                                del new_package.ruta[0]#elimino el id al que le voy a enviar el mensaje de la ruta principal
                                new_package.idEnlace=enlaceReconectar.idEnlace#agrego al paquete el id del enlace dinamico que esta haciendo la solicitud
                                self.listaNodosSolicitados.add(nodoConexion)
                                self.numeroSolicitudes+=1
                                #envio la solicitud al nodo con el que me quiero conectar mediante la ruta de menor longitud encontrada previamente
                                newevent = Event("SOLICITUD-CONEXION", event.time + 1.0, nextStep, self.id, new_package)            
                                self.transmit(newevent)
                                #if(self.__main.grafo!=3):#si no estoy trabajando en un anillo:
                                    #print("soy",self.id,"mando",newevent.name,"a",nodoConexion,"en el tiempo",newevent.time,"mediante mi enlace dinamico numero",enlaceReconectar.idEnlace,"la distancia del enlace es",self.__encaminamiento.distancia(self.id,clave,self.__main),"la ruta a seguir es",new_package.rutaAuxiliar,"ahorita envio el mensaje a",nextStep)
                                #else:#si estoy trabajando en un anillo:
                                    #print("soy",self.id,"mando",newevent.name,"a",nodoConexion,"en el tiempo",newevent.time,"mediante mi enlace dinamico numero",enlaceReconectar.idEnlace,"la distancia del enlace es",self.__encaminamiento.distanciaAnillo(self.id,clave,self.__main),"la ruta a seguir es",new_package.rutaAuxiliar,"ahorita envio el mensaje a",nextStep)
                            #else:
                                #print("soy",self.id,"no encontre a nadie con quien reconectarme, por lo tanto no hago nada")
                        #else:
                            #print("soy",self.id,"no pude hacer recableado, self.f_e[clavesF_e[0]]<self.frecEnlace no es verdadero")

            if(self.numeroSolicitudes==0):
                if self.count == 0:
                    if(self.father!=self.id):
                        newevent = Event("PIF-NEGOCIACION", self.clock+1.0, self.father, self.id, event.package)
                        self.transmit(newevent)
                        #print ("soy",self.id,"TERMINO y envio",newevent.name,"hacia mi papa",newevent.target,"en el tiempo",newevent.time)
                        self.reiniciaPIF()
                    else:
                        self.reiniciaPIF()
                        #print("soy",self.id,"COORDINADOR y declaro terminada la fase de NEGOCIACION, ahora inicio el PIF-CONEXION ciclo:",self.contadorCiclos)
                        newevent = Event("PIF-CONEXION", self.clock+1.0, self.id, self.id, None)
                        self.transmit(newevent)   

        elif(event.name=="SOLICITUD-CONEXION"):
            if(len(event.package.ruta)>0):#si el mensaje no viene para mi, simplemente lo reenvio
                nextStep=event.package.ruta[0]
                del event.package.ruta[0]
                newevent = Event("SOLICITUD-CONEXION", event.time + 1.0, nextStep, self.id, event.package)        
                self.transmit(newevent)
                #print("soy",self.id,"reenvio el mensaje",newevent.name,"a",newevent.target,"proveniente de",event.source,"en el tiempo",newevent.time,"ruta faltante",event.package.ruta)
            else:#si el mensaje si viene para mi:
                #print("soy",self.id,"ya llego",event.name,"que venia de",event.package.rutaAuxiliar[0],"en el tiempo",event.time)
                #hago lo necesario para mandar el mensaje de contestacion
                event.package.ruta=event.package.rutaAuxiliar[:]#copio la rutaAuxiliar en la ruta principal
                event.package.ruta.reverse()
                del event.package.ruta[0]#borro el primer nodo de la ruta, que en este caso soy yo
                nextStep=event.package.ruta[0]#asigno a nextStep el nodo en la posicion cero de la ruta
                del event.package.ruta[0]#elimino el primer elemento de la lista        
                if(event.package.sourceID in self.listaNodosSolicitados):#si yo solicite al nodo que me envio solicitud:
                    if(self.id>event.package.sourceID and self.conexionesAceptadas<self.__maximoConexionesPermitidas):#si yo tengo el id mayor de los 2, envio un mensaje de ACEPTACION
                        self.conexionesAceptadas+=1
                        newevent = Event("ACEPTO-CONEXION", event.time + 1.0, nextStep, self.id, event.package)            
                        self.transmit(newevent)
                        self.neighborsPendientes.append(event.package.rutaAuxiliar[0])
                        #print("soy",self.id,"yo pierdo",newevent.name,"con",event.package.rutaAuxiliar[0],"ahora le mando mensaje por la ruta",event.package.ruta,"ahorita el mensaje va a",nextStep,"conexionesAceptadas",self.conexionesAceptadas)
                    else:
                        newevent = Event("RECHAZO-CONEXION", event.time + 1.0, nextStep, self.id, event.package)            
                        self.transmit(newevent)
                        #print("soy",self.id,"yo gano y",newevent.name,"con",event.package.rutaAuxiliar[0],"ahora le mando mensaje por la ruta",event.package.ruta,"ahorita el mensaje va a",nextStep,"conexionesAceptadas",self.conexionesAceptadas)
                else:#pero, si yo no lo solicite:
                    if(self.conexionesAceptadas<self.__maximoConexionesPermitidas):
                        self.conexionesAceptadas+=1
                        newevent = Event("ACEPTO-CONEXION", event.time + 1.0, nextStep, self.id, event.package)            
                        self.transmit(newevent)
                        self.neighborsPendientes.append(event.package.rutaAuxiliar[0])
                        #print("soy",self.id,"yo no te solicite y",newevent.name,"con",event.package.rutaAuxiliar[0],"ahora le mando mensaje por la ruta",event.package.ruta,"ahorita el mensaje va a",nextStep,"conexionesAceptadas",self.conexionesAceptadas)
                    else:
                        newevent = Event("RECHAZO-CONEXION", event.time + 1.0, nextStep, self.id, event.package)            
                        self.transmit(newevent)
                        #print("soy",self.id,"yo no te solicité, pero ya no tengo conexiones disponibles por lo que",newevent.name,"con",event.package.rutaAuxiliar[0],"ahora le mando mensaje por la ruta",event.package.ruta,"ahorita el mensaje va a",nextStep,"conexionesAceptadas",self.conexionesAceptadas)

        elif(event.name=="ACEPTO-CONEXION"):
            if(len(event.package.ruta)>0):#si el mensaje no viene para mi, simplemente lo reenvio
                nextStep=event.package.ruta[0]
                del event.package.ruta[0]
                newevent = Event("ACEPTO-CONEXION", event.time + 1.0, nextStep, self.id, event.package)        
                self.transmit(newevent)
                #print("soy",self.id,"reenvio el mensaje",newevent.name,"a",newevent.target,"proveniente de",event.source,"en el tiempo",newevent.time,"ruta faltante",event.package.ruta)
            else:#si el mensaje si viene para mi:
                #print("soy",self.id,"ya llego",event.name,"que venia de",event.package.rutaAuxiliar[len(event.package.rutaAuxiliar)-1],"en el tiempo",event.time)
                self.numeroSolicitudes-=1#decremento el numero de solicitudes, cuando este llegue a cero quiere decir que ya acabe el recableado
                #agrego a mi lista de solicitudes aprobadas el paquete recibido
                self.solicitudesPendientes.append(event.package)
                #busco el enlace que hizo la peticion
                enlaceConexion=self.enlacesDinamicosDict[event.package.idEnlace]
                #print("soy",self.id,"enlaceConexion=",enlaceConexion.idConectado,"mis vecinos son",self.neighbors,"vecinosConectadosDinamicos",self.vecinosConectadosDinamicos)        
                if(enlaceConexion.idConectado!=-1):#si este es un recableado, envio DESCONEXION al nodo conectado mediante mi enlace dinamico para que me elimine de sus vecinos
                    self.numeroSolicitudes+=1
                    newevent = Event("DESCONEXION", event.time + 1.0, enlaceConexion.idConectado, self.id, None)            
                    self.transmit(newevent)
                    #print("soy",self.id,"mando",newevent.name,"a",newevent.target,"en el tiempo",newevent.time)
            
            if(self.numeroSolicitudes==0):
                if self.count == 0:
                    if(self.father!=self.id):
                        newevent = Event("PIF-NEGOCIACION", self.clock+1.0, self.father, self.id, event.package)
                        self.transmit(newevent)
                        #print ("soy",self.id,"TERMINO y envio",newevent.name,"hacia mi papa",newevent.target,"en el tiempo",newevent.time)
                        self.reiniciaPIF()
                    else:
                        self.reiniciaPIF()
                        #print("soy",self.id,"COORDINADOR y declaro terminada la fase de NEGOCIACION, ahora inicio el PIF-CONEXION ciclo:",self.contadorCiclos)
                        newevent = Event("PIF-CONEXION", self.clock+1.0, self.id, self.id, None)
                        self.transmit(newevent)
        
        elif(event.name=="DESCONEXION"):
            #print("soy",self.id,"ya llego",event.name,"que venia de",event.source,"en el tiempo",event.time)
            self.neighborsPendientesEliminacion.append(event.source)
            self.conexionesAceptadas-=1
            #print("soy",self.id,"agrego a mi lista de neighbors pendientes de eliminacion a",event.source,"que me envio la solicitud de DESCONEXION conexionesAceptadas",self.conexionesAceptadas)
            #hago lo necesario para mandar el mensaje de contestacion
            newevent = Event("DESCONEXION-RECIBIDA", event.time + 1.0, event.source, self.id, None)            
            self.transmit(newevent)
            #print("soy",self.id,"mando",newevent.name,"a",newevent.target,"en el tiempo",newevent.time)        

        elif(event.name=="DESCONEXION-RECIBIDA"):
            #print("soy",self.id,"ya llego",event.name,"que venia de",event.source,"en el tiempo",event.time)
            self.numeroSolicitudes-=1

            if(self.numeroSolicitudes==0):
                if self.count == 0:
                    if(self.father!=self.id):
                        newevent = Event("PIF-NEGOCIACION", self.clock+1.0, self.father, self.id, event.package)
                        self.transmit(newevent)
                        #print ("soy",self.id,"TERMINO y envio",newevent.name,"hacia mi papa",newevent.target,"en el tiempo",newevent.time)
                        self.reiniciaPIF()
                    else:
                        self.reiniciaPIF()
                        #print("soy",self.id,"COORDINADOR y declaro terminada la fase de NEGOCIACION, ahora inicio el PIF-CONEXION ciclo:",self.contadorCiclos)
                        newevent = Event("PIF-CONEXION", self.clock+1.0, self.id, self.id, None)
                        self.transmit(newevent)
        
        elif(event.name=="RECHAZO-CONEXION"):
            if(len(event.package.ruta)>0):#si el mensaje no viene para mi, simplemente lo reenvio
                nextStep=event.package.ruta[0]
                del event.package.ruta[0]
                newevent = Event("RECHAZO-CONEXION", event.time + 1.0, nextStep, self.id, event.package)        
                self.transmit(newevent)
                #print("soy",self.id,"reenvio el mensaje",newevent.name,"a",newevent.target,"proveniente de",event.source,"en el tiempo",newevent.time,"ruta faltante",event.package.ruta)
            else:#si el mensaje si viene para mi:
                #print("soy",self.id,"ya llego",event.name,"que venia de",event.package.rutaAuxiliar[len(event.package.rutaAuxiliar)-1],"en el tiempo",event.time)
                self.numeroSolicitudes-=1#decremento el numero de solicitudes, cuando este llegue a cero quiere decir que ya acabe el recableado
            
            if(self.numeroSolicitudes==0):
                if self.count == 0:
                    if(self.father!=self.id):
                        newevent = Event("PIF-NEGOCIACION", self.clock+1.0, self.father, self.id, event.package)
                        self.transmit(newevent)
                        #print ("soy",self.id,"TERMINO y envio",newevent.name,"hacia mi papa",newevent.target,"en el tiempo",newevent.time)
                        self.reiniciaPIF()
                    else:
                        self.reiniciaPIF()
                        #print("soy",self.id,"COORDINADOR y declaro terminada la fase de NEGOCIACION, ahora inicio el PIF-CONEXION ciclo:",self.contadorCiclos)
                        newevent = Event("PIF-CONEXION", self.clock+1.0, self.id, self.id, None)
                        self.transmit(newevent)   
        
        #PIF fase de CONEXION--------------
        elif(event.name=="PIF-CONEXION"):
            self.count-=1
            if(self.visited==False):
                #print("soy",self.id,"recibo",event.name,"por primera vez, viene de",event.source,"en el tiempo",event.time)
                self.visited=True
                self.father=event.source
                #antes de propagar mi PIF-CONEXION, empiezo mi fase de CABLEADO-RECABLEADO
                #print("soy",self.id,"mis self.neighborsPendientes son",self.neighborsPendientes,"vecinos,",self.neighbors,"vecinos dinamicos",self.vecinosConectadosDinamicos)
                if(len(self.neighborsPendientes)>0):#lista de vecinos que voy a agregar en la etapa de recableado, se da en los nodos destinos (los que aceptan solicitudes)
                    for i in self.neighborsPendientes:
                        self.neighbors.append(i)
                #print("soy",self.id,"mis self.neighborsPendientesEliminacion son",self.neighborsPendientesEliminacion,"vecinos,",self.neighbors,"vecinos dinamicos",self.vecinosConectadosDinamicos)
                if(len(self.neighborsPendientesEliminacion)>0):#lista de vecinos que voy a eliminar en la etapa de recableado
                    for i in self.neighborsPendientesEliminacion:
                        self.neighbors.remove(i)                         
                #print("soy",self.id,"mis vecinos quedan asi",self.neighbors,"vecinos dinamicos",self.vecinosConectadosDinamicos)
                if(len(self.solicitudesPendientes)>0):#lista de solicitudes que tengo que atender en la etapa de recableado, se da en los nodos origen (los que mandan solicitudes)
                    #ahora hago la conexión de mi enlace dinámico con el nodo que acepto la conexión
                    #busco mi enlace dinámico que hizo la solicitud
                    for package in self.solicitudesPendientes:
                        enlace = self.enlacesDinamicosDict[package.idEnlace]
                        #una vez que ya encontre el enlace, establezco la conexión
                        enlace.libre=False#el enlace ahora ya no esta libre
                        if(enlace.idConectado!=-1):
                            self.vecinosConectadosDinamicos.remove(enlace.idConectado)
                            print("r ",self.id," ",enlace.idConectado," ",package.rutaAuxiliar[len(package.rutaAuxiliar)-1]," ",self.contadorCiclos)
                            #actualizo el grafo networkx de la simulacion:
                            #print("soy",self.id,"DESCONEXION mis vecinos en el grafo antes del cambio son",list(self.__main.graph.neighbors(self.id)))
                            #print("aristas antes del cambio:",list(self.__main.graph.edges()))
                            self.__main.graph.remove_edge(int(self.id),int(enlace.idConectado))
                            #print("soy",self.id,"DESCONEXION con",enlace.idConectado,"ahora mis vecinos en el grafo son",list(self.__main.graph.neighbors(self.id)))
                            #print("aristas despues del cambio:",list(self.__main.graph.edges()))
                        else:
                            print("c ",self.id," ",-1," ",package.rutaAuxiliar[len(package.rutaAuxiliar)-1]," ",self.contadorCiclos)
                        #actualizo el grafo networkx de la simulacion:
                        #print("soy",self.id,"CONEXION mis vecinos en el grafo antes del cambio son",list(self.__main.graph.neighbors(self.id)))
                        #print("aristas antes del cambio:",list(self.__main.graph.edges()))
                        self.__main.graph.add_edge(int(self.id),int(package.rutaAuxiliar[len(package.rutaAuxiliar)-1]))
                        #print("soy",self.id,"CONEXION con",package.rutaAuxiliar[len(package.rutaAuxiliar)-1],"ahora mis vecinos son",list(self.__main.graph.neighbors(self.id)))
                        #print("aristas despues del cambio:",list(self.__main.graph.edges()))
                        enlace.idConectado=package.rutaAuxiliar[len(package.rutaAuxiliar)-1]
                        self.vecinosConectadosDinamicos.append(enlace.idConectado)
                        #print("soy",self.id,"mi enlace numero",enlace.idEnlace,"ya esta libre=",enlace.libre,"y me conecta con",enlace.idConectado)
                        #print("soy",self.id,"mis vecinos van asi:",self.neighbors+self.vecinosConectadosDinamicos)

                #propago mi PIF-CONEXION
                conjuntoVecinos=self.neighbors+self.vecinosConectadosDinamicos
                for i in conjuntoVecinos:
                    if i != self.father:
                        newevent=Event("PIF-CONEXION",self.clock+1.0,i,self.id,None)
                        self.transmit(newevent)
                        self.count+=1
                
            if self.count == 0:
                if(self.father!=self.id):
                    newevent = Event("PIF-CONEXION", self.clock+1.0, self.father, self.id, event.package)
                    self.transmit(newevent)
                    #print("soy",self.id,"TERMINO y envio",newevent.name,"hacia mi papa",newevent.target,"en el tiempo",newevent.time,"conexionesAceptadas",self.conexionesAceptadas)
                    #print("soy",self.id,"mis neighbors son",self.neighbors,"mis vecinosConectadosDinamicos son",self.vecinosConectadosDinamicos)
                    #print("soy",self.id,"mis neighbors en el objeto grafo networkx son",list(self.__main.graph.neighbors(self.id)))
                    #for i in self.enlacesDinamicos:
                        #print("soy",self.id,"mi enlace con id",i.idEnlace,"esta conectado con",i.idConectado)
                    self.contadorCiclos+=1
                    self.reiniciaPIF()
                    self.reiniciaAtributosParaCicloNuevo()
                else:
                    self.reiniciaPIF()
                    self.reiniciaAtributosParaCicloNuevo()
                    #print("soy",self.id,"COORDINADOR y declaro terminada la fase de CONEXION ciclo:",self.contadorCiclos,"conexionesAceptadas",self.conexionesAceptadas)
                    #print("soy",self.id,"mis neighbors son",self.neighbors,"mis vecinosConectadosDinamicos son",self.vecinosConectadosDinamicos)
                    #print("soy",self.id,"mis neighbors en el objeto grafo networkx son",list(self.__main.graph.neighbors(self.id)))
                    #for i in self.enlacesDinamicos:
                        #print("soy",self.id,"mi enlace con id",i.idEnlace,"esta conectado con",i.idConectado)
                    #nx.write_adjlist(self.__main.graph,"graph"+str(self.contadorCiclos)+".adjlist")   
                    if(self.contadorCiclos<self.__main.ciclos):
                        new_package=Paquete(self.id,0)
                        new_package.diametro=nx.diameter(self.__main.graph)
                        if not(new_package.diametro==2):
                            newevent = Event("PIF-EXPLORACION", self.clock+1.0, self.id, self.id, new_package)
                            self.transmit(newevent)
                            self.contadorCiclos+=1
                            #print("soy",self.id,"COORDINADOR inicio el ciclo numero",self.contadorCiclos,"arrancando el PIF-EXPLORACION")
                        #else:
                            #print("soy",self.id,"COORDINADOR detuve la simulacion por que el diametro ya es 2")
                    #else:
                        #print("soy",self.id,"COORDINADOR doy por terminada la simulacion")
        
    def creaLongitudEnlacesDinamicos(self):
        for i in range(self.__numEnlacesDinamicos):
            enlace=Enlace(i,self.__main.tamEnlace)#aqui se puede cambiar facilmente la longitud de cada enlace
            self.f_e[enlace.idEnlace]=0#inicializo la frecuencia de uso del enlace dinamico en cero
            self.enlacesDinamicos.append(enlace)
            self.enlacesDinamicosDict[enlace.idEnlace]=enlace

    def reiniciaPIF(self):
        self.visited=False
        self.father=-1
        self.count=1

    def reiniciaAtributosParaCicloNuevo(self):
        self.paquetes.clear()
        self.paquetesRegreso=0
        self.f_n.clear()
        #reincio la frecuencia de mi enlaces dinamicos
        claves=self.f_e.keys()
        for i in claves:
            self.f_e[i]=0
        self.listaNodosSolicitados.clear()
        self.numeroSolicitudes=0
        self.neighborsPendientes.clear()
        self.neighborsPendientesEliminacion.clear()
        self.solicitudesPendientes.clear()
        self.__encaminamiento.clearShortestPathCache()