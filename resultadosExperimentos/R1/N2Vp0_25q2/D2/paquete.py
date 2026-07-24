
class Paquete:

	def __init__(self,sourceID,packageID):
		self.__sourceID=sourceID#identificador del nodo que genera el paquete
		self.__idPaquete=packageID#identificador del paquete
		self.__idEnlace=-1#id del enlace dinámico que esta haciendo una solicitud
		self.__ruta=[]#ruta por donde viajará el paquete
		self.__distanciaMaxima=-1#distancia máxima a la que viajará el paquete usando random walk
		self.__destino=-1#nodo destino al que llegará el paquete usando compass routing
		self.__rutaAuxiliar=[]#ruta que me sirve para regresar el paquete al origen
		self.__diametro=-1#diámetro nuevo de la red en cada ciclo de simulación, se usa en random walk


	@property
	def idPaquete(self):
		return self.__idPaquete

	@property
	def sourceID(self):
		return self.__sourceID

	@sourceID.setter
	def sourceID(self,value):
		self.__sourceID=value

	@property
	def idEnlace(self):
		return self.__idEnlace

	@idEnlace.setter
	def idEnlace(self,value):
		self.__idEnlace=value

	@property
	def ruta(self):
		return self.__ruta

	@ruta.setter
	def ruta(self,value):
		self.__ruta=value

	@property
	def distanciaMaxima(self):
		return self.__distanciaMaxima

	@distanciaMaxima.setter
	def distanciaMaxima(self,value):
		self.__distanciaMaxima=value

	@property
	def destino(self):
		return self.__destino

	@destino.setter
	def destino(self,value):
		self.__destino=value

	@property
	def rutaAuxiliar(self):
		return self.__rutaAuxiliar

	@rutaAuxiliar.setter
	def rutaAuxiliar(self,value):
		self.__rutaAuxiliar=value

	@property
	def diametro(self):
		return self.__diametro

	@diametro.setter
	def diametro(self,value):
		self.__diametro=value