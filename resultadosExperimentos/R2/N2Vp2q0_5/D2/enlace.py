
class Enlace:

	def __init__(self,idEnlace,value):
		self.__idEnlace=idEnlace
		self.__longitud=value
		self.__libre=True
		self.__idConectado=-1

	@property
	def idEnlace(self):
		return self.__idEnlace

	@idEnlace.setter
	def idEnlace(self,value):
		self.__idEnlace=value

	@property
	def longitud(self):
		return self.__longitud

	@longitud.setter
	def longitud(self,value):
		self.__longitud=value

	@property
	def libre(self):
		return self.__libre

	@libre.setter
	def libre(self,value):
		self.__libre=value

	@property
	def idConectado(self):
		return self.__idConectado

	@idConectado.setter
	def idConectado(self,value):
		self.__idConectado=value