
""" An instance of "Event" encapsulates the information exchanged between the
active entities (processes) of a distributed system. """

#---------------------------------------------
class Event:                               # Descends from "Object" (default)
    """ Event attributes: "name", "time", "target", "source", & "port", 
    contains constructor and getters in python 3 style. """
    
    def __init__(self, name, time, target, source, package ,port=0):
        """ Builds an instance of "Event". """
        self.__name = name
        self.__time = time
        self.__target = target
        self.__source = source
        self.__package = package
        self.__port = port

    @property
    def name(self):
        """ Invoke as x.name, without "()". """
        return self.__name

    @property
    def time(self):
        """ Invoke as x.time, without "()". """
        return self.__time

    @property
    def target(self):
        """ Invoke as x.target, without "()". """
        return self.__target

    @property
    def source(self):
        """ Invoke as x.source, without "()". """
        return self.__source

    @property
    def package(self):
        """ Invoke as x.package, without "()". """
        return self.__package

    @property
    def port(self):
        """ Invoke as x.port, without "()". """
        return self.__port

    """método setter del artibuto event, que permite que la capa cero sea capaz de modificar dicho atributo en los mensajes generados
    en capas superiores"""
    @time.setter    # the property decorates with `.setter` now
    def time(self, value):   # name, e.g. "attribute", is the same
        self.__time = value   # the "value" name isn't special
