
import sys
import networkx as nx

def getProperties(G):
    avCl = nx.average_clustering(G)
    nCom = nx.number_connected_components(G)
    order = G.order()
    diam = nx.diameter(G)
    avSPL = nx.average_shortest_path_length(G)
    properties = [avCl,nCom,order,diam,avSPL]
    return properties

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please supply a file name")
        raise SystemExit(1)
    #f = open(sys.argv[1],'r',encoding='utf-16-le')  #windows terminal
    f = open(sys.argv[1],'r',encoding='utf-8')  #linux terminal
    lines = f.readlines()
    f.close()
    fd = open("datos-"+sys.argv[1],'w')
    fd.write('#cicle\tavCl\tcomponents\tdiam\taspl\torder\n')
    G = nx.read_adjlist("graph.adjlist", nodetype=int)
    #Get average clustering, diameter and average shortests path length
    data = getProperties(G)
    cicloAnterior=str(0)
    for line in lines:
        accion,fuente,desconecta,conecta,ciclo = line.split()
        if(ciclo != cicloAnterior):
            nx.write_adjlist(G,"graph_"+sys.argv[2]+"_"+cicloAnterior+".adjlist")
            fh = open("hist_"+sys.argv[2]+"_"+cicloAnterior+".txt",'w')
            hist = nx.degree_histogram(G)
            j = 0
            for i in hist:
                fh.write(str(j)+'\t'+str(i)+'\t'+str(i/2500)+'\n')
                j = j+1
            fh.close()
            data = getProperties(G)
            fd.write(cicloAnterior+'\t')
            fd.write('{0:.6f}\t'.format(data[0]))
            fd.write(str(data[1])+'\t')
            fd.write(str(data[3])+'\t')
            fd.write('{0:.6f}\t'.format(data[4]))
            fd.write(str(data[2])+'\n')
        cicloAnterior=ciclo
        if (accion=="r"):   # Disconnection
            G.remove_edge(int(fuente),int(desconecta))
        G.add_edge(int(fuente),int(conecta))#Connection
    data = getProperties(G)
    fd.write(cicloAnterior+'\t')
    fd.write('{0:.6f}\t'.format(data[0]))
    fd.write(str(data[1])+'\t')
    fd.write(str(data[3])+'\t')
    fd.write('{0:.6f}\t'.format(data[4]))
    fd.write(str(data[2])+'\n')
    fd.close()
    fh = open("hist_"+sys.argv[2]+"_"+cicloAnterior+".txt",'w')
    hist = nx.degree_histogram(G)
    j = 0
    for i in hist:
        fh.write(str(j)+'\t'+str(i)+'\n')
        j=j+1
    fh.close()
    nx.write_adjlist(G,"graph_"+sys.argv[2]+"_"+cicloAnterior+".adjlist")