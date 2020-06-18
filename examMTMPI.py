# Marco Barbiero - QAP - 200220
# Works with Python 3.7.3 64-bit
# coding=UTF8

# Try using MPI - [MPI for Python](https://mpi4py.readthedocs.io/en/stable/)


# Imports
import numpy as np
import sys

import time
# importing datetime module for now()
import datetime

import os

# parte parallel
from mpi4py import MPI
import time

# Problem definition: There are a set of n facilities and a set of n locations. For each pair of locations, a distance is specified and for each pair of facilities a weight or flow is specified (e.g., the amount of supplies transported between the two facilities). The problem is to assign all facilities to different locations with the goal of minimizing the sum of the distances multiplied by the corresponding flows.
# Intuitively, the cost function encourages factories with high flows between each other to be placed close together. => Problema di minimo.
# Target: Find the bijection f : P → L, ovvero dire che facility metto al nodo i-esimo.
# La matrice A definisce i flussi, la matrice B la distanza
# si lavora a permutazioni, quindi la soluzione è dello stesso tipo del TSP.

# region Functions

"""Return the cost of a path (facilityNelNodo)
    Args:
        facilityNelNodo: list of facilities ordered by their positions (es. facilityNelNodo[i] = j means that facility j is located in node i)
        flussi: matrix of fluxes between facilities
        distanze: matrix of distances between nodes

    Returns:
        the cost!
"""


def costo(facilityNelNodo: list, flussi: np.ndarray, distanze: np.ndarray):
    N = len(facilityNelNodo)
    # if i!=j -> il problema include anche i==j (ma la distanza non dovrebbe essere 0? No, qui è diversa da zero!)
    return sum([flussi[facilityNelNodo[i]][facilityNelNodo[j]]*distanze[i][j] for i in range(N) for j in range(N)])


"""Return the cost difference between a path (percorso) and a opt-2 permutation defined by (pos1, pos2)
    Args:
        facilityNelNodo: list of facilities ordered by their positions (es. facilityNelNodo[i] = j means that facility j is located in node i)
        flussi: matrix of fluxes between facilities
        distanze: matrix of distances between nodes
        pos1, pos2: the nodes that have to be swapped

    Returns:
        the difference of cost (>0 means the permutation is worse)

"""


def variazioneCosto(facilityNelNodo: list, flussi: np.ndarray, distanze: np.ndarray, pos1: int, pos2: int):
    init = costo(facilityNelNodo, flussi, distanze)
    facilityNelNodo2 = facilityNelNodo.copy()
    facilityNelNodo2[pos1] = facilityNelNodo[pos2]
    facilityNelNodo2[pos2] = facilityNelNodo[pos1]
    end = costo(facilityNelNodo2, flussi, distanze)
    return end - init


"""Return the cost of a opt-2 permutation defined by (pos1, pos2)
    Args:
        facilityNelNodo: list of facilities ordered by their positions (es. facilityNelNodo[i] = j means that facility j is located in node i)
        flussi: matrix of fluxes between facilities
        distanze: matrix of distances between nodes
        pos1, pos2: the nodes that have to be swapped

    Returns:
        the new cost of the path
"""


def variazioneCostoOPT(facilityNelNodo: list, flussi: np.ndarray, distanze: np.ndarray, pos1: int, pos2: int):
    # init = costo(facilityNelNodo, flussi, distanze)
    facilityNelNodo2 = facilityNelNodo.copy()
    facilityNelNodo2[pos1] = facilityNelNodo[pos2]
    facilityNelNodo2[pos2] = facilityNelNodo[pos1]
    end = costo(facilityNelNodo2, flussi, distanze)
    return end


"""Return the best permutation according to Greedy
    Args:
        flussi: matrix of fluxes between facilities
        distanze: matrix of distances between nodes
        start: initial facility for Greedy alg.
        random: True if GRASP approach desired
        TOPLEN: length of best nodes list for GRASP
        optimization: True for a distance-based probability distribution, False for an uniform distribution, in GRASP

    Returns:
        the best list of facilities ordered by their positions (es. facilityNelNodo[i] = j means that facility j is located in node i) using Greedy Approach (or GRASP)
"""


def GreedyRandom(flussi: np.ndarray, distanze: np.ndarray, start: int, random: bool = False, TOPLEN: int = 3, optimization: bool = True):
    N = len(flussi)
    # print(str(start))
    # params
    # TOPLEN = 5 # massimo numero top
    # init
    if(not random):
        TOPLEN = 1
    if(N < 2):
        print("Non ci sono abbastanza punti!")
        return start
    puntiRimasti = [i for i in range(0, N)]  # in valore di indice
    # print (str(puntiRimasti))
    facilityNelNodo = []  # come coppia tupla - NO! Ora metto come permutazione
    # add start
    facilityNelNodo.append(int(start))
    puntiRimasti.remove(start)
    pI = start
    # log
    newRoute = 0
    while(len(puntiRimasti) > 1):
        # 1. trovo top 5 min dist
        top5 = -np.ones([(TOPLEN+1)])
        top5dist = sys.maxsize*np.ones([TOPLEN+1])
        added = 0
        for i in range(0, len(puntiRimasti)):
            # dista: la cosa da minimizzare, ovvero prendo quello che mi porta la dista minore. Sembra andare meglio il prodotto dei due
            # dista = costoSingolo(punti[int(pI)], punti[puntiRimasti[i]])
            # dista = distanze[int(pI)][puntiRimasti[i]]/(flussi[int(pI)][puntiRimasti[i]]+0.0001)
            # dista = (flussi[int(pI)][puntiRimasti[i]])/distanze[int(pI)][puntiRimasti[i]]
            dista = (flussi[int(pI)][puntiRimasti[i]]) * \
                distanze[int(pI)][puntiRimasti[i]]
            j = TOPLEN
            # j-1 oppure mi mangio sempre uno
            while j > 0 and top5dist[j-1] > dista:
                top5[j] = top5[j-1]
                top5dist[j] = top5dist[j-1]
                j = j-1
            if(j < TOPLEN):
                top5[j] = puntiRimasti[i]
                top5dist[j] = dista
                added = added + 1
                # 2. add random of
        if(added < 1):
            print(added)
        if(random):
            # best = top5[np.random.randint(0, min(TOPLEN, added))] # serve un po' di ottimizzazione sulla ricerca
            if (optimization):
                prob = [1/(d+.01)**4 for d in top5dist[0:min(TOPLEN, added)]]
                tot = sum(prob)
                prob = [p/tot for p in prob]
                best = top5[np.random.choice(
                    np.arange(0, min(TOPLEN, added)), p=prob)]
            else:
                # serve un po' di ottimizzazione sulla ricerca
                best = top5[np.random.randint(0, min(TOPLEN, added))]

            if(top5[0] != best):
                # print("Prendo altra strada: " +
                #       str(top5[0]) + " | " + str(best))
                newRoute = newRoute+1
        else:
            best = top5[0]
        # print(best)
        facilityNelNodo.append(int(best))
        puntiRimasti.remove(best)
        pI = best  # aggiorno il punto di partenza

    facilityNelNodo.append(puntiRimasti[0])
    if (False and newRoute > 0):
        print("Probabilità nuovi Percorsi: " +
              str(newRoute*100/len(punti)) + "%")
    # print("Parto da "+str(start)+"")
    return facilityNelNodo


"""Return the best permutation according to Greedy using Local Search (opt-2)
    Args:
        flussi: matrix of fluxes between facilities
        distanze: matrix of distances between nodes
        start: initial facility for Greedy alg.
        LOCAL_SEARCH: True to enable local search opt-2
        random: True if GRASP approach desired
        TOPLEN: length of best nodes list for GRASP
        optimization: True for a distance-based probability distribution, False for an uniform distribution, in GRASP

    Returns:
        the best list of facilities ordered by their positions (es. facilityNelNodo[i] = j means that facility j is located in node i) using Greedy Approach (or GRASP)
"""


def GreedyPLS(flussi: np.ndarray, distanze: np.ndarray, start: int, LOCAL_SEARCH: bool = True, random: bool = False, TOPLEN: int = 3, optimization: bool = True):
    N = len(flussi)
    facilityNelNodo = GreedyRandom(
        flussi, distanze, start, random, TOPLEN, optimization)
    # parte LS
    while(LOCAL_SEARCH):  # not oldSol == (facilityNelNodo)):
        nowCost = costo(facilityNelNodo, flussi, distanze)
        swap = min([(i, j) for i in range(0, N) for j in range(0, N) if i < j], key=lambda v: variazioneCostoOPT(
            facilityNelNodo, flussi, distanze, v[0], v[1]))  # if i!= j # spesso si interrompe alla prima soluzione migliore (così taglio l'N^2)
        if(variazioneCostoOPT(facilityNelNodo, flussi, distanze, swap[0], swap[1]) >= nowCost):
            # print("Non Migliorato! :cry - Nuovo giro!") # meglio non correre rischi
            return facilityNelNodo
        temp = facilityNelNodo[swap[0]]
        facilityNelNodo[swap[0]] = facilityNelNodo[swap[1]]
        facilityNelNodo[swap[1]] = temp
    return facilityNelNodo

# endregion


# Function to summarize all - ricordarsi che si hanno un master (0 - le scritture sono tutte sue) e degli slave nella concezione più comoda
def doFun(sfida, LOCAL_SEARCH, MAX_ITER, RANDOM, MAX_LIST, PARALLEL, filePath, nomeFilePerSoluzioni='soluzioni.txt', OPTIMIZATION=True, OLD_PARSER=False):
    if(not RANDOM):
        MAX_ITER = 1
        MAX_LIST = 1

    # filePath = "C:\\Users\\marcobarbiero\\OneDrive\\Documenti\\Dottorato\\Corsi\\IE-4-2020\\Application\\"

    fileName = filePath + "data\\" + sfida + ".dat"
    fileSoluzione = filePath + "solutions\\" + sfida + ".sln"

    # parte MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    sizeC = comm.Get_size()
    ultimoThread = sizeC-1
    thread0 = 0
    thread1 = 2  # 2

    print("Io sono il thread " + str(rank) + " di " + str(sizeC) + " e sto facendo la sfida: " + sfida)

    # Guardo se c'è la soluzione....
    go = False

    if rank == thread0:
        try:
            f = open(fileSoluzione, "r")
            fl = f.readlines()
            f.close()
            solLoro = []
            for i in range(1, len(fl)):
                row = fl[i]
                splits = row.split(' ')
                # print(splits)
                [solLoro.append(int(val))
                 for val in splits if val != '' and val != '\n']
            solLoro = [solLoro[i]-1 for i in range(0, len(solLoro))]
            go = True
            comm.send(solLoro, dest=ultimoThread)

        except:
            print("Non c'è la soluzione! Salto :wink")
            # devo gestire questa cosa altrimenti i core stanno ad aspettarmi all'infinito!

    go = comm.bcast(go, root=thread0)  # caso scalare

    if(not go):
        print("Niente Soluzione :cry - prossima sfida!")
        return

    if rank == ultimoThread:
        solLoro = comm.recv(source=0)
        print("Soluzione Letta: ", end="")
        print(solLoro)

    # Apro il file del problema - core 1 - dichiara tutte le variabili che ti servono sincronizzate altrimenti non va...
    N = None
    flussi = None
    distanze = None
    if rank == thread1:
        f = open(fileName, "r")
        fl = f.readlines()
        f.close()
        n = int(fl[0])
        print("Dimensione: " + str(n))

        A = np.empty((n, n))
        B = np.empty((n, n))

        N = n

        rigaInCorsoA = 0
        rigaInCorsoB = 0
        temp = 0  # variabile per ricordarmi se ho lasciato a metà una riga
        for i in range(1, len(fl)):
            # devo capire se sono in A o B
            row = fl[i]
            if(len(fl) < 2):
                print('Riga Nulla')
                continue
            # sono in A
            splits = row.split(' ')
            # elimino errori di allineamento
            splits = list(filter(lambda x: x != '' and x !=
                                 '\t' and x != '\n', splits))
            if len(splits) == 0:  # riga inutile
                continue
            if(rigaInCorsoA < N):
                for j in range(0, len(splits)):
                    A[rigaInCorsoA][j+temp] = float(splits[j])
                if(len(splits)+temp >= N):
                    temp = 0
                    rigaInCorsoA = rigaInCorsoA+1
                else:
                    temp = temp + len(splits)
            else:
                # sono in B
                for j in range(0, len(splits)):
                    B[rigaInCorsoB][j+temp] = float(splits[j])
                if(len(splits)+temp >= N):
                    temp = 0
                    rigaInCorsoB = rigaInCorsoB+1
                else:
                    temp = temp + len(splits)

        print('Righe A: ' + str(rigaInCorsoA))
        flussi = B
        distanze = A
        print('Righe B: ' + str(rigaInCorsoB))

    # sincronizzo i valori
    # comm.Barrier()
    n = comm.bcast(N, root=thread1)  # caso scalare
    # flussi = comm.Bcast([flussi, MPI.INT], root=thread1)  # caso numpy
    # distanze = comm.Bcast([distanze, MPI.INT], root=thread1)  # caso numpy

    # caso numpy - sembra che il normale vada bene lo stesso
    flussi = comm.bcast(flussi, root=thread1)
    # caso numpy - sembra che il normale vada bene lo stesso
    distanze = comm.bcast(distanze, root=thread1)

    # print("Io sono il thread " + str(rank) + " e il mio n: " + str(n) + ". Il mio flusso è: " + str(flussi))

    # siamo tutti sincronizzati... Si parte! Il rank 0 è quello che coordina ma comunque come gli altri invece fa i conti

    tic = time.perf_counter()

    # do something

    bestSol = []
    bestCost = sys.maxsize
    # primo passaggio non random
    for c in range(len(distanze)):  # potrei usare uno scatter, più intelligente di questo imho
        if(c % sizeC == rank):  # se è il mio turno di lavorare
            facilityNelNodo = GreedyPLS(flussi, distanze, c, LOCAL_SEARCH, False, MAX_LIST)
            nowCost = costo(facilityNelNodo, flussi, distanze)
            # print(str(c) + ": " + str(facilityNelNodo) + " -> " + str(nowCost))
            if(nowCost < bestCost):
                # print("SWAP")
                bestSol = facilityNelNodo
                bestCost = nowCost

    if(RANDOM):
        # allora faccio anche random
        for c in range(len(distanze)):  # potrei usare uno scatter, più intelligente di questo imho
            if(c % sizeC == rank):  # se è il mio turno di lavorare
                facilityNelNodo = GreedyPLS(flussi, distanze, c, LOCAL_SEARCH, RANDOM, MAX_LIST)
                nowCost = costo(facilityNelNodo, flussi, distanze)
                # print(str(c) + ": " + str(facilityNelNodo) + " -> " + str(nowCost))
                if(nowCost < bestCost):
                    # print("SWAP")
                    bestSol = facilityNelNodo
                    bestCost = nowCost
    toc = time.perf_counter()

    # ora raccogliamo tutte le soluzioni e teniamo la minore - se ne occupa l'ultimo thread mentre gli altri aspettano (in teoria l'ultimo thread è quello che ha più probabilità di finire prima (finirebbe come gli altri solo nel caso c sia multiplo della dimensione del comm))

    print(f"Done in {toc - tic:0.4f} seconds")
    print("Sono il core " + str(rank) + " e ho appena finito! La mia soluzione: " + str(bestCost))
    # debug
    bestSolArr = comm.gather(bestSol, root=ultimoThread)
    bestCostArr = comm.gather(bestCost, root=ultimoThread)

    # conti finali
    if rank == ultimoThread:
        # Tratto:
        bestCost = min(bestCostArr)
        migliore = bestCostArr.index(bestCost)
        print("Best Cost: " + str(bestCostArr) + " -> " + str(bestCost) + " (ovvero il " + str(migliore) + ")")
        bestSol = bestSolArr[migliore]
        loroCost = costo(solLoro, flussi, distanze)
        print("\n\nRISULTATI per " + sfida)
        print("Io: " + str(bestSol) + " => "+str(bestCost))
        print("Loro: " + str(solLoro) + " => " +
              str(loroCost))
        # fix per quando succede che il costo è uguale a 0...
        if(loroCost == 0):
            loroCost2 = 1e-10
        else:
            loroCost2 = loroCost
        if(bestCost == 0):
            bestCost2 = 1e-10
        else:
            bestCost2 = bestCost

        print("Errore nel costo: " + str(100*bestCost2 /
                                         loroCost2-100) + "%")

        # scrivo sul file
        outputFile = nomeFilePerSoluzioni
        f = open(outputFile, "a+")
        print("\n\nRISULTATI per " + sfida, file=f)
        print("Io: " + str(bestSol) + " => "+str(bestCost), file=f)
        print("Loro: " + str(solLoro) + " => " +
              str(costo(solLoro, flussi, distanze)), file=f)
        print("Errore nel costo: " + str(100*bestCost2 /
                                         loroCost2-100) + "%", file=f)
        # print(f"Done in {toc - tic:0.4f} seconds", file=f)
        print(f"I core hanno prodotto: {bestCostArr}", file=f)
        print(
            f"Settings: LOCAL_SEARCH = {LOCAL_SEARCH}, MAX_ITER = {MAX_ITER}, RANDOM = {RANDOM}, MAX_LIST = {MAX_LIST}, PARALLEL = {PARALLEL}. DateTime = {datetime.datetime.now()}.", file=f)
        f.close()
    comm.barrier()

# main: - mi immagino di avere 4 CPU come minimo


def main():
    # ora ho matrici A e B -> devo fare solo l'algoritmo :sweat
    # print command line arguments
    sfida = sys.argv[1]
    if (sfida.lower() == "all"):
        sfida = ""
    if(len(sys.argv) < 3):
        doveCercareLeSfide = "I:\\SkyDrive\\Documenti\\Dottorato\\Corsi\\IE-4-2020\\Application\\"
    else:
        doveCercareLeSfide = sys.argv[2]
    if(len(sys.argv) < 4):
        FILE_SOL = "soluzioniCAPRI.txt"
    else:
        FILE_SOL = sys.argv[3]
    print("Sono il core" + str(MPI.COMM_WORLD.Get_rank()) + " in partenza per la sfida " + sfida + " nella cartella " + doveCercareLeSfide + " e scriverò tutto in " + FILE_SOL)
    LOCAL_SEARCH = True
    MAX_ITER = 10  # 0
    RANDOM = True
    MAX_LIST = 3
    PARALLEL = True
    OPTIMIZATION = True  # False
    PAZZIA = True  # attiva tutte le sfide che cominciano con sfida!
    # OLD_PARSER = False

    # end PARAMS

    # doveCercareLeSfide = "C:\\Users\\marcobarbiero\\OneDrive\\Documenti\\Dottorato\\Corsi\\IE-4-2020\\Application\\data\\"

    if(PAZZIA):
        initSfida = sfida
        for file in os.listdir(doveCercareLeSfide+"\\data\\"):
            if file.startswith(initSfida):
                if file.endswith(".dat"):
                    sfida = (file[0:-4])
                    # print("Sto facendo la sfida: " + sfida)
                    doFun(sfida, LOCAL_SEARCH, MAX_ITER, RANDOM,
                          MAX_LIST, PARALLEL, doveCercareLeSfide, FILE_SOL, OPTIMIZATION)

    else:
        doFun(sfida, LOCAL_SEARCH, MAX_ITER, RANDOM, MAX_LIST,
              PARALLEL, doveCercareLeSfide, FILE_SOL, OPTIMIZATION)
    print("Sono il core" + str(MPI.COMM_WORLD.Get_rank()) + " e ho finito alle ore " + str(datetime.datetime.now()) + "!")


if __name__ == '__main__':
    main()
