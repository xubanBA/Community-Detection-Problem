
### Aurreko faseko kodigoa ###

# SQL
import sqlite3

# Pandas
import pandas as pd

# Graph
import community
import networkx as nx

# Plot
import matplotlib.pyplot as plt
import seaborn as sns

# Combinations
import itertools

import numpy as np
import math
from operator import itemgetter


# pip install ipynb exekutatu liburutegi hau erabili ahal izateko.
from itertools import product
#from ipynb.fs.full.CDP_Sarrera_Ikasle import sortu_grafoa 
import community
import networkx as nx
import numpy as np
np.set_printoptions(threshold=20)


### Helburu-funtzioa
def modularitatea(G, partizioa, weight='weight'):
    
    ### Bete hemen 15-20 lerro
    e = dict([]) # eii
    a = dict([]) # aii
    m = G.size()
    modul = 0
    
    for v in G:
        com = partizioa[v] # u ze komunitatean dagoen
        a[com] = a.get(com,0) + G.degree(v, weight=weight) # u dagoen komunitatearen nodo bakoitzaren degree gehitzen joan
        
        for w, datas in G[v].items(): # u nodoaren auzokidea
            if partizioa[w] == com: # u eta w partizio berekoak
                if v == w:
                    e[com] = e.get(com, 0) + datas[weight] * 2
                else:
                    e[com] = e.get(com, 0) + datas[weight]     
    
    for com in set(partizioa.values()):
        modul += (e.get(com, 0) / (2*m)) - (a.get(com, 0) / (2*m)) ** 2

    return modul


''' 
def sortu_grafoa():
    
    # Datuak irakurri
    # BETE HEMEN 8 lerro
    connect = sqlite3.connect('./data/database.sqlite')
    query = """
    SELECT pa.paper_id, pa.author_id, a.name
    FROM paper_authors AS pa JOIN papers AS p ON pa.paper_id = p.id
    JOIN authors as a ON pa.author_id = a.id
    WHERE p.Year BETWEEN '2014' AND '2015'
    """
    df = pd.read_sql(query, connect)
    
    # Sortu grafoa
    # BETE HEMEN 7-10 lerro
    
    # Initialize graph
    G = nx.Graph()

    # Transform
    # Autorearen IDa erabili beharrean erabili izena.
    for p, a in df.groupby('paper_id')['name']: 
        for u, v in itertools.combinations(a, 2):
            if G.has_edge(u, v):
                G[u][v]['weight'] +=1
            else:
                G.add_edge(u, v, weight=1)
                
    ######### Grafoa murrizteko ########
    original_len = G.number_of_nodes() 
    deleted_node = [] 
    deleted = [] # ezabatuen indizea gorde gero berriro gehitzeko     
    for i, elem in enumerate(G):
        if G.degree(elem) < 5:
            deleted.append(i)
            deleted_node.append(elem)
    G.remove_nodes_from(deleted_node) # ezabatu
    ####################################
            
    # Print graph size 
    print('\nAutore kopurua grafoan:', G.number_of_nodes())
    print('\nElkarlan kopurua grafoan:', G.number_of_edges())
    
    return G, deleted, original_len
'''

def osatu_soluzioa(best_solution, deleted, original_len, numCom):
    sol = []
    cont = 0
    for i in range(original_len): 
        if i in deleted: # ez da borratu nodo
            sol.append(numCom+1) # bestela: sol.append(np.random.randint(numCom))
        else:
            sol.append(best_solution[cont])
            cont += 1
    return np.array(sol)

def bistaratu_grafoa(G):
    plt.figure(figsize=(13, 9))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size = 20, node_color='0.75', label=True)
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1)
    plt.show()

    plt.axis('off')
    plt.show()



    


### Algoritmo Genetikoa ###

def hasieraketa_random(pop_size, sol_len, numCom):
    """ Funtzio honek populazio bat ausaz sortzen du. 
        Sarrerako parametroak:
            - pop_size: populazioaren tamaina (zenbat indibiduo)
            - sol_len: soluzoaren luzeera. Kasu honetan, grafoak dituen nodo kopurua
            - numCom: komunitate kopurua. 
        
        Irteera: soluzioak gordetzen dituen numpy array bat   
    """
    population = np.zeros((pop_size, sol_len), dtype=int) # populazioa gordetzeko 
    for i in range(pop_size):
        solution = np.random.randint(numCom, size=sol_len) # ausaz soluzio posible bat sortu
        population[i] = solution       
    return population


def soluzioak_ebaluatu(G, population, evals):
    """ Funtzio honek populazio bateko soluzio guztiak ebaluatzen ditu helburu-funtzioa erabiliz. 
        Sarrerako parametroak:
            - G: problemaren grafoa
            - population: ebaluatu nahi den populazioa
            - evals: ebaluaketa kopurua eguneratzeko 
        
        Irteera: 
            - ebaluatuak: sarrerako populazioa, baina orain soluzio bakoitza bere fitness-arekin gordetzen da tupla batean 
            - evals: ebaluaketa kopurua eguneratua
    """
    ebaluatuak = []
    for sol in population:
        partition = dict(zip(G.nodes, sol)) 
        fitness = modularitatea(G, partition)
        ebaluatuak.append((fitness, sol))
        evals += 1 
    return ebaluatuak, evals


def aukeraketa_operadorea(ebaluatuak, num_sel):
    """ Funtzio honek populazio bateko num_sel soluzio aukeratzen ditu Truncation Selection jarraituz. 
        Sarrerako parametroak:
            - ebaluatuak: soluzio bakoitza bere fitness-arekin tupla moduan gordetzen duen lista
            - num_sel: zenbat soluzio aukeratu
            
        Irteera: soluzio bakoitza bere fitness-arekin tupla moduan gordetzen duen lista
    """
    ordenatua = sorted(ebaluatuak, key=itemgetter(0), reverse=True) 
    hoberenak = ordenatua[:num_sel]
    return hoberenak


def birkonbinaketa_operadorea(aukeratuak, off_size, sol_len):   
    """ Funtzio honek populazio batetik 2 soluzio ausaz aukeratu eta One-Point Crossover erabiliz 2 soluzio berri sortzen ditu.
        Sarrerako parametroak:
            - aukeratuak: birkonbinaketa egin nahi den populazioa. Tupla bakoitzeko soluzioak eta fitnessa gordetzen duen lista.
            - off_size: sortuko diren soluzio berri kopurua
            - sol_len: soluzoaren luzeera. Kasu honetan, grafoan dauden nodo kopurua
        
        Irteera: soluzio berriak gordetzen dituen lista (fitness gabe) 
    """
    population = [sol[1] for sol in aukeratuak] # tuplak direnez, soluzioekin bakarrik geratu (eta ez fitness)
    new_pop = []
    
    for i in range(off_size // 2):
        index_sel1 = np.random.randint(len(aukeratuak)) # ausaz aukeratu 2 indibuduo (indizeak)
        index_sel2 = np.random.randint(len(aukeratuak))
        sel1 = population[index_sel1] # aukeratu birkonbinaketarako biak
        sel2 = population[index_sel2]
        
        pos = np.random.randint(sol_len) # pos bat ausaz
        new1 = np.append(sel1[:pos], sel2[pos:]) # berriak sortu
        new2 = np.append(sel1[pos:], sel2[:pos])
        
        new_pop.append(new1)
        new_pop.append(new2)
    
    return new_pop


def mutazio_operadorea(popBerria, mutProb, numCom):
    """ Funtzio honek populazio batetik 2 soluzio ausaz aukeratu eta One-Point Crossover erabiliz 2 soluzio berri sortzen ditu.
        Sarrerako parametroak:
            - popBerria: birkonbinaketan lortu diren soluzioak gordetzen duen lista.
            - mutProb: mutazio probabilitatea.
            - numCom: komunitate kopurua.
        
        Irteera: soluzio berriak gordetzen dituen lista (fitness gabe) 
    """
    for sol in popBerria: # indibiduo bat hartu
        for gen in sol: # gene bakoitzan mutazio probabilitatea aztertu
            randProb = np.random.uniform(0, 1)
            if randProb < mutProb:
                sol[gen] = np.random.randint(numCom)
    return popBerria


def eguneraketa_operadorea(G, oldEvaluated, newPopulation, pop_size, evals):
    """ Funtzio honek 2 populazio jaso eta hauen arteko pop_size hoberank itzultzen ditu 
        Sarrerako parametroak:
            - oldEvaluated: guraso populazio soluzio bakoitza bere fitness-arekin (ebaluatua)
            - newPopulation: mutaziotik atera berri den populazioa ebaluatu gabe
            - pop_size: populazio berriaren tamaina
            - evals: ebaluaketa kopurua eguneratzeko (berriak ebaluatu egingo dira eta)
        
        Irteera: 
            - hoberenak: soluzio hoberenak bere fitness-arekin tupla moduan gorsetzen dituen lista
    """
    newEvaluated, evals = soluzioak_ebaluatu(G, newPopulation, evals)
    tot = oldEvaluated + newEvaluated

    ordenatua = sorted(tot, key=itemgetter(0), reverse=True)
    hoberenak = ordenatua[:pop_size]
    return hoberenak, evals


def genetic_algorithm(G, numCom, sol_len, pop_size, num_sel, off_size, mutProb, max_evals):
    evals = 0
    populazioa = hasieraketa_random(pop_size, sol_len, numCom)
    while evals < max_evals:  # stop_criterion = ebaluazio kopurua 
        ebaluatuak, evals = soluzioak_ebaluatu(G, populazioa, evals)
        aukeratuak = aukeraketa_operadorea(ebaluatuak, num_sel)
        popBerria = birkonbinaketa_operadorea(aukeratuak, off_size, sol_len)
        popMutatuak = mutazio_operadorea(popBerria, mutProb, numCom)
        populazioBerria, evals = eguneraketa_operadorea(G, ebaluatuak, popMutatuak, pop_size, evals)
        populazioa = [x[1] for x in populazioBerria] # hurrengo iteraziorako soluzioak bakarrik behar dira ebaluatzeko
        
    
    # Azken populaziotik soluzio hobrerena
    popAzkena = sorted(populazioBerria, key=itemgetter(0), reverse=True)
    best_fitness = popAzkena[0][0]
    best_solution = popAzkena[0][1]
    return best_fitness, best_solution, evals



# Simulated Annealing

def hasierako_temperatura(fu, fl):
    ediff = -(fu - fl)
    return ediff / np.log(0.75)

def oreka(sol_len):
    #p = np.random.uniform(0, 1)
    p = 0.2
    return (p * sol_len)

def lortu_beta(initialTemp, oreka, max_evals):
    beta = initialTemp / (max_evals / oreka)
    return beta

def tenperatura_eguneratu(temp, beta):
    return temp - beta

def hamming_ingurune(solution, numCom):
    i = np.random.randint(len(solution))
    j = np.random.randint(numCom)
                               
    solution[i] = j
    return solution

def simulated_annealing(G, numCom, sol_len, max_evals, initialTemp, oreka, beta):
    best_solution = np.random.randint(numCom, size=sol_len)  # hasierako soluzioa ausaz
    best_partition = dict(zip(G.nodes, best_solution)) # hasierako partizioa
    best_fitness = modularitatea(G, best_partition) # hasierako fitness
    
    current = best_solution  # momentuko soluzioa godetuko da
    current_fitness = best_fitness # momentuko soluzioaren fitness-a godetuko da
    
    temp = initialTemp
    evals = 0
    while evals < max_evals:
        cont = 0
        while cont < oreka:
            rand_solution = hamming_ingurune(current, numCom) # inguruneak sortu
            rand_partition = dict(zip(G.nodes, rand_solution)) 
            rand_fitness = modularitatea(G, rand_partition) # ausazkoaren fitness
            evals += 1

            diff = rand_fitness - current_fitness
            if diff > 0: # Soluzio hobea
                current_solution = rand_solution
                if rand_fitness > best_fitness:
                    best_fitness = rand_fitness
                    best_solution = current_solution
            else:
                p = np.random.uniform(0, 1)
                prob = math.exp((diff) / temp)
                if p <= prob:
                    current = rand_solution
                    current_fitness = rand_fitness
            cont += 1
                           
        temp = tenperatura_eguneratu(temp, beta)
                
    return best_fitness, best_solution, evals