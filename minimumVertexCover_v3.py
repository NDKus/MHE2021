import itertools
import random
import json
import copy
import math
import sys

with open("problem.json") as jsonfile:
    jsonparsed = json.load(jsonfile)
problem = jsonparsed["graph"]

#"graph": [
#[ 1, 2, 3 ],
#[ 5, 1 ],
#[ 2, 6, 7 ],
#[ 3, 4 ],
#[ 5, 6, 8 ],
#[ 4, 7, 9 ],
#[ 8, 9 ]
#]

'''
[0]: 1, 2, 3
[1]: 3, 4
[2]: 2, 6, 7
[3]: 1, 5
[4]: 5, 6, 8
[5]: 4, 7, 9
[6]: 8, 9

graph = [[1,2,3], [5, 1], [2, 6, 7], [3, 4], [5, 6, 8], [4, 7, 9], [8, 9]]
sol = [0, 5, 4, 6]
sol2 = [0, 1, 2, 3, 6]
sol3 = [0, 1]

'''

def goalFunction(solution, problem):
    '''Reprezentacja rozwiazania - lista kolejnych wierzchołków do odwiedzenia.
    Jako, iż zdecydowaliśmy się na reprezentację grafu:
    [[wierzchołek 0: krawędzie 1,2],[1:1,3]; Wykorzystujemy do numeracji krawędzi strukturę set, która zatrzymuje
    tylko unikatowe wartości eliminując powtórzenia, tak aby była tylko jedna krawedź o oznaczeniu 1 itd.
    Tym sposobem wiemy zarówno jak prezentują się połączenia wszystkich wierzchołków z krawędziami i wiemy ile
    krawędzi znajduje się w grafie

    Sprawdzamy czy wszystkie krawedzie z przykładowego rozwiązania zostały pokryte. W tym celu sprawdzamy pierwszy
    element tablicy o indeksie [0], czyli wierzchołek "1" i widzimy np., że ma krawędzie [1,2] - usuwamy więc je
    (discard) z tablicy krawedzi. Iterujemy i sprawdzamy cały graf.
    Jeśli edges zostanie puste (len(edges) = 0), to pokrycie jest stuprocentowe, i wówczas naszym wynikiem jest
    suma wierzchołków.
    Jeśli pokrycie stuprocentowe nie zostanie osiągnięte, zwracamy długość grafu aby później móc porównać to rozwiązanie
    z innymi (return len(problem).'''

    sumOfVertices = 0
    edges = {edge for vertex in problem for edge in vertex}
    # edges = [4, 5, 6, 7]

    '''
    Powyższy zapis, to to samo co zapis alternatywny:
    edges = set()
    for vertex in problem:
        for edge in vertex:
            edges.add(edge)    
    '''

    print(solution)
    for i in range(0, len(solution)):
        vertex = problem[solution[i]]
        for edge in vertex:
            if edge in edges:
                edges.discard(edge)
            else:
                sumOfVertices += 1
        sumOfVertices += 1

        if len(edges) == 0:
            print("Local score: " + str(sumOfVertices))
            return sumOfVertices
            
    print(len(problem))
    return len(problem)
#ile razy dana krawedz zostala powielona ? O ile jest pokryta wiecej niz raz.

def generateRandomVisitOrder(n):
    """Generuje losową kolejność odwiedzenia wierzchołków i zwraca ją. Skopiowane z przykładu z zajęć.
    r to nasza tablica wierzchołków w kolejności do odwiedzenia"""
    r = []
    f = list(range(0, n)) #lista o długości 0-n gdzie n to długość całego grafu
    for i in range(0, n):
        p = int(random.uniform(0, len(f) - 1)) #p to wygenerowany, losowy wierzchołek, odejmujemy 1 ponieważ indeks pierwszego elementu to 0
        r.append(f[p]) #dopisujemy do tablicy kolejności wierzchołków do odwiedzenia wyegenrowaną liczbę
        del f[p] #usuwamy wygenerowaną liczbę z listy pozostałych wierzchołków do odwiedzenia
        # print(f)
        # print(r)
    return r


def fullSearch(goalFunction, problem, printSolutionFunction):
    '''
    Nasz BruteForce, którym sprawdzamy wszystkie możliwe permutacje kolejności odwiedzin wierzchołków celem sprawdzenia pokryć
    dla wszystkich z nich, nadpisując zawsze najniższą sumę wierzchołków.
    itertool.permutations() dostarcza nam wszystkie możliwe permutacje.
    '''
    potentialSolution = list(range(0, len(problem)))
    forNowBestSolution = potentialSolution
    iteration = 0
    for newPotentialSolution in itertools.permutations(potentialSolution):
        if goalFunction(newPotentialSolution) < goalFunction(forNowBestSolution):
            forNowBestSolution = newPotentialSolution
        printSolutionFunction(iteration,forNowBestSolution,goalFunction)
        iteration += 1
        print(newPotentialSolution)
    return forNowBestSolution



def randomProbe(goal, gensol, iterations):
    '''
    (dot. SOL) Random probe - generuje kilka razy losowe rozwiązanie (kolejność), które przekazuje do funkcji celu,
    iterując powtarza czynność kilka razy i sprawdza ktore rozwiazanie bylo najlepsze.
    '''
    forNowBestSolution = gensol()
    for i in range(0, iterations):
        newPossibleSolution = gensol()
        if goal(newPossibleSolution) < goal(forNowBestSolution):
            forNowBestSolution = newPossibleSolution
        #print(goal(forNowBestSolution))
    return forNowBestSolution


def generateProblem(size, probability): #<<< nie zawsze działa, czasem wierzchołek jest pusty
    dummy = [v for v in range(size)]
    graph = set()
    for combination in itertools.combinations(dummy, 2):
        a = random.uniform(0,1)
        for vertex in dummy:
            if a < probability:
                graph.add(combination)
                
    result = [list() for i in range(size)]
    index = 0
    print(graph)
    for edge in graph: 
        index += 1
        for vertex in edge:
            result[vertex].append(index)

    return result



def printSolution(i, forNowBestSolution, goal):
    print("" + str(i) + " | Best Score: " + str(goal(forNowBestSolution)))

def doNotPrintSolution(i, forNowBestSolution, goal):
    return 0


def getRandomNeighbour(forNowBestSolution):
    pointIndex = int(random.uniform(0, len(forNowBestSolution)-1))
    copyOfBestSolution = copy.deepcopy(forNowBestSolution)
    copyOfBestSolution[(pointIndex+1) % len(forNowBestSolution)] = forNowBestSolution[pointIndex]
    copyOfBestSolution[pointIndex] = forNowBestSolution[(pointIndex+1) % len(forNowBestSolution)]
    return copyOfBestSolution

def getBestNeighbour(forNowBestSolution, goal):
    for pointIndex in range(0, len(forNowBestSolution) - 1):
        copyOfBestSolution = copy.deepcopy(forNowBestSolution)
        copyOfBestSolution[(pointIndex + 1) % len(forNowBestSolution)] = forNowBestSolution[pointIndex]
        copyOfBestSolution[pointIndex] = forNowBestSolution[(pointIndex + 1) % len(forNowBestSolution)]
        if (goal(copyOfBestSolution) <= goal(forNowBestSolution)):
            forNowBestSolution = copyOfBestSolution
    return forNowBestSolution

def getRandomNeighbour2(forNowBestSolution):
    for i in range(0, int(min(abs(random.normalvariate(0.0, 2.0)) + 1, 500))):
        pointIndex = int(random.uniform(0, len(forNowBestSolution) - 1))
        copyOfBestSolution = copy.deepcopy(forNowBestSolution)
        copyOfBestSolution[(pointIndex + 1) % len(forNowBestSolution)] = forNowBestSolution[pointIndex]
        copyOfBestSolution[pointIndex] = forNowBestSolution[(pointIndex + 1) % len(forNowBestSolution)]
        forNowBestSolution = copyOfBestSolution
    return forNowBestSolution


def hillClimbingRandomized(goal, gensol, genNeighbour, iterations,onIteration):
    ''''''
    forNowBestSolution = gensol()
    for i in range(0, iterations):
        newPossibleSolution = genNeighbour(forNowBestSolution)
        if (goal(newPossibleSolution) <= goal(forNowBestSolution)):
            forNowBestSolution = newPossibleSolution
        onIteration(i, forNowBestSolution, goal)
    return forNowBestSolution

def hillClimbingDeterministic(goal, gensol, genBestNeighbour, iterations,onIteration):
    ''''''
    forNowBestSolution = gensol()
    for i in range(0, iterations):
        newPossibleSolution = genBestNeighbour(forNowBestSolution, goal)
        if (newPossibleSolution == forNowBestSolution):
            return forNowBestSolution
        forNowBestSolution = newPossibleSolution
        onIteration(i, forNowBestSolution, goal)
    return forNowBestSolution

def simAnnealing(goal, gensol, genNeighbour, T, iterations, onIteration):
    ''''''
    forNowBestSolution = gensol()
    V = [forNowBestSolution]
    for i in range(1, iterations + 1):
        newPossibleSolution = genNeighbour(forNowBestSolution)
        if (goal(newPossibleSolution) <= goal(forNowBestSolution)):
            forNowBestSolution = newPossibleSolution
            V.append(forNowBestSolution)
        else:
            e = math.exp(- abs(goal(newPossibleSolution) - goal(forNowBestSolution)) / T(i))
            u = random.uniform(0.0, 1.0)
            if (u < e):
                forNowBestSolution = newPossibleSolution
                V.append(forNowBestSolution)
        onIteration(i - 1, forNowBestSolution, goal)

    return min(V, key=goal)

iterations = 1000

for arg in sys.argv:
    if arg == '-hillClimbingRandomized':
        sol = hillClimbingRandomized(lambda s: goalFunction(s, problem), lambda: generateRandomVisitOrder(len(problem)), getRandomNeighbour, iterations, printSolution)
        print(goalFunction(sol, problem))
    if arg == '-hillClimbingDeterministic':
        sol = hillClimbingDeterministic(lambda s: goalFunction(s, problem), lambda: generateRandomVisitOrder(len(problem)), getBestNeighbour, iterations, printSolution)
        print(goalFunction(sol, problem))
    if arg == '-simAnnealing':
        sol = simAnnealing(lambda s: goalFunction(s, problem), lambda: generateRandomVisitOrder(len(problem)), getRandomNeighbour2, lambda k: 1000.0 / k, iterations, printSolution)
        print(goalFunction(sol, problem))
    if arg == '-fullSearch':
        if len(problem) < 9: #jeżeli ilość wierzchołków jest mniejsza od 9 to opłaca się stosować bruteforca
            sol = fullSearch(lambda s: goalFunction(s, problem), problem, printSolution)
            print(goalFunction(sol, problem))

#print(goalFunction(sol, problem))
#problem = generateProblem(4, 3)
#randomProbe(lambda s: goalFunction(s, problem), lambda: generateRandomVisitOrder(len(problem)), 3)
#print(problem)