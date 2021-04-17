import itertools
import random
import json

with open("problem.json") as jsonfile:
    jsonparsed = json.load(jsonfile)
problem = jsonparsed["graph"]

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

    '''
    Powyższy zapis, to to samo co zapis alternatywny:
    edges = set()
    for vertex in problem:
        for edge in vertex:
            edges.add(edge)    
    '''

    for i in range(0, len(solution)):
        vertex = problem[solution[i]]
        for edge in vertex:
            edges.discard(edge)
        sumOfVertices += 1

        if len(edges) == 0:
            return sumOfVertices

    return len(problem)


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


def fullSearch(goal, problem):
    '''
    Nasz BruteForce, którym sprawdzamy wszystkie możliwe permutacje kolejności odwiedzin wierzchołków celem sprawdzenia pokryć
    dla wszystkich z nich, nadpisując zawsze najniższą sumę wierzchołków.
    itertool.permutations() dostarcza nam wszystkie możliwe permutacje.
    '''
    f = list(range(0, len(problem)))
    currentBest = f
    for newSol in itertools.permutations(f):
        # print(newSol)
        if goal(newSol) < goal(currentBest):
            currentBest = newSol
    return currentBest


def randomProbe(goal, gensol, iterations):
    '''
    (dot. SOL) Random probe - generuje kilka razy losowe rozwiązanie (kolejność), które przekazuje do funkcji celu,
    iterując powtarza czynność kilka razy i sprawdza ktore rozwiazanie bylo najlepsze.
    '''
    currentBest = gensol()
    for i in range(0, iterations):
        newSol = gensol()
        if goal(newSol) < goal(currentBest):
            currentBest = newSol
        print(goal(currentBest))
    return currentBest


def generateProblem(size, probability): #<<< nie działa, czasem wierzchołek jest pusty
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

#problem = generateProblem(5, 3)
print(problem)

# print (generateRandomVisitOrder(4))
# print (goalFunction([0,2,1,3],problem))

sol = randomProbe(lambda s: goalFunction(s, problem), lambda: generateRandomVisitOrder(len(problem)), 3)

print(sol)
print(goalFunction(sol, problem))
if len(problem) < 10: #jeżeli ilość wierzchołków jest mniejsza od 10 to opłaca się stosować bruteforca
    brute = fullSearch(lambda s: goalFunction(s, problem), problem)
    print(fullSearch(lambda s: goalFunction(s, problem), problem))

print(goalFunction(brute, problem))
