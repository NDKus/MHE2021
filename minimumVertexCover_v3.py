import itertools
import random
import json
import copy
import math
import sys

with open("problem.json") as jsonfile:
    jsonparsed = json.load(jsonfile)
problem = jsonparsed["graph"]

#Poprzednio używany, prostszy graf
#"graph": [
#[ 1, 2, 3 ],
#[ 5, 1 ],
#[ 2, 6, 7 ],
#[ 3, 4 ],
#[ 5, 6, 8 ],
#[ 4, 7, 9 ],
#[ 8, 9 ]
#]


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

    sumofVerticies = 0
    edges = {edge for vertex in problem for edge in vertex}
   

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
                #W naszym przypadku im niższy wynik, tym lepiej, dlatego w przypadku natknięcia się na krawędź, która była już pokryta, 
                #dodajemy punkt "karny" pogarszający to rozwiązanie.
                sumofVerticies += 1
        sumofVerticies += 1

        if len(edges) == 0:
            print("Local score: " + str(sumofVerticies))
            return sumofVerticies
            
    print(len(problem))
    return len(problem)

def generateRandomVistitOrder(n):
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

def getRandomNeighbour(forNowBestSolution):
    '''
    1. Losuje index
    2. Kopiuję aktualne najlepsze rozwiązanie przy pomocy deepcopy (pełna kopia tego obiektu) 
    3. Jeśli np wylosowaliśmy z kopii, index 5 a długość problemu to 7, to wykonujemy operację 5+1%7 = 6, 
    bierzemy z tej kopi element o indexie 6 i nadpisujemy go elementem o indexie 5 z forNowBest Solution
    4. Z naszej kopii bierzemy nasz index (tutaj np. 5) i nadpisujemy ten element elementem o indexie 6 - czyli w skutku uzyskujemy kopię 
    rozwiązania z zamienionymi dwoma elementami
    5. Zwracamy tą kopię
    '''
    indexOfRandomSolution = int(random.uniform(0, len(forNowBestSolution)-1)) 
    copyOfBestSolution = copy.deepcopy(forNowBestSolution) 
    copyOfBestSolution[(indexOfRandomSolution+1) % len(forNowBestSolution)] = forNowBestSolution[indexOfRandomSolution] 
    copyOfBestSolution[indexOfRandomSolution] = forNowBestSolution[(indexOfRandomSolution+1) % len(forNowBestSolution)] 
    return copyOfBestSolution

def getBestNeighbour(forNowBestSolution, goal):
    '''
    1. Wykonuję peętlę tyle razy, ile wynosi długość problemu
    2. Wykonuje pełną kopię forNowBestSolution
    3. Numer itracji tej pętli, np 5 daje nam elementy 5 i 6, które zostają zamienione w copyOfBestSolution, a nastepnie porównujemy wynik copyOfBestSolutiun tj. najlepszego rozwiązania z forNowBestSollution
    4. Jeżeli kopia jest lepsza, zostaje zapisana jako nowe forNowBestSollution
    5. Porównuje ilość kroków potrzebną do rozwiązania aktualnego copyOfBestSolution z forNowBestSolution
    6. Jeśli jest mniejsza (lepsza), to copyOfBestSolution jest teraz naszym nowym forNowBestSolution, a następnie iteruje do kolejnego sąsiada
    7. Po porównianiu wszystkich wartości dla indeksów, zwaraca forNowBestSolution
    '''
    for indexOfRandomSolution in range(0, len(forNowBestSolution) - 1):
        copyOfBestSolution = copy.deepcopy(forNowBestSolution)
        copyOfBestSolution[(indexOfRandomSolution + 1) % len(forNowBestSolution)] = forNowBestSolution[indexOfRandomSolution]
        copyOfBestSolution[indexOfRandomSolution] = forNowBestSolution[(indexOfRandomSolution + 1) % len(forNowBestSolution)]
        if (goal(copyOfBestSolution) <= goal(forNowBestSolution)):
            forNowBestSolution = copyOfBestSolution
    return forNowBestSolution

def getRandomNeighbour2(forNowBestSolution):
    '''
    Ulepszona funkcja getBestNeighbour na potrzeby wyżarzania, wzorowana na przykładzie z zajęć
    '''
    for i in range(0, int(min(abs(random.normalvariate(0.0, 2.0)) + 1, 500))):
        indexOfRandomSolution = int(random.uniform(0, len(forNowBestSolution) - 1))
        copyOfBestSolution = copy.deepcopy(forNowBestSolution)
        copyOfBestSolution[(indexOfRandomSolution + 1) % len(forNowBestSolution)] = forNowBestSolution[indexOfRandomSolution]
        copyOfBestSolution[indexOfRandomSolution] = forNowBestSolution[(indexOfRandomSolution + 1) % len(forNowBestSolution)]
        forNowBestSolution = copyOfBestSolution
    return forNowBestSolution

def hillClimbingRandomized(goal, gensol, genNeighbour, iterations,onIteration):
    '''
    W przypadku problemów NP, często okazuje się że wspinaczka losowa jest zaskakująco skuteczna. Czasami lepiej
    jest oszczędzić sobie czasu obliczeniowego na badanie pełnej przestrzeni i wprowadzić element losowości, niż deterministycznie sprawdzać wszystkie możliwości.
    
    hillClimbingRandomized nie sprawdza wszystkich sąsiednich węzłów przed podjęciem decyzji, który węzeł wybrać. W dowolnym punkcie przestrzeni stanów 
    wyszukiwanie porusza się tylko w tym kierunku, który optymalizuje koszt funkcji z nadzieją na znalezienie optymalnego rozwiązania na końcu. 

    1. Losowo przestawiamy kolejność odwiedzania wierzchołków jako pierwsze rozwiązanie
    2. W pętli ustawiamy newPossibleSolution przy pomocy naszej metody GenNeighbour (zamiana miejscami)
    3. Porównujemy czy NewPossibleSolution jest lepsze niż forNowBestSolution
    4. Jeśli jest lepsze, forNowBestSolution jest jest teraz naszym najlepszym rozwiązaniem
    5. printujemy informacje o iteracji
    6. po porównaniu wszystkich możliwych rozwiązań z podanej ilości itreacji, zwracamy najlepszy wynik
    '''
    forNowBestSolution = gensol() 
    for i in range(0, iterations): 
        newPossibleSolution = genNeighbour(forNowBestSolution) 
        if (goal(newPossibleSolution) <= goal(forNowBestSolution)): 
            forNowBestSolution = newPossibleSolution 
        onIteration(i, forNowBestSolution, goal) 
    return forNowBestSolution 

def hillClimbingDeterministic(goal, gensol, genBestNeighbour, iterations,onIteration):
    '''Deterministyczne podejście do wspinaczki. Czasochłonne, może przy braku szczęścia wymagać bardzo dużej ilości iteracji aby dać wynik podobny do 
    zrandomizowanego wariantu.

    1. Losowo przestawiamy kolejność odwiedzania wierzchołków jako pierwsze rozwiązanie
    2. W pętli ustawiamy newPossibleSolution wybierająć najlepszego sąsiąda 
    3. Jeśli natkniemy się na ten sam NewPossibleSolution co forNowBestSollution, zwracamy forNowBestSolution
    4. Jeśli jest inne, forNowBestSolution jest jest teraz naszym najlepszym rozwiązaniem
    5. Printujemy informacje o iteracji
    6. Jeśli wszystkie wykonane iteracje nie zwrócą newPossibleSolution identycznego z forNowBestSollution, zwracamy forNowBestSollution
    '''
    forNowBestSolution = gensol()
    for i in range(0, iterations):
        newPossibleSolution = genBestNeighbour(forNowBestSolution, goal)
        if (newPossibleSolution == forNowBestSolution):
            return forNowBestSolution
        forNowBestSolution = newPossibleSolution
        onIteration(i, forNowBestSolution, goal)
    return forNowBestSolution

def simAnnealing(goal, gensol, genNeighbour, T, iterations, onIteration):
    '''
    W symulowanym wyżarzaniu korzystamy ze zmiennej temperatury na wzór procesu metalurgicznego o tej samej nazwie, aby symulować ten proces "ogrzewania". 
    Początkowo ustawiamy temperaturę wysoko, a następnie pozwalamy „ostygnąć” w trakcie działania algorytmu. 
    Chociaż ta zmienna temperatury jest wysoka, algorytm będzie mógł, z większą częstotliwością, akceptować rozwiązania gorsze niż nasze obecne rozwiązanie. 
    Daje to algorytmowi możliwość wyskoczenia z wszelkich lokalnych optimów, w których znajduje się na wczesnym etapie wykonywania. 
    Wraz ze spadkiem temperatury rośnie też szansa na zaakceptowanie gorszych rozwiązań, co pozwala algorytmowi na stopniowe skupianie się na 
    obszarze, w którym można znaleźć rozwiązanie bliskie optymalnemu.

    1. Losowo przestawiamy kolejność odwiedzania wierzchołków jako pierwsze rozwiązanie
    2. Tworzymy nową listę ze wszystkimi forNowBestSollution podczas wykonywania operacji 
    3. W pętli iteracyjnej ustawiamy nowe newPossibleSolution, które ma zamienioen dwie wartości
    4. Porównujemy czy wynik newPossibleSoltuion jest lepszy lub równy jak pierwsze rozwiązanie (forNowBestSolution)
    5. Jeśli jest lepsze lub równe, forNowBestSollution jest poprzednim newPossibleSolution
    6. Dodaje nowe forNowBestSolution do listy wszystkich najlepszych rozwiązań
    7. Jeśli newPossibleSolution było gorsze:
    ...
    ...
    9. Printujemy informacje o iteracji
    10. Zwracamy najlepsze rozwiązanie do listOfBestSolutions
    '''
    forNowBestSolution = gensol()
    listOfBestSolutions = [forNowBestSolution]
    for i in range(1, iterations + 1):
        newPossibleSolution = genNeighbour(forNowBestSolution)
        if (goal(newPossibleSolution) <= goal(forNowBestSolution)):
            forNowBestSolution = newPossibleSolution
            listOfBestSolutions.append(forNowBestSolution)
        else:
            e = math.exp(- abs(goal(newPossibleSolution) - goal(forNowBestSolution)) / T(i))
            u = random.uniform(0.0, 1.0)
            if (u < e):
                forNowBestSolution = newPossibleSolution
                listOfBestSolutions.append(forNowBestSolution)
        onIteration(i - 1, forNowBestSolution, goal)

    return min(listOfBestSolutions, key=goal)

iterations = 500

for arg in sys.argv:
    if arg == '-hillClimbingRandomized':
        sol = hillClimbingRandomized(lambda s: goalFunction(s, problem), lambda: generateRandomVistitOrder(len(problem)), getRandomNeighbour, iterations, printSolution)
        print(goalFunction(sol, problem))
    if arg == '-hillClimbingDeterministic':
        sol = hillClimbingDeterministic(lambda s: goalFunction(s, problem), lambda: generateRandomVistitOrder(len(problem)), getBestNeighbour, iterations, printSolution)
        print(goalFunction(sol, problem))
    if arg == '-simAnnealing':
        sol = simAnnealing(lambda s: goalFunction(s, problem), lambda: generateRandomVistitOrder(len(problem)), getRandomNeighbour2, lambda k: 1000.0 / k, iterations, printSolution)
        print(goalFunction(sol, problem))
    if arg == '-fullSearch':
        if len(problem) < 9: #jeżeli ilość wierzchołków jest mniejsza od 9 to opłaca się stosować bruteforca
            sol = fullSearch(lambda s: goalFunction(s, problem), problem, printSolution)
            print(goalFunction(sol, problem))

#print(goalFunction(sol,problem))
#problem = generateProblem(4, 3)
#randomProbe(lambda s: goalFunction(s, problem), lambda: generateRandomVistitOrder(len(problem)), 3)
#print(problem)