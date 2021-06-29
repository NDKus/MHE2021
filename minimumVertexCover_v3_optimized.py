import itertools
import multiprocessing
import random
import json
import copy
import math
import sys
import operator
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
import time


with open("veryComplicatedProblem.json") as jsonfile:
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
            #print("Local score: " + str(sumofVerticies))
            return sumofVerticies
            
    #print(len(problem))
    return len(problem)

def goalToFitness(steps_amount, queue):
    '''
    Reprezentacja wyniku fitnes
    '''
    fitnessResult =  1.0/(1.0+steps_amount) 
    queue.put(fitnessResult) #dodajemy do kolejki nasz wynik wynik fitnesu

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

def GeneratePopulation(populationSize,problem):
    '''
    Chcemy pełnej populacji, więc zapętlamy losowanie z naszej metody "generateRandomVisitOrder" i tworzymy listę będąca reprezentacją rozwiązań.
    '''
    population = []
    for i in range(0, populationSize):
        population.append(generateRandomVistitOrder(len(problem)))
    return population
    
def RankPopulation(population, goalFunction):
    '''
    Aby zasymulować nasze „przetrwanie najsilniejszych”, możemy wykorzystać naszą gotową już funkcję celu, aby uszeregować każde rozwiązanie w populacji. 
    Naszym wynikiem będzie uporządkowana lista z indeksami rozwiązań z każdym powiązanym wynikiem oceny.
    '''
    populationscores = {}
    queue = multiprocessing.Queue() 
    procs = [] #tworzymy tablicę procesów
    for i in range(0,len(population)): #dla każdego rozwiązania w tej populacji
        proc = Process(target=goalToFitness, args = (goalFunction(population[i]),queue)) #tworzymy nowy proces, którego targetem jest "goalToFitness", argumentem wynik funkcji celu
        #przekazujemy też referencję do modułu queue z bioblioteki multiprocessing ^
        procs.append(proc) #dodajemy proces do tablicy procesów
        proc.start() #uruchamiamy ten proces
        
        #Takim sposobem uruchamiamy te procesy dla każdego rozwiązania.
        
    for index,proc in enumerate(procs): #dla każdego procesu w tablicy z uwzględnieniem indexu tego procesu
        proc.join() #synchronizujemy wynik procesu z wątkiem głównym
        populationscores[index] = queue.get() #dodajemy wynik tego procesu do zbioru pupulationscores
        print(str(populationscores))

        
    print("Posortowane wyniki populacji dla obecnej generacji: "+ str(sorted(populationscores.items(), key = operator.itemgetter(1),reverse=True)))
    return sorted(populationscores.items(), key = operator.itemgetter(1),reverse=True)

def selection(rankedPopulation, tournament_size):
    '''
    Metoda selekcji zwraca listę indexów rozwiązań, których możemy użyć do utworzenia puli osobników do rozmnażania w funkcji matingPool.
    '''
    selectionResults = []

    for i in range(0,len(rankedPopulation)):
        randomPickedSolutionIndex = random.randint(0,len(rankedPopulation)-1) #index pozycji
        best = rankedPopulation[randomPickedSolutionIndex] #[1] czyli wartosc - [0] to bylby index w populacji
        for j in range(1,tournament_size):
            randomPickedOpponentSolutionIndex = random.randint(0,len(rankedPopulation)-1)
            opponent = rankedPopulation[randomPickedOpponentSolutionIndex]
            if best[1] < opponent[1]:
                best = opponent
        selectionResults.append(best[0])
    print("Ilość wybranych rozwiązań: " + str(len(selectionResults)))
    return selectionResults
    
def matingPool(population, selectionResults):
    '''
    Teraz, gdy mamy już indexy rozwiązań, które utworzą naszą pulę osobników do rozmnażania z funkcji selekcji, możemy  wyodrębnić wybrane osobniki 
    z naszej populacji.
    '''
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

def crossover(parent1, parent2):
    '''
    Skoro nasza pula osobników do rozmnażania jest już gotowa, możemy utworzyć następną generację w procesie krzyżowania (crossover).
    Minimum Vertex Cover jest wyjątkowy, ponieważ wszystkie wierzchołki musimy uwzględnić dokładnie jeden raz, dlatego stosujemy tzw.
    "ordered crossover".
    
    W krzyżowaniu uporządkowanym, losowo wybieramy podzbiór pierwszego łańcucha rodzicielskiego (pierwsza pętla metody), 
    a następnie wypełniamy pozostałą część genami drugiego rodzica w kolejności, w jakiej się pojawiają, bez duplikowania żadnych genów 
    w wybranym podzbiorze od pierwszego rodzica.
    
    '''
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    for item in parent2:
        if item not in childP1:
            childP2.append(item)

    child = childP1 + childP2
    return child

def crossoverPopulation(matingpool, population):
    '''
    Następnie uogólnimy to, aby stworzyć naszą populację potomstwa. 
    Następnie w drugiej pętli używamy zdefiniowanej już osobnej metody krzyżowania, aby wypełnić resztę następnej generacji.
    '''
    children = []
    pool = random.sample(matingpool, len(matingpool))
    
    for i in range(0, len(population)):
        child = crossover(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

def mutate(individual, mutationRate):
    '''
    Mutacje chornią nas przed stagnacją, wprowadzając nowe rozwiązania.
    W naszym przypadku dobrze sprawdzi się "swap" tj. z określonym w parametrach programu, (najlepiej niewielkim)
    prawdopodobieństwem, dwa wierzchołki zamienią się kolejnością odwiedzeń w naszym rozwiązaniu. Kiedy funkcja mutacji się wykona,
    zrobimy to tylko dla jednego "osobnika" (individual)
    '''
    

    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            vertice1 = individual[swapped]
            vertiece2 = individual[swapWith]
            
            individual[swapped] = vertiece2
            individual[swapWith] = vertice1
            
    return individual

def mutatePopulation(population, mutationRate):
    '''
    Następnie możemy rozszerzyć funkcję mutacji, aby działała przez nową populację.
    '''
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

def NextGeneration(currentPopulation, tournamentSize, mutationRate, goalFunction):
    '''
    Łączy nasze metody, aby stworzyć funkcję, która stworzy nową generację. 
    Najpierw szeregujemy rozwiązania w obecnej generacji za pomocą rankPopulation. 
    Następnie określamy naszych potencjalnych rodziców, uruchamiając funkcję selekcji, która pozwala nam stworzyć pulę osobników do "rozmnażania", 
    za pomocą funkcji matingPool. Na koniec tworzymy naszą nową generację za pomocą funkcji crossoverPopulation zarządzająca by dla każdego osobnika
    z matingPool wykonał się crossovering (krzyzówka), a następnie stosujemy mutację za pomocą funkcji mutatePopulation.
    '''
    popRanked = RankPopulation(currentPopulation,goalFunction)
    selectionResults = selection(popRanked, tournamentSize)
    print("Aktualna populacja: " + str(currentPopulation))
    matingpool = matingPool(currentPopulation, selectionResults)
    children = crossoverPopulation(matingpool, currentPopulation)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

def EvolutionaryProgram( goalFunction, population, mutationRate, tournamentSize, iterations, printSolutionFunction):
    '''
    Nasza główna pętla. Tworzymy początkową populację, a następnie przechodzimy przez dowolną ilość generacji.
    '''

    progress = []

    progress.append(RankPopulation(population,goalFunction)[0][1])
    
    for i in range(0, iterations):
        progress.append(RankPopulation(population,goalFunction)[0][1])
        print("\n Generacja: " + str(i) + "\n")
        population = NextGeneration(population, tournamentSize, mutationRate, goalFunction)
    bestRouteIndex = RankPopulation(population,goalFunction)[0][0]
    bestRoute = population[bestRouteIndex]

    print ("Czas pracy algorytmu wyniósł: ", time.time() - start_time, "s")
    #Wykres
    plt.plot(progress)
    plt.gca().invert_yaxis()
    plt.ylabel('Fitness')
    plt.xlabel('Generacja')
    plt.show()
    

    return bestRoute

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

start_time = time.time()

#Parametry
iterations = 5
mutationRate = 0.01
popsize = 30
tournamentSize = 6

if  __name__ == '__main__':
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
            sol = fullSearch(lambda s: goalFunction(s, problem), problem, printSolution)
            print(goalFunction(sol, problem))
        if arg == '-evolutionProgram':
            sol = EvolutionaryProgram(lambda s:goalFunction(s, problem), GeneratePopulation(popsize, problem), mutationRate, tournamentSize, iterations, printSolution)

    print(sol)
    print(goalFunction(sol,problem))
    #problem = generateProblem(4, 3)
    #randomProbe(lambda s: goalFunction(s, problem), lambda: generateRandomVistitOrder(len(problem)), 3)
    #print(problem) 