#Author: Przemysław Scharmach

# MHE2021
Metaheurystyki, Minimum Vertex Cover

Problem polega na wskazaniu węzłów (wierzchołków) które zapewnią pokrycie wszystkich krawędzi w grafie, jednocześnie stosując tych węzłów jak najmniej aby ten efekt uzyskać.

Opis problemu - wikipedia: https://en.wikipedia.org/wiki/Vertex_cover
Opis problemu - źródło inne: https://morioh.com/p/98d2eb221761

Dane wejściowe to tablica tablic. Index tablicy wewnętrznej oznacza wierzchołek w grafie, a jej zawartość to krawędzie jakie ten wierzchołek posiada. 
Np. Wierzchołek o indexie 0, posiada krawędzie 1,2,3 - bazując na tym wiemy gdzien umieścić kolejny wierzchołek [1] posiadający np. krawedzie [5,1]
Przykładowa tablica:

problem = [[1, 2, 3], [5, 1], [2, 6, 7], [3, 4], [5, 6, 8], [4, 7, 9], [8, 9]]  - 7 wierzchołków, tablica 7 elementowa


Dane wyjściowe to kolejność poruszania się po wierzchołkach aby uzyskać pokrycie w optymalnym czasie, a także informacja
o tym w którym kroku zapewniono pełne pokrycie:

np. (0, 4, 5, 1, 2, 3, 6) - 
(W przypadku tych danych, już na trzecim kroku zapewnimy pełne pokrycie krawędzi)