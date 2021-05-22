echo "-simAnnealing";
python ./minimumVertexCover_v3.py -simAnnealing > simAnnealing.txt
echo "-hillClimbingDeterministic";
python ./minimumVertexCover_v3.py -hillClimbingDeterministic > hillClimbingDeterministic.txt
echo "-hillClimbingRandomized";
python ./minimumVertexCover_v3.py -hillClimbingRandomized > hillClimbingRandomized.txt
