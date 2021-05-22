echo "simAnnealing";
python ./minimumVertexCover_v3.py -simAnnealing > simAnnealing.csv
echo "hillClimbingDeterministic";
python ./minimumVertexCover_v3.py -hillClimbingDeterministic > hillClimbingDeterministic.csv
echo "hillClimbingRandomized";
python ./minimumVertexCover_v3.py -hillClimbingRandomized > hillClimbingRandomized.csv
