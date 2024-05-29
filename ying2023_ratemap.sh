for sigma in 0 2 5 10
do
for edge in 0
do
python ying2023_ratemap.py --res 35 --sigma $sigma --edge $edge
python ying2023_ratemap.py --res 35 --sigma $sigma --edge $edge --shuffle
python ying2023_ratemap.py --res 35 --sigma $sigma --edge $edge --permute
done
done