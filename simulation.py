import numpy as np
import pandas as pd

import argparse

from grid_cells import GridCells
from scores import GridScorer

def evaluate(scorer:object, symmetry: int, scale:float, hsize: float, vsize:float, hnoise:float, vnoise:float):
    cells = GridCells(symmetry=symmetry, scale=scale, size=(hsize, vsize), noise=(hnoise,vnoise))
    # generate rate maps
    activations = cells.run(16)
    # calculate scores
    data = []
    for act in activations:
        old_scores = scorer.calc_score(act, return_as_dict=True)
        new_scores = scorer.calc_score_new(act, return_as_dict=True)
        for freq in range(1,10):
            data.append([symmetry, scale, hsize, vsize, hnoise, vnoise, 'old', freq, old_scores['fpcpol'][freq]])
            data.append([symmetry, scale, hsize, vsize, hnoise, vnoise, 'new', freq, new_scores['fpcpol'][freq]])
    return data

if __name__ == '__main__':
    scorer = GridScorer(res=200)

    head = ['expID','symmetry','scale', 'hsize', 'vsize', 'hnoise', 'vnoise', 'score_type', 'freq', 'score']
    data = []

    # sweep over scales
    print("Sweeping over scales...")
    hnoise, vnoise = 0, 0
    for symmetry in [4,6]:
        for scale in np.arange(40,165,5):
            hsize, vsize = scale/4, scale/4
            data.extend([['Exp1'] + line for line in evaluate(scorer, symmetry, scale, hsize, vsize, hnoise, vnoise)])

    # sweep over sizes
    print("Sweeping over sizes...")
    scale = 100
    for symmetry in [4,6]:
        for size in np.arange(10,32,2):
            hsize, vsize = size, size
            data.extend([['Exp2'] + line for line in evaluate(scorer, symmetry, scale, hsize, vsize, hnoise, vnoise)])

    # sweep over sizes
    print("Sweeping over sizes...")
    scale = 100
    for symmetry in [2,4,6]:
        for size in np.arange(20,52,2):
            hsize, vsize = size, 20
            data.extend([['Exp3'] + line for line in evaluate(scorer, symmetry, scale, hsize, vsize, hnoise, vnoise)])

    # sweep over noise
    print("Sweeping over noise...")
    scale = 100
    hsize, vsize = 20, 20
    for symmetry in [4,6]:
        for noise in np.arange(5,16):
            vnoise = hnoise
            data.extend([['Exp4'] + line for line in evaluate(scorer, symmetry, scale, hsize, vsize, hnoise, vnoise)])

    df = pd.DataFrame(data, columns=head)
    df.to_csv('data/simulation.csv', index=False)