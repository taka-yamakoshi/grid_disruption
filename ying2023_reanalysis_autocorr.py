import numpy as np
import scipy
import polars as pl
import glob
import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from ying2023_ratemap import calc_rate_map

def calc_autocorr(run_ID, cond, dir_name, res, sigma, shuffle, edge, vmax):
    data = scipy.io.loadmat(dir_name)
    rmap, spike, total = calc_rate_map(data, res=res, sigma=sigma, shuffle=shuffle, edge=edge)

    from scores import GridScorer
    scorer = GridScorer(0)
    corr, fpcorr = scorer.calc_score_rot(rmap)

    fname = dir_name.replace("../Code-for-Ying-et-al.-2023/extracted_all/","").replace(cond+"/","")
    nid = fname.split("-")[0]
    csv_lines = [[cond, nid, freq, power] for freq, power in zip(np.arange(1,10), fpcorr[1:])]

    fig, axs = plt.subplots(1,6,figsize=(15,3.2),gridspec_kw=dict(wspace=0.15))
    ax = axs[0]
    ax.imshow(total,vmin=0,vmax=vmax)
    ax.axis('off')
    ax.set_title('Occupancy')

    ax = axs[1]
    ax.imshow(spike,vmin=0,vmax=vmax/10)
    ax.axis('off')
    ax.set_title('Spike')

    ax = axs[2]
    ax.imshow(rmap,vmin=0)
    ax.axis('off')
    ax.set_title('Ratemap')

    ax = axs[3]
    ax.imshow(scorer.sac, alpha=scorer.mask.astype(float)/2+1/2)
    ax.axis('off')
    ax.set_title('SAC')

    ax = axs[4]
    ax.plot(corr)
    sns.despine(ax=ax)
    ax.set_xticks([])

    ax = axs[5]
    ax.scatter(np.arange(1,10),fpcorr[1:])
    sns.despine(ax=ax)
    ax.set_xticks([])

    fig.savefig(f'images/ying2023_autocorr/{run_ID}/{cond}/{fname.replace(".mat",".png")}',bbox_inches = "tight")

    plt.clf()
    plt.close()

    return corr, csv_lines

if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=int, default=35)
    parser.add_argument('--sigma', type=int, default=2)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--edge', type=float, default=0.0)
    args = parser.parse_args()

    vmax = 100 if args.res < 50 else 10

    out_dict = {}

    run_ID = f'{args.res}-{args.sigma}-{args.edge}-shuffled' if args.shuffle else f'{args.res}-{args.sigma}-{args.edge}'
    print(f'Running {run_ID}')

    csv_data = []

    head = ['gtype','nid','freq','power']
    for cond in ['wty','wta','j20y','j20a']:
        print(f'Running {cond}')
        out_dict[cond] = []
        dir_list = glob.glob(f'../Code-for-Ying-et-al.-2023/extracted_all/{cond}/*')
        os.makedirs(f'images/ying2023_autocorr/{run_ID}/{cond}',exist_ok=True)
        pool_args = [(run_ID, cond, dir_name, args.res, args.sigma, args.shuffle, args.edge, vmax) for dir_name in dir_list]
        with mp.Pool(processes=32) as p:
            results = p.starmap(calc_autocorr, pool_args)
        out_dict[cond], csv_data_cond = zip(*results)
        csv_data.extend([subline for line in csv_data_cond for subline in line])

    np.savez(f'data/ying2023_reanalysis_autocorr_{run_ID}.npz',**out_dict)
    df = pl.DataFrame(csv_data,schema=head,orient='row')
    df.write_csv(f'data/ying2023_reanalysis_autocorr_{run_ID}.csv')