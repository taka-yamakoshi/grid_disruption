import numpy as np
import scipy
import argparse
import glob
import os

def calc_rate_map(data: dict, res: int = 35, sigma: float = 5.0, shuffle: bool = False, edge: float = 0.0):
    x, y = data['x'][:,0], data['y'][:,0]
    spki = data['spki'][:,0] - 1 # convert to python indexing
    if shuffle:
        rng = np.random.default_rng()
        spki = rng.permutation(len(x))[:len(spki)]

    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    xbins = np.linspace(xmin,xmax,res+1)
    ybins = np.linspace(ymin,ymax,res+1)
    spkx, spky = x[spki], y[spki]

    spike, _, _ = np.histogram2d(spkx,spky,[xbins,ybins])
    total, _, _ = np.histogram2d(x,y,[xbins,ybins])

    if edge > 0:
        d = int(np.floor(spike.shape[0]*edge))
        spike = spike[d:-d,d:-d]
        total = total[d:-d,d:-d]

    if sigma > 0:
        spike = scipy.ndimage.gaussian_filter(spike,sigma=sigma)
        total = scipy.ndimage.gaussian_filter(total,sigma=sigma)

    rmap = np.divide(spike,total+1e-10)
    return rmap, spike, total

def worker(dir_name, res, sigma, shuffle, edge):
    data = scipy.io.loadmat(dir_name)
    rmap, spike, total = calc_rate_map(data, res=res, sigma=sigma, shuffle=shuffle, edge=edge)
    from scores import GridScorer
    scorer = GridScorer(0) # get the scorer just to calculate sac
    corr, fpcorr = scorer.calc_score_rot(rmap)
    return rmap, spike, total, fpcorr

if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=int, default=35)
    parser.add_argument('--sigma', type=int, default=2)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--edge', type=float, default=0.0)
    args = parser.parse_args()

    run_ID = f'{args.res}-{args.sigma}-{args.edge}-shuffled' if args.shuffle else f'{args.res}-{args.sigma}-{args.edge}'
    print(f'Running {run_ID}')

    rmaps = {}
    spkes = {}
    totls = {}
    fpows = {}
    for cond in ['wty','wta','j20y','j20a']:
        rmaps[cond] = []
        spkes[cond] = []
        totls[cond] = []
        fpows[cond] = []
        dir_list = glob.glob(f'../Code-for-Ying-et-al.-2023/extracted_all/{cond}/*')
        pool_args = [(dir_name, args.res, args.sigma, args.shuffle, args.edge) for dir_name in dir_list]
        with mp.Pool(processes=32) as p:
            results = p.starmap(worker, pool_args)
        rmaps[cond], spkes[cond], totls[cond], fpows[cond] = zip(*results)

    os.makedirs(f'data/rmaps/{run_ID}/',exist_ok=True)
    np.savez(f'data/rmaps/{run_ID}/rmaps.npz',**rmaps)
    np.savez(f'data/rmaps/{run_ID}/spikes.npz',**spkes)
    np.savez(f'data/rmaps/{run_ID}/totals.npz',**totls)
    np.savez(f'data/rmaps/{run_ID}/fpowers.npz',**fpows)