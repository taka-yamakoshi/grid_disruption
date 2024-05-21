import numpy as np
import scipy
import argparse
import glob
import os
import time

def calc_rate_map(data: dict, res: int = 35, sigma: float = 5.0, shuffle: bool = False, permute: bool = False, edge: float = 0.0):
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

    if permute:
        rng = np.random.default_rng()
        rmap = rng.permutation(rmap.ravel()).reshape(rmap.shape[0],rmap.shape[1])

    return rmap, spike, total

def worker(dir_name, cond, res, sigma, shuffle, permute, edge):
    data = scipy.io.loadmat(dir_name)
    rmap, spike, total = calc_rate_map(data, res=res, sigma=sigma, shuffle=shuffle, permute=permute, edge=edge)
    from scores import GridScorer
    scorer = GridScorer(0) # get the scorer just to calculate sac
    autocorr, fpautocorr = scorer.calc_score_rot(rmap)
    cpol, fpcpol, speccorr = scorer.calc_score_fourier(rmap, new_res=255)

    speccorr_new, fpspeccorr_new = scorer.calc_score_fourier_new(rmap, new_res=255)

    fname = dir_name.replace("../Code-for-Ying-et-al.-2023/extracted_all/","").replace(cond+"/","")
    nid = fname.split("-")[0]
    return rmap, spike, total, autocorr, fpautocorr, cpol, fpcpol, speccorr, speccorr_new, fpspeccorr_new, int(data['animal']), int(data['age']), int(nid)

if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=int, default=35)
    parser.add_argument('--sigma', type=int, default=2)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--permute', action='store_true')
    parser.add_argument('--edge', type=float, default=0.0)
    args = parser.parse_args()

    run_ID = f'{args.res}-{args.sigma}-{args.edge}'
    run_ID = f'{run_ID}-shuffled' if args.shuffle else run_ID
    run_ID = f'{run_ID}-permuted' if args.permute else run_ID
    print(f'Running {run_ID}')

    rmaps, spkes, totls, autocorrs, fpautocorrs, cpols, fpcpols, speccorrs, speccorrs_new, fpspeccorrs_new, animals, ages, nids = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    for cond in ['wty','wta','j20y','j20a']:
        print(f'Running {cond}')
        dir_list = glob.glob(f'../Code-for-Ying-et-al.-2023/extracted_all/{cond}/*')
        pool_args = [(dir_name, cond, args.res, args.sigma, args.shuffle, args.permute, args.edge) for dir_name in dir_list]
        with mp.Pool(processes=32) as p:
            results = p.starmap(worker, pool_args)
        rmaps[cond], spkes[cond], totls[cond], autocorrs[cond], fpautocorrs[cond], cpols[cond], fpcpols[cond], speccorrs[cond], speccorrs_new[cond], fpspeccorrs_new[cond], animals[cond], ages[cond], nids[cond] = zip(*results)

    os.makedirs(f'data/rmaps/{run_ID}/',exist_ok=True)
    np.savez(f'data/rmaps/{run_ID}/rmaps.npz',**rmaps)
    np.savez(f'data/rmaps/{run_ID}/spikes.npz',**spkes)
    np.savez(f'data/rmaps/{run_ID}/totals.npz',**totls)
    np.savez(f'data/rmaps/{run_ID}/autocorr.npz',**autocorrs)
    np.savez(f'data/rmaps/{run_ID}/fpautocorr.npz',**fpautocorrs)
    np.savez(f'data/rmaps/{run_ID}/cpol.npz',**cpols)
    np.savez(f'data/rmaps/{run_ID}/fpcpol.npz',**fpcpols)
    np.savez(f'data/rmaps/{run_ID}/speccorr.npz',**speccorrs)
    np.savez(f'data/rmaps/{run_ID}/speccorr_new.npz',**speccorrs_new)
    np.savez(f'data/rmaps/{run_ID}/fpspeccorr_new.npz',**fpspeccorrs_new)
    np.savez(f'data/rmaps/{run_ID}/animal.npz',**animals)
    np.savez(f'data/rmaps/{run_ID}/age.npz',**ages)
    np.savez(f'data/rmaps/{run_ID}/nid.npz',**nids)