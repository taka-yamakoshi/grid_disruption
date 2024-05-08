import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import os

from scores import GridScorer

def calc_rate_map(data: dict, res: int = 35, sigma: float = 5.0, shuffle: bool = False):
    x, y = data['x'][:,0], data['y'][:,0]
    spki = data['spki'][:,0] - 1 # convert to python indexing
    if shuffle:
        rng = np.random.default_rng()
        spki = rng.permutation(spki)

    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    xbins = np.linspace(xmin,xmax,res+1)
    ybins = np.linspace(ymin,ymax,res+1)
    spkx, spky = x[spki], y[spki]

    spike, _, _ = np.histogram2d(spkx,spky,[xbins,ybins])
    total, _, _ = np.histogram2d(x,y,[xbins,ybins])

    if sigma > 0:
        spike = scipy.ndimage.gaussian_filter(spike,sigma=sigma)
        total = scipy.ndimage.gaussian_filter(total,sigma=sigma)

    rmap = np.divide(spike,total+1e-10)
    return rmap, spike, total

if __name__ == '__main__':
    res = 35
    sigma = 2
    new_res = 255
    shuffle = True

    run_ID = f'{res}-{sigma}-{new_res}-shuffled' if shuffle else f'{res}-{sigma}-{new_res}'

    vmax = 100

    head = ['gtype','nid','max_freq','max_phase','freq','power']
    csv_data = []

    for cond in ['wty','wta','j20y','j20a']:
        dir_list = glob.glob(f'../Code-for-Ying-et-al.-2023/extracted_all/{cond}/*')
        os.makedirs(f'images/ying2023_all/{run_ID}/{cond}',exist_ok=True)
        for dir_name in dir_list:
            data = scipy.io.loadmat(dir_name)
            rmap, spike, total = calc_rate_map(data, res=res, sigma=sigma, shuffle=shuffle)

            scorer = GridScorer(res)
            max_freq, max_phase, score_60, score_90, cpol, fpcpol = scorer.calc_score_new(rmap,new_res=new_res)

            fname = dir_name.replace("../Code-for-Ying-et-al.-2023/extracted_all/","").replace(cond+"/","")
            nid = fname.split("-")[0]
            for i in range(1,10):
                csv_data.append([cond, int(nid), max_freq, max_phase, i, fpcpol[i]])

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
            ax.imshow(rmap)
            ax.axis('off')
            ax.set_title('Ratemap')

            ax = axs[3]
            ax.imshow(scorer.spectrum)
            ax.axis('off')
            ax.set_title('Spectrum')

            ax = axs[4]
            ax.plot(cpol)
            ax.set_yticks([])
            ax.set_xticks([])
            sns.despine(ax=ax)

            ax = axs[5]
            ax.scatter(np.arange(10)[1:],fpcpol[1:])
            sns.despine(ax=ax)
            ax.set_xticks([])

            fig.savefig(f'images/ying2023_all/{run_ID}/{cond}/{fname.replace(".mat",".png")}',bbox_inches = "tight")

            plt.clf()
            plt.close()


    df = pd.DataFrame(csv_data, columns=head)
    df.to_csv(f'data/ying2023_reanalysis_{run_ID}.csv', index=False)