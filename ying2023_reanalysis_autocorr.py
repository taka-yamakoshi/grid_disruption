import numpy as np
import scipy
import polars as pl
import glob
from tqdm import tqdm

from ying2023_reanalysis import calc_rate_map
from scores import GridScorer

if __name__ == '__main__':
    out_dict = {}
    #csv_data = []
    #head = ['gtype','nid','angle','corr']
    for cond in ['wty','wta','j20y','j20a']:
        out_dict[cond] = []
        dir_list = glob.glob(f'../Code-for-Ying-et-al.-2023/extracted_all/{cond}/*')
        for dir_name in tqdm(dir_list):
            data = scipy.io.loadmat(dir_name)
            rmap, spike, total = calc_rate_map(data, res=36, sigma=0)

            scorer = GridScorer(0) # get the scorer just to calculate sac
            sac = scorer.calc_sac(rmap)
            cpol = scorer.calc_cpol(sac, 0, (sac.shape[0]/2)*0.7)
            cpol = scipy.ndimage.gaussian_filter(cpol,sigma=3,mode='wrap')

            #dx, dy = (sac.shape[0]-1)//2, (sac.shape[0]-1)//2
            #disc = scorer._get_disc(xlims=[-dx,dx],ylims=[-dy,dy],res=sac.shape[0])
            #mask = disc<min(dx,dy)

            #corr = []
            #angles = np.linspace(0,360,361)
            #for angle in angles:
            #    rot_sac = scipy.ndimage.rotate(sac, angle, reshape=False)
            #    corr.append(scipy.stats.pearsonr(sac[mask].ravel(), rot_sac[mask].ravel()).statistic)
            #out_dict[cond].append(corr)
            out_dict[cond].append(list(cpol))

            #fname = dir_name.replace("../Code-for-Ying-et-al.-2023/extracted_all/","").replace(cond+"/","")
            #nid = fname.split("-")[0]
            #csv_data.extend([[cond, nid, angl, val] for angl, val in zip(angles, corr)])
    np.savez(f'data/ying2023_reanalysis_autocorr.npz',**out_dict)
    #df = pl.DataFrame(csv_data,schema=head,orient='row')
    #df.write_csv(f'data/ying2023_reanalysis_autocorr.csv')