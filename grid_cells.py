from typing import Tuple
import scipy
import numpy as np

class GridCells(object):
    def __init__(self,
                 symmetry:int = 6,
                 scale:float = 80.0,
                 size:Tuple[float,float] = (10.0,10.0),
                 noise:Tuple[float,float] = (0.0,0.0),
                 res:int = 200,
                 shape:str = 'gaussian',
                 ):
        '''
        Generates grid templates.
        Parameters:
        ----------
        symmetry: either 2,4 or 6
        scale: scale of the grid
        size: size of the grid in sigmas of the gaussian
        noise: noise added to the grid locations
        res: resolution of the image
        shape: shape of the receptive field

        Returns:
        -------
        mat: numpy array of shape (res, res)
        '''
        self.symmetry = symmetry
        self.scale = scale
        self.size = size
        self.noise = noise
        self.res = res
        self.shape = shape
    
    def __str__(self):
        return f"{self.symmetry}grids_scale{self.scale}_size{self.size[0]}-{self.size[1]}_noise{self.noise[0]}-{self.noise[1]}_res{self.res}_{self.shape}"
    
    def __repr__(self):
        return f"{self.symmetry}grids_scale{self.scale}_size{self.size[0]}-{self.size[1]}_noise{self.noise[0]}-{self.noise[1]}_res{self.res}_{self.shape}"
    
    def _get_intervals(self, res: int, scale:float, noise: float):
        start = np.random.rand()*scale
        intervals = np.arange(start, res, scale)
        intervals = intervals+np.random.randn(len(intervals))*noise
        mask = (intervals >= 0) & (intervals < res)
        intervals = intervals[mask]
        intervals = np.round(intervals).astype(int)
        return intervals
    
    def generate(self, tmp):
        angle = np.random.rand()*360 # sample the rotation angle
        mres = 2*self.res # increase resolution by 2
        mat = np.zeros((mres, mres))
        if self.symmetry == 2:
            intervals_h = self._get_intervals(mres, self.scale, self.noise[0])
            mat[intervals_h] = 1

        elif self.symmetry == 4:
            intervals_h = self._get_intervals(mres, self.scale, self.noise[0])
            intervals_v = self._get_intervals(mres, self.scale, self.noise[1])
            X, Y = np.meshgrid(intervals_h, intervals_v)
            mat[X, Y] = 1

        elif self.symmetry == 6:
            intervals_h = self._get_intervals(mres, self.scale, self.noise[0])
            intervals_v = self._get_intervals(mres, self.scale*np.sqrt(3)/2, self.noise[1])
            X, Y = np.meshgrid(intervals_h, intervals_v)
            X[::2] += np.round(self.scale//2).astype(int)
            mask = (X >= 0) & (X < mres)
            X = X[mask]
            Y = Y[mask]
            mat[X, Y] = 1

        if self.shape == 'gaussian':
            mat = scipy.ndimage.gaussian_filter(mat,sigma=self.size)
        else:
            raise ValueError(f"Shape {self.shape} not supported")

        mat = scipy.ndimage.rotate(mat, angle, reshape=False)
        mat = mat[self.res//2:-self.res//2, self.res//2:-self.res//2]

        return mat