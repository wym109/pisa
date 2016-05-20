import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib.offsetbox import AnchoredText


class plotter(object):

    def __init__(self, outdir='.', stamp='PISA cake test', size=(8,8), fmt='pdf', log=True):
        self.outdir = outdir
        self.stamp = stamp
        self.fmt = fmt
        self.mapset = None
        self.size = size
        self.fig = None
        self.log = log

    def init_fig(self):
        if self.fig is not None:
            plt.clf()
            plt.close(self.fig)
        self.fig, self.axes = plt.subplots(1,1,figsize=self.size)
        self.fig.patch.set_facecolor('none')

    def add_mapset(self, mapset):
        self.mapset = mapset

    def plot_2d(self):
        for map in self.mapset:
            self.plot_2d_map(map)

    def plot_2d_map(self, map):
        self.init_fig()
        bins = [map.binning[name] for name in map.binning.names]
        bin_edges = map.binning.bin_edges
        cmap = np.log10(map.hist) if self.log else map.hist
        #cmap = np.ma.masked_invalid(cmap) if not invalid else cmap

        extent = [np.min(bin_edges[0].m), np.max(bin_edges[0].m), np.min(bin_edges[1].m), np.max(bin_edges[1].m)]
        img = plt.imshow(cmap,origin='lower',interpolation='nearest',extent=extent,aspect='auto', cmap='rainbow')
        plt.xlabel(bins[0].label)
        plt.ylabel(bins[1].label)
        if bins[0].is_log:
            plt.xscale('log')
        col_bar = plt.colorbar(format=r'$10^{%.1f}$') if self.log else plt.colorbar()
        self.dump(map.name)

    def dump(self,fname):
        a_text = AnchoredText(self.stamp, loc=2, frameon=False)
        plt.gca().add_artist(a_text)
        plt.savefig(self.outdir+'/'+fname+'.'+self.fmt, dpi=150, edgecolor='none',facecolor=self.fig.get_facecolor())
