import numpy as np
import matplotlib as mpl
# headless mode
mpl.use('Agg')
# fonts
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
#mpl.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib.offsetbox import AnchoredText

class plotter(object):

    def __init__(self, outdir='.', stamp='PISA cake test', size=(8,8), fmt='pdf', log=True, label='# events'):
        self.outdir = outdir
        self.stamp = stamp
        self.fmt = fmt
        self.mapset = None
        self.size = size
        self.fig = None
        self.log = log
        self.label = label

    def init_fig(self):
        if self.fig is not None:
            plt.clf()
            plt.close(self.fig)
        self.fig, self.axes = plt.subplots(1,1,figsize=self.size)
        self.fig.patch.set_facecolor('none')

    def add_mapset(self, mapset):
        self.mapset = mapset

    def plot_2d_maps(self):
        for map in self.mapset:
            self.init_fig()
            self.plot_2d_map_on_axis(map, plt.gca())
            self.dump(map.name)

    def plot_2d_array(self, n_rows, n_cols):
        n = len(self.mapset)
        assert( n <= n_cols * n_rows)
        self.size = (n_cols*8, n_rows*8)
        self.init_fig()
        plt.tight_layout()
        h_margin = 1. / self.size[0]
        v_margin = 1. / self.size[1]
        self.fig.subplots_adjust(hspace=0.3, wspace=0.3, top=1-v_margin, bottom=v_margin, left=h_margin, right=1-h_margin)
        for i, map in enumerate(self.mapset):
            plt.subplot(n_rows,n_cols,i+1)
            self.plot_2d_map_on_axis(map, plt.gca())
        self.dump('test')

    def plot_2d_map_on_axis(self, map, axis):
        bins = [map.binning[name] for name in map.binning.names]
        bin_edges = map.binning.bin_edges
        cmap = np.log10(map.hist) if self.log else map.hist
        #cmap = np.ma.masked_invalid(cmap) if not invalid else cmap

        extent = [np.min(bin_edges[0].m), np.max(bin_edges[0].m), np.min(bin_edges[1].m), np.max(bin_edges[1].m)]
        # needs to be flipped for imshow
        img = plt.imshow(cmap.T,origin='lower',interpolation='nearest',extent=extent,aspect='auto', cmap='rainbow')
        axis.set_xlabel(bins[0].label)
        axis.set_ylabel(bins[1].label)
        if bins[0].is_log:
            axis.set_.xscale('log')
        if bins[1].is_log:
            axis.set_yscale('log')
        col_bar = plt.colorbar(format=r'$10^{%.1f}$') if self.log else plt.colorbar()
        a_text = AnchoredText(self.stamp + '\n' + r'$%s$'%map.tex, loc=2, frameon=False)
        axis.add_artist(a_text)
        if self.label:
            col_bar.set_label(self.label)
        

    def dump(self,fname):
        plt.savefig(self.outdir+'/'+fname+'.'+self.fmt, dpi=150, edgecolor='none',facecolor=self.fig.get_facecolor())
