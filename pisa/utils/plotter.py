import numpy as np
from uncertainties import unumpy as unp
import matplotlib as mpl
# headless mode
mpl.use('Agg')
# fonts
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib.offsetbox import AnchoredText

class plotter(object):

    def __init__(self, outdir='.', stamp='PISA cake test', size=(8,8), fmt='pdf', log=True, label='# events', grid=True, ratio=False):
        self.outdir = outdir
        self.stamp = stamp
        self.fmt = fmt
        self.size = size
        self.fig = None
        self.log = log
        self.label = label
        self.grid = grid
        self.ratio = ratio

    # --- helper functions ---

    def init_fig(self,figsize=None):
        ''' clear/initialize figure '''
        if figsize is not None:
            size = figsize
        else:
            size = self.size
        if self.fig is not None:
            plt.clf()
            plt.close(self.fig)
        self.fig, self.axes = plt.subplots(1,1,figsize=size)
        self.fig.patch.set_facecolor('none')

    def add_stamp(self, text=None):
        ''' ad common stamp with text '''
        if text is not None:
            a_text = AnchoredText(self.stamp + '\n' + r'$%s$'%text, loc=2, frameon=False)
        else:
            a_text = AnchoredText(self.stamp, loc=2, frameon=False)
        plt.gca().add_artist(a_text)

    def add_leg(self):
        ''' initialize legend '''
        plt.gca().legend(loc='upper right',ncol=2, frameon=False,numpoints=1)

    def dump(self,fname):
        ''' dump figure to file'''
        plt.savefig(self.outdir+'/'+fname+'.'+self.fmt, dpi=150, edgecolor='none',facecolor=self.fig.get_facecolor())

    # --- 2d plots ---

    def plot_2d_single(self, mapset):
        ''' plot all maps in individual plots '''
        for map in mapset:
            self.init_fig()
            self.plot_2d_map(map)
            self.add_stamp(map.tex)
            self.dump(map.name)

    def plot_2d_array(self, mapset, n_rows, n_cols):
        ''' plot all maps in a single plot '''
        self.plot_array(mapset, n_rows, n_cols,'plot_2d_map')
        self.dump('test2d')

    # --- 1d plots ---

    def plot_1d_single(self, mapset, plot_axis):
        ''' plot all maps in individual plots '''
        for map in mapset:
            self.init_fig()
            self.plot_1d_projection(map, plot_axis)
            self.add_stamp(map.tex)
            self.dump(map.name)

    def plot_1d_array(self, mapset, n_rows, n_cols, plot_axis):
        self.plot_array(mapset, n_rows, n_cols,'plot_1d_projection', plot_axis)
        self.dump('test1d')

    def plot_1d_all(self, mapset, plot_axis):
        ''' all one one canvas '''
        self.init_fig()
        for map in mapset:
            self.plot_1d_projection(map, plot_axis)
        self.add_stamp()
        self.add_leg()
        self.dump('all')

    def plot_1d_stack(self, mapset, plot_axis):
        ''' all maps stacked on top of each other '''
        self.init_fig()
        for i, map in enumerate(mapset):
            for j in range(i):
                map += mapset[j]
            self.plot_1d_projection(map, plot_axis)
        self.add_stamp()
        self.add_leg()
        self.dump('stack')

    def plot_1d_cmp(self, mapset0, mapset1, plot_axis):
        ''' 1d comparisons for two mapsets '''
        for map0, map1 in zip(mapset0, mapset1):
            self.init_fig()
            if self.ratio:
                ax1 = plt.subplot2grid((4,1), (0,0), rowspan=3)
                plt.setp(ax1.get_xticklabels(), visible=False)
            self.plot_1d_projection(map0, plot_axis)
            self.plot_1d_projection(map1, plot_axis, ptype='data')
            self.add_stamp()
            self.add_leg()
            if self.ratio:
                plt.subplot2grid((4,1), (3,0),sharex=ax1)
                self.plot_1d_ratio([map1, map0], plot_axis)
            self.dump('cmp_%s'%map0.name)

    # --- plotting core functions ---

    def plot_array(self, mapset, n_rows, n_cols, fun, *args):
        ''' plot mapset in array using a function fun '''
        n = len(mapset)
        assert( n <= n_cols * n_rows)
        size = (n_cols*self.size[0], n_rows*self.size[1])
        self.init_fig(size)
        plt.tight_layout()
        h_margin = 1. / size[0]
        v_margin = 1. / size[1]
        self.fig.subplots_adjust(hspace=0.3, wspace=0.3, top=1-v_margin, bottom=v_margin, left=h_margin, right=1-h_margin)
        for i, map in enumerate(mapset):
            plt.subplot(n_rows,n_cols,i+1)
            getattr(self, fun)(map, *args)

    def plot_2d_map(self, map):
        ''' plot map on current axis in 2d'''
        axis = plt.gca()
        bins = [map.binning[name] for name in map.binning.names]
        bin_edges = map.binning.bin_edges
        cmap = np.log10(map.hist) if self.log else map.hist
        extent = [np.min(bin_edges[0].m), np.max(bin_edges[0].m), np.min(bin_edges[1].m), np.max(bin_edges[1].m)]
        # needs to be flipped for imshow
        img = plt.imshow(cmap.T,origin='lower',interpolation='nearest',extent=extent,aspect='auto', cmap='rainbow')
        axis.set_xlabel(bins[0].label)
        axis.set_ylabel(bins[1].label)
        if bins[0].is_log:
            axis.set_xscale('log')
        if bins[1].is_log:
            axis.set_yscale('log')
        col_bar = plt.colorbar(format=r'$10^{%.1f}$') if self.log else plt.colorbar()
        if self.label:
            col_bar.set_label(self.label)
        
    def plot_1d_projection(self, map, plot_axis,ptype='hist', **kwargs):
        ''' plot map projected on plot_axis'''
        axis = plt.gca()
        plt_binning = map.binning[plot_axis]
        hist = self.project_1d(map, plot_axis)
        if ptype == 'hist':
            axis.hist(plt_binning.bin_centers, weights=hist, bins=plt_binning.bin_edges, histtype='step', lw=1.5, label=r'$%s$'%map.tex, **kwargs)
        elif ptype == 'data':
            axis.errorbar(plt_binning.bin_centers, hist, fmt='o',color='black', markersize='4', label=r'$%s$'%map.tex, **kwargs)
        axis.set_xlabel(plt_binning.label)
        if self.label:
            axis.set_ylabel(self.label)
        if plt_binning.is_log:
            axis.set_xscale('log')
        if self.log:
            axis.set_yscale('log')
        axis.set_xlim(plt_binning.bin_edges.m[0], plt_binning.bin_edges.m[-1])
        if self.grid:
            plt.grid(True, which="both", ls='-', alpha=0.2)

    def project_1d(self, map, plot_axis):
        hist = map.hist
        plt_axis_n = map.binning.names.index(plot_axis)
        for i in range(len(map.binning)):
            if i == plt_axis_n:
                continue
            hist = np.sum(map.hist, i)
        return hist

    def plot_1d_ratio(self, maps, plot_axis):
        axis = plt.gca()
        map0 = maps[0]
        plt_binning = map0.binning[plot_axis]
        hist0 = self.project_1d(map0, plot_axis)
        axis.set_xlim(plt_binning.bin_edges.m[0], plt_binning.bin_edges.m[-1])
        gmin = 1.0
        gmax = 1.0
        for map in maps[1:]:
            hist1 = self.project_1d(map, plot_axis)
            ratio = hist1/hist0
            mi = np.nanmin(ratio)
            gmin = min(mi, gmin)
            ma = np.nanmax(ratio)
            gmax = max(ma, gmax)
            ratio = np.nan_to_num(ratio)
            axis.hist(plt_binning.bin_centers, weights=ratio, bins=plt_binning.bin_edges, histtype='step', lw=1.5, label=r'$%s$'%map.tex)

        if self.grid:
            plt.grid(True, which="both", ls='-', alpha=0.2)
        self.fig.subplots_adjust(hspace=0)
        axis.set_ylabel('ratio')
        axis.set_xlabel(plt_binning.label)
        # calculate nice scale:
        off = max(gmax-1, 1-gmin)
        axis.set_ylim(1 - 1.2 * off, 1 + 1.2 * off )
        axis.axhline(1.0, color='k')
