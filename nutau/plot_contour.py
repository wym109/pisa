#! /usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os, sys
from scipy.stats import chi2
from scipy import optimize
from scipy.interpolate import griddata
from scipy.ndimage import zoom
from matplotlib.offsetbox import AnchoredText
from matplotlib import colors, ticker, cm
from pisa.utils.fileio import from_file
import matplotlib.lines as mlines

ZOOM = 10

def get_data(dir,x_var,y_var):
    x = []
    y = []
    q = []

    # get llh denominators for q for each seed
    for filename in os.listdir(dir):
        if filename.endswith('.json') and y_var in filename:
            file = from_file(dir +'/'+filename)
            cond = file[0][0]
            if file[0][0].has_key('llh'): metric = 'llh'
            elif file[0][0].has_key('conv_llh'): metric = 'conv_llh'
            elif file[0][0].has_key('chi2'): metric = 'chi2'
            elif file[0][0].has_key('mod_chi2'): metric = 'mod_chi2'
            elif file[0][0].has_key('barlow_llh'): metric = 'barlow_llh'
            flag = False
            name = filename[:-5]
            assert name.split('_')[0] == y_var
            y_val = float(name.split('_')[1])
            y.append(y_val)
            x = cond[x_var][0]
            if 'llh' in metric:
                q.append(2*cond[metric])
            else:
                q.append(-cond[metric])

    y = np.array(y)
    q = [z for (k,z) in sorted(zip(y,q))]
    y.sort()
    q = np.array(q).T
    #print x, y ,q
    x = np.sin(x*np.pi/180.)
    x = np.square(x)
    print x
    print y
    print q

    #smooth out
    xs = zoom(x,ZOOM)
    ys = zoom(y,ZOOM)
    qs = zoom(q,ZOOM)

    return xs, ys, qs

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-d1','--dir1',metavar='dir',help='directory containg output json files', default=None) 
    parser.add_argument('-d2','--dir2',metavar='dir',help='directory containg output json files', default=None) 
    parser.add_argument('-d3','--dir3',metavar='dir',help='directory containg output json files', default=None) 
    parser.add_argument('-d4','--dir4',metavar='dir',help='directory containg output json files', default=None) 
    parser.add_argument('-x',help='outer loop variable', default='') 
    parser.add_argument('-y',help='inner loop variable', default='') 
    parser.add_argument('-t','--tag',help='tag', default='')
    parser.add_argument('-t1','--tag1',help='tag1', default='')
    parser.add_argument('-t2','--tag2',help='tag2', default='')
    parser.add_argument('-t3','--tag3',help='tag1', default='')
    parser.add_argument('-t4','--tag4',help='tag2', default='')
    args = parser.parse_args()
    livetime = 3.


    legs = []
    #colors = ['red', 'black']*2
    colors = ['red', 'grey', 'blue', 'green']
    tags = [args.tag1, args.tag2,args.tag3, args.tag4]
    dirs = []
    for dir in [args.dir1, args.dir2, args.dir3, args.dir4]:
        if dir is not None:
            dirs.append(dir) 


    levels = [2.3,4.61]#,5.99,9.21,19.35]
    flevels = 2*np.logspace(-2,2,100)
    fmt = {2.3:r'$68\%$',4.61:r'$90\%$',5.99:r'$2\sigma$',9.21:'99%',19.35:r'$4\sigma$'}
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    for color, dir, tag in zip(colors, dirs, tags):
        xs, ys, qs = get_data(dir,args.x,args.y)
        #CS = ax1.contour(xs,ys,qs.T, levels=levels, linewidth=2, colors=['darkorange','red','dimgrey','green'])
        CS = ax1.contour(xs,ys,qs.T, levels=levels, linewidth=2, linestyles=[':','-','--'],colors=[color]*len(levels))
        # proxy thing for legend
        legs.append(mlines.Line2D([], [], color=color, label=tag))
        #CS = ax1.contour(x,y,q.T, levels=levels, linewidth=2, colors=['r','orchid','blue','green'])
        #CF = plt.contourf(x,y,q.T, cmap=cm.Set3,levels=flevels)
        ax1.clabel(CS, inline=1, fontsize=10, fmt=fmt)

    ax1.set_xlabel(r'$\sin^2(\theta_{23})$')
    ax1.set_ylabel(r'$\Delta m_{31}^2\ \rm{(10^{-3}eV^2)}$')
    a_text = AnchoredText('Gen2 Phase 1 Preliminary'+'\nNormal mass ordering assumed'+'\n%s years Asimov'%livetime, loc=2, frameon=False)
    ax1.add_artist(a_text)
    ax1.grid()
    ax1.set_xlim(0.3,0.7)
    ax1.set_ylim(2.4,2.9)
    ax1.legend(loc='upper right',ncol=1, frameon=False,numpoints=1,fontsize=10,handles=legs)


    #NOvA & T2K
    theta = np.square(np.sin(np.array([39.23, 46.83])/180.*np.pi))
    dm = np.array([2.67, 2.545])
    text = ['NOvA','T2K']
    ax1.scatter(theta,dm, marker='+', c='k')
    for x,y,t in zip(theta,dm,text):
        ax1.annotate(t, (x,y),fontsize=10,ha='right',textcoords='offset points',xytext=(0, 5))


    plt.show()
    plt.savefig('contour_%s.pdf'%(args.tag), edgecolor='none')
    plt.savefig('contour_%s.png'%(args.tag),dpi=150, edgecolor='none')
