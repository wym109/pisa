#! /usr/bin/env python

######################################################
#                                                    #
# This class represents a Fisher matrix and          #
# provides some typical operations                   #
#                                                    #
# written by Lukas Schulte, Sebastian Boeser         #
# Copyright: University of Bonn, 2013                #
#                                                    #
######################################################

# Copied from PaPA
# TODO: clean up

### IMPORTS ###

import numpy as n
import operator
import copy
import sys
import itertools
import json
from scipy.stats import chi2

from matplotlib import pylab
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.lines import Line2D
from math import pi

from pisa.utils.jsons import from_json, to_json


### HELPER FUNCTIONS ###

def derivative_from_polycoefficients(coeff, loc):
    """
    Return derivative of a polynomial of the form
    
        f(x) = coeff[0] + coeff[1]*x + coeff[2]*x**2 + ...
    
    at x = loc
    """
    
    result = 0.
    
    for n in range(len(coeff))[1:]: # runs from 1 to len(coeff)
        
        result += n*coeff[n]*loc**(n-1)
    
    return result



### FISHER MATRIX CLASS DEFINITION ###

class FisherMatrix:
    
    def __init__(self, matrix=None, parameters=None, best_fits=None, 
                 priors=None, labels=None):
        """
        Construct a fisher matrix object from
           matrix: matrix itself,
           parameters: the identifiers used for each parameter
           best_fits: best fit values for each parameter
        """
        
        self.matrix = n.matrix(matrix)
        self.parameters = list(parameters)
        self.best_fits = list(best_fits)
        self.priors = list(priors) if priors is not None else [None for p in self.parameters]
        self.labels = list(labels) if labels is not None else parameters
        
        #I think we need to check Consistency here  
        self.checkConsistency()
        
        self.calculateCovariance()
        
    
    @classmethod
    def fromFile(cls, filename):
        """
        Load a Fisher matrix from a json file
        """
        
        return cls(**from_json(filename))
    
    
    @classmethod
    def fromPaPAFile(cls, filename):
        """
        Load Fisher matrix from json file
        """
        
        loaded_dict = json.load(open(filename, 'r'))
        
        matrix = n.matrix(loaded_dict.pop('matrix'))
        parameters = loaded_dict.pop('parameters')
        best_fits = loaded_dict.pop('best_fits')
        labels = loaded_dict.pop('labels')
        
        new_fm = cls(matrix=matrix, parameters=parameters, 
                     best_fits=best_fits, labels=labels)
        
        while not len(loaded_dict)==0:
            
            par, prior_dict = loaded_dict.popitem()
            
            #self.parameters.append(par)
            for value in prior_dict.itervalues():
                new_fm.addPrior(par, value)
        
        new_fm.checkConsistency()
        new_fm.calculateCovariance()
        
        return new_fm
    
    
    def __add__(self, other):
        
        # merge parameter lists
        new_params = list(set(self.parameters+other.parameters))
        
        new_best_fits = []
        new_labels = []
        for param in new_params:
            try:
                value = self.getBestFit(param)
                lbl = self.getLabel(param)
            except IndexError:
                value = other.getBestFit(param)
                lbl = other.getLabel(param)
            
            new_best_fits.append(value)
            new_labels.append(lbl)
        
        # generate blank matrix
        new_matrix = n.matrix( n.zeros((len(new_params), len(new_params))) )
        
        # fill new matrix
        for (i,j) in itertools.product(range(len(new_params)), range(len(new_params))):
            
            for summand in [self, other]:
                try:
                    i_sum = summand.getParameterIndex(new_params[i])
                    j_sum = summand.getParameterIndex(new_params[j])
                except IndexError:
                    continue
                
                new_matrix[i,j] += summand.matrix[i_sum, j_sum]
        
        # create new FisherMatrix object
        new_object = FisherMatrix(matrix=new_matrix, parameters=new_params, 
                                  best_fits=new_best_fits, labels=new_labels)
        new_object.checkConsistency()
        new_object.calculateCovariance()
        
        # fill in priors
        for par in new_object.parameters:
            
            for summand in [self, other]:
                
                try:
                    prior_dict = summand.getPriorDict()
                except IndexError:
                    continue
                
                for par, sigma in prior_dict.items():
                    new_object.addPrior(par, sigma)
        
        # ...and we're done!
        return new_object
    
    
    def checkConsistency(self):
        """
        Check whether number of parameters matches dimension of matrix, 
        matrix is symmetrical, and parameter names are unique
        """
        
        if not len(self.parameters)==n.shape(self.matrix)[1]:
            raise IndexError('Number of parameters does not match dimension of Fisher matrix! [%i, %i]' \
                %(len(self.parameters), len(self.matrix)) )
        
        if not n.all(self.matrix.T==self.matrix):
            raise ValueError('Fisher matrix not symmetric!')
        
        if not len(self.parameters)==len(set(self.parameters)):
            raise ValueError('Parameter names not unique! %s' \
                %(n.array2string(n.array(self.parameters))) )
        
        return True
    
    
    def saveFile(self, filename):
        """
        Write Fisher matrix to json file
        """
        
        dict_to_write = {}
        dict_to_write['matrix'] = self.matrix
        dict_to_write['parameters'] = self.parameters
        dict_to_write['best_fits'] = self.best_fits
        dict_to_write['labels'] = self.labels
        dict_to_write['priors'] = self.priors
        
        to_json(dict_to_write, filename)
    
    
    def getParameterIndex(self, par):
        """
        Whether par is already existing in parameter list
        """
        
        if not par in self.parameters:
            raise IndexError('%s not found in parameter list %s'\
                %(par, n.array2string(n.array(self.parameters)) ) )
        else:
            return self.parameters.index(par)
    
    
    ## -> Why on earth would we ever want to do that?
    def renameParameter(self, fromname, toname):
        """
        Rename a parameter
        """
        
        idx = self.getParameterIndex(fromname)
        
        if toname in self.parameters[self.parameters!=fromname]:
            raise ValueError('%s already in parameter list %s'\
                %(toname, n.array2string(n.array(self.parameters)) ) )
        
        self.parameters[idx] = toname
    
    
    def calculateCovariance(self):
        """
        Calculate covariance matrix from Fisher matrix (i.e. invert including priors).
        """
        
        if n.linalg.det(self.matrix)==0:
            raise ValueError('Fisher Matrix is singular, cannot be inverted!')
        
        self.covariance = n.linalg.inv(self.matrix \
                                + n.diag([1./self.getPrior(p)**2 for p in self.parameters]))
    
    
    def getBestFit(self, par):
        """
        Get best fit value for given parameter
        """
        
        idx = self.getParameterIndex(par)
        
        return self.best_fits[idx]
    
    
    def getLabel(self, par):
        """
        Get pretty-print label for given parameter
        """
        
        idx = self.getParameterIndex(par)
        
        return self.labels[idx]
    
    
    def setLabel(self, par, newlabel):
        """
        Change the pretty-print label for given parameter
        """
        
        idx = self.getParameterIndex(par)
        
        self.labels[idx] = newlabel
    
    
    def removeParameter(self, par):
        """
        Remove par from Fisher matrix and recalculate covariance
        """
        
        idx = self.getParameterIndex(par)
        
        # drop from parameter, best fit, and prior list
        self.parameters = list(n.delete(n.array(self.parameters), idx))
        self.best_fits = list(n.delete(n.array(self.best_fits), idx))
        self.labels = list(n.delete(n.array(self.labels), idx))
        self.priors = list(n.delete(self.priors, idx))
        
        # drop from matrix (first row, then column)
        self.matrix = n.delete(n.delete(self.matrix, idx, axis=0), idx, axis=1)
        
        self.calculateCovariance()
    
    
    def setPrior(self, par, sigma):
        """
        Set prior for parameter 'par' to value sigma. If sigma==None, no prior is assumed
        """
        
        idx = self.getParameterIndex(par)
        
        self.priors[idx] = sigma
        
        self.calculateCovariance()
    
    
    def addPrior(self, par, sigma):
        """
        Add a prior of value sigma to the existing one for par (in quadrature)
        """
        
        idx = self.getParameterIndex(par)
        
        if self.priors[idx] is not None:
            self.priors[idx] = 1./n.sqrt(1./self.priors[idx]**2 + 1./sigma**2)
            self.calculateCovariance()
        else:
            self.setPrior(par, sigma)
    
    
    def removeAllPriors(self):
        """
        Remove *all* priors from this Fisher Matrix
        """
        
        self.priors = [None for p in self.parameters]
        
        self.calculateCovariance()
    
    
    def getPrior(self, par):
        """
        List the prior (sigma value) for par
        """
        
        idx = self.getParameterIndex(par)
        prior = self.priors[idx]
        
        if prior is None:
            return n.inf
        else:
            return prior
    
    
    def getPriorDict(self):
        """
        List priors of all parameters (sigma values)
        """
        
        return dict(zip(self.parameters, self.priors))
    
    
    def getCovariance(self, par1, par2):
        """
        Returns the covariance of par1 and par2
        """
        
        #Return the respective element
        idx1, idx2 = self.getParameterIndex(par1), self.getParameterIndex(par2)
        return self.covariance[idx1, idx2]
    
    
    def getVariance(self, par):
        """
        Returns full variance of par
        """
        
        return self.getCovariance(par,par)
    
    
    def getSigma(self, par):
        """
        Returns full standard deviation of par,
        marginalized over all other parameters
        """
        
        return n.sqrt(self.getVariance(par))
    
    
    def getSigmaNoPriors(self, par):
        """
        Returns standard deviation of par, marginalized over all other 
        parameters, but ignoring priors on this parameter
        """
        
        idx = self.getParameterIndex(par)
        
        # make temporary priors with the ones corresponding to par removed
        temp_priors = copy.deepcopy(self.priors)
        temp_priors[idx] = None
        
        # calculate covariance with these priors
        temp_covariance = n.linalg.inv(self.matrix \
                            + n.diag([1./s**2 if s is not None else 0. for s in temp_priors]))
        
        return n.sqrt(temp_covariance[idx,idx])
    
    
    def getSigmaStatistical(self, par):
        """
        Returns standard deviation of par,
        if all other parameters are fixed (i.e. known infinitely well)
        """
        
        idx = self.getParameterIndex(par)
        return 1./n.sqrt(self.matrix[idx,idx])
    
    
    def getSigmaSystematic(self, par):
        """
        Returns standard deviation of par for infinite statistics
        (i.e. systematic error)
        """
        
        return n.sqrt((self.getSigmaNoPriors(par))**2 \
                        - (self.getSigmaStatistical(par))**2)
    
    
    def getErrorEllipse(self, par1, par2, confLevel=0.6827):
        """
        Returns a, b, tan(2 theta) of confLevel error ellipse 
        in par1-par2-plane with:
        
        a: large half axis
        b: small half axis
        tan(2 theta): tilt angle, has to be divided by the aspect
                      ratio of the actual plot before taking arctan
        
        Formulae taken from arXiv:0906.4123
        """
        
        sigma1, sigma2 = self.getSigma(par1), self.getSigma(par2)
        cov = self.getCovariance(par1, par2)
        
        #for this we need sigma1 > sigma2, otherwise just swap parameters
        if sigma1 > sigma2:
          a_sq = (sigma1**2 + sigma2**2)/2. + n.sqrt((sigma1**2 - sigma2**2)**2/4. + cov**2)
          b_sq = (sigma1**2 + sigma2**2)/2. - n.sqrt((sigma1**2 - sigma2**2)**2/4. + cov**2)
        else:
          a_sq = (sigma2**2 + sigma1**2)/2. - n.sqrt((sigma2**2 - sigma1**2)**2/4. + cov**2)
          b_sq = (sigma2**2 + sigma1**2)/2. + n.sqrt((sigma2**2 - sigma1**2)**2/4. + cov**2)

        #Note: this has weird dimensions (actual size of the plot)!
        tan_2_th = 2.*cov / (sigma1**2 - sigma2**2)
        
        # we are dealing with a 2D error ellipse here
        scaling = n.sqrt(chi2.ppf(confLevel, 2))
        
        return scaling*n.sqrt(a_sq), scaling*n.sqrt(b_sq), tan_2_th
    
    
    def getCorrelation(self, par1, par2):
        """
        Returns correlation coefficient between par1 and par2
        """
        
        return self.getCovariance(par1, par2)/(self.getSigma(par1)*self.getSigma(par2))
    
    
    def printResults(self, parameters=None, file=None):
        """
        Prints statistical, systematic errors, priors, best fits 
        for specified (default: all) parameters
        """
        
        pars = parameters if parameters is not None else copy.deepcopy(self.parameters)
        pars.sort()
        
        if file is not None:    # redirect stdout
            orig_stdout = sys.stdout
            sys.stdout = open(file, 'w')
        
        param_width = max([max([len(name) for name in pars]), len('parameters')])
        header = (param_width, 'parameter', 'best fit', 'full', 'stat', 'syst', 'priors')
        print '%*s     %9s     %9s     %9s     %9s     %9s' %header
        print '-'*(70+param_width)
        
        for par in pars:
            result = (param_width, par, self.getBestFit(par), self.getSigma(par),
                      self.getSigmaStatistical(par), self.getSigmaSystematic(par),
                      self.getPrior(par))
            par_str = '%*s    %10.3e     %.3e     %.3e     %.3e     %.3e'%result
            par_str = par_str.replace('inf', 'free')
            print par_str
        
        """
        # needed for PINGU only:
        if 'hierarchy' in pars: 
            
            # calculate proper significance according to arXiv:1305.5150
            sigma_gauss = 1./self.getSigma('hierarchy')
            sigma_bin = conv_sigma_to_bin_sigma(sigma_gauss)
            
            print '\nSignificance of hierarchy measurement: %.2f sigma' %sigma_bin
        """
        
        if file is not None:    # switch stdout back
            sys.stdout = orig_stdout
    
    
    def printResultsSorted(self, par, file=None, latex=False):
        """
        Prints statistical, systematic errors, priors, best fits 
        sorted by parameter par
        """
        
        if file is not None:    # redirect stdout
            orig_stdout = sys.stdout
            sys.stdout = open(file, 'w')
        
        if latex:
            # table header
            print '\\begin{tabular}{lrrrrrr} \n\\toprule'
            print 'Parameter & Impact & Best Fit & Full & Stat. & Syst. & Prior \\\\ \n\\midrule'
        else:
            param_width = max([max([len(name) for name in self.parameters]), len('parameters')])
            header = (param_width, 'parameter', 'impact [%]','best fit', 'full', 'stat', 'syst', 'priors')
            
            print '%*s     %10s     %9s     %9s     %9s     %9s     %9s' %header
            print '-'*(85+param_width)
        
        sorted = self.sortByParam(par)
        
        for (par, impact) in sorted:
        
            # print the line
            if latex:
                result = (self.getLabel(par), impact, self.getBestFit(par), self.getSigma(par),
                          self.getSigmaStatistical(par), self.getSigmaSystematic(par),
                          self.getPrior(par))
                par_str = '%s & %.1f & \\num{%.2e} & \\num{%.2e} & \\num{%.2e} & \\num{%.2e} & \\num{%.2e} \\\\'%result
                par_str = par_str.replace('\\num{inf}', 'free')
            else:
                result = (param_width, par, impact, self.getBestFit(par), self.getSigma(par),
                          self.getSigmaStatistical(par), self.getSigmaSystematic(par),
                          self.getPrior(par))
                par_str = '%*s          %5.1f    %10.3e     %.3e     %.3e     %.3e     %.3e'%result
                par_str = par_str.replace('inf', 'free')
            
            print par_str
        
        if latex:
            # table outro
            print '\\bottomrule \n\\end{tabular}'
        
        if file is not None:    # switch stdout back
            sys.stdout = orig_stdout
    
    
    def sortByParam(self, par):
        """
        Sorts the parameters by their impact on parameter par.
        Relevant quantity is covariance(par,i)/sigma_i.
        
        Returns sorted list of (parameters, impact), par first, 
        then ordered descendingly by impact.
        """
        
        # calculate impact
        impact = dict([[p, self.getCorrelation(p, par)**2 * 100] \
                        for p in self.parameters])
        
        # sort the dict by value
        sorted_impact = sorted(impact.iteritems(), 
                               key=operator.itemgetter(1),
                               reverse=True)
        
        return sorted_impact


# FIXME: Do we need this?
class PrettyFisher:
    '''
    A wrapper class around a fisher matrix that allows to draw
    and pretty print in iypthon
    '''

    def __init__(self, fisher=None, parnames=None, parvalues=None):
        '''Constructor takes
           - fisher: a Fisher matrix object
           - parnames: a list of display names for the parameters
           - parvalues: a list of parameter values (fiducial model)
        '''

        self.fisher = fisher
        self.parnames = parnames
        self.parvalues = parvalues

        #Some consistency checks
        if not (str(self.fisher.__class__) == 'Fisher.Fisher.FisherMatrix'):
            raise ValueError('Expected FisherMatrix object, got %s instead'%(fisher.__class__))

        if not len(self.fisher.parameters)==len(self.parnames):
            raise IndexError('Number of parameters names does not match number of parameters! [%i, %i]' \
                %(len(self.fisher.parameters), len(self.parnames)) )

        if not len(self.fisher.parameters)==len(self.parvalues):
            raise IndexError('Number of default values does not match number of parameters! [%i, %i]' \
                %(len(self.fisher.parameters), len(self.parvalues)) )


    def ipynb_pretty_print(self):
        '''
        Pretty print the matrix for the ipyhton notebook
        '''
        from IPython.display import Latex
        
        #Show the fiducial model
        outstr = r'Fiducial model: \begin{align}'
        for parname, parvalue in zip(self.parnames,self.parvalues):
            outstr += r' %s = %.2e \newline'%(parname,parvalue)
        outstr += r' \end{align} '

        #Now add the fisher matrix itself
        outstr += r'Fisher Matrix: $$ \mathcal{F} = \begin{vmatrix} '
        
        for row in self.fisher.matrix:
            for val in (row.flat[0:-1]).flat:
                  outstr += r' %.2e & '%val

            #Add last value with newline
            outstr += r' %.2e \newline'%row.flat[-1]

        outstr += r'\end{vmatrix} $$'

        return Latex(outstr)


    def draw(self, confLevels=[0.6827,0.997], parameters=None, fontsize=16):
        '''
        Make a nice plot with all the error ellpsises
        '''

        #If no parameters are specified, just use all of them
        if parameters is None:
            parameters = self.fisher.parameters 
            
        #Otherwise, match parvalues and parlabels to given list
        parvalues=[]
        parnames=[]
        for par in parameters:
            idx =  self.fisher.getParameterIndex(par)
            parvalues.append(self.parvalues[idx])
            parnames.append(self.parnames[idx])


        #Make a figure with size matched to the number of parameters
        nPar = len(parameters)
        size = min(nPar-1,8)*4
        fig = pylab.figure(figsize=(size,size))
        #Remove space inbetween the subplots
        fig.subplotpars.wspace=0.
        fig.subplotpars.hspace=0.
        #Define the color arguments for the Ellipses
        ellipseArgs = { 'facecolor' : 'b',
                         'linewidth' : 0 }
        markerArgs = { 'marker':'o',
                       'markerfacecolor': 'r',
                       'linewidth': 0 }
        lineArgs = {'linestyle' : '--',
                    'color' : 'r' }


        #Loop over all parameters
        for idx1, par1 in enumerate(parameters):
            #Loop over all other parameters
            for idx2, par2 in list(enumerate(parameters))[idx1+1:]:

                #Make a new subplot in that subfigure
                axes = pylab.subplot(nPar-1,nPar-1,idx2*(nPar-1) + (idx1-(nPar-2)))
                #Only show tick marks in the left-most column and bottom row
                axes.label_outer()
                axes.tick_params(which='both', labelsize=fontsize-2)
                
                #Draw all the requested sigma levels
                for sigma in sorted(confLevels):
                    #Get the three-sigma error ellipse for that parameter pair
                    semiA, semiB, tilt = self.fisher.getErrorEllipse(par1, par2, confLevel=sigma)
                    ell = Ellipse(xy=(parvalues[idx1],parvalues[idx2]),
                                  width=2*semiA, height=2*semiB, angle=tilt*180./pi,
                                  alpha=1.-sigma, **ellipseArgs)
                    
                    axes.add_artist(ell)

                #Draw a red marker for the fiducial model
                pylab.plot(parvalues[idx1],parvalues[idx2],**markerArgs)
                #Only set labels in the left-most column and bottom row
                if axes.is_last_row():
                    pylab.xlabel(parnames[idx1],fontsize=fontsize)
                if axes.is_first_col():
                    pylab.ylabel(parnames[idx2],fontsize=fontsize)

                #A useful range is best obtained from the marginalized errors on the parameters
                sigma1, sigma2 = self.fisher.getSigma(par1), self.fisher.getSigma(par2)
                #Check the max sigma level that is drawn
                sigmaMax=max(sigmaLevels)
                pylab.xlim(parvalues[idx1]-sigmaMax*sigma1,parvalues[idx1]+sigmaMax*sigma1)
                pylab.ylim(parvalues[idx2]-sigmaMax*sigma2,parvalues[idx2]+sigmaMax*sigma2)

                #Show numbers for parameters in column on top
                if idx2 == idx1+1:
                    sigmaTot = sigma1
                    sigmaStat = self.fisher.getSigmaStatistical(par1)
                    sigmaSys = self.fisher.getSigmaSystematic(par1)
                    pylab.title("%s\n $\mathsf{= %.2f \pm %.2f(stat) \pm %.2f(sys)}$"%
                                (parnames[idx1], parvalues[idx1], sigmaStat, sigmaSys),
                                fontsize=fontsize)

                #Now there is one parameter missing, that we show as right label
                #in the last row
                if axes.is_last_row() and axes.is_last_col(): 
                    sigmaTot = sigma2
                    sigmaStat = self.fisher.getSigmaStatistical(par2)
                    sigmaSys = n.sqrt(sigmaTot**2 - sigmaStat**2)
                    axes.yaxis.set_label_position('right')
                    pylab.ylabel("%s\n $\mathsf{= %.2f \pm %.2f(stat) \pm %.2f(sys)}$"%
                                (parnames[idx2], parvalues[idx2], sigmaStat, sigmaSys),
                                fontsize=fontsize, horizontalalignment='center',
                                rotation=-90.,labelpad=+24)


                #Plot vertical and horizontal range to indicate one-sigma
                # marginalized levels
                pylab.axvline(parvalues[idx1]-sigma1,**lineArgs)
                pylab.axvline(parvalues[idx1]+sigma1,**lineArgs)
                pylab.axhline(parvalues[idx2]-sigma2,**lineArgs)
                pylab.axhline(parvalues[idx2]+sigma2,**lineArgs)

        #Use the top right corner to draw a legend
        axes = pylab.subplot(nPar-1,nPar-1,nPar-1)
        axes.axison = False

        #Create dummies for the legend objects
        legendObjs=[Line2D([0],[0],**markerArgs),
                    Line2D([0],[0],**lineArgs)]
        legendLabels=[r'default value',
                      r'$1\sigma$ stat.+syst.']
        for sigma in sorted(sigmaLevels):
            legendObjs.append(Rectangle([0,0],0,0,alpha=1./sigma,**ellipseArgs))
            legendLabels.append(r'$%u\sigma$ conf. region'%sigma)

        #Draw a legend with all of these
        pylab.legend(legendObjs,legendLabels,
                     loc='center',
                     numpoints=1,
                     frameon=False,
                     prop={'size':fontsize})
        
        return fig

