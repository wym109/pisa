#! /usr/bin/env python
#
# gradients.py
#
# Tools for building the Fisher Matrix from a set of derivatives and a fiducial
# map.
#
# author: Lukas Schulte - schulte@physik.uni-bonn.de
#         Sebastian Boeser - sboeser@uni-mainz.de
#


# copied from PaPA, TODO: adopt to PISA scheme
def build_fisher_matrix(gradient_maps, fiducial_map, template_settings)

      #Find non-empty bins in flattened map
      nonempty =n.nonzero(fiducial_map['map'].flatten())
      logging.info("Using %u non-empty bins of %u"%(len(nonempty[0]),len(fiducial_map['map'].flatten())))

      #get gradients as calculated above for non-zero bins
      gradients = n.array([ data[parName]['grad'+flavour][nonempty] for parName in data.keys() ])
      # get error estimate from best-fit bin count for non-zero bins
      sigmas = n.sqrt(fiducial_map['map'].flatten()[nonempty])

      #Loop over all parameter per bin (simple transpose) and calculate Fisher
      #matrix per by getting the outer product of all gradients in a bin.
      #Result is sum of matrix for all bins
      for bin_gradients, bin_sigma in zip(gradients.T,sigmas.flatten()):
        fmatrix += n.outer(bin_gradients, bin_gradients)/bin_sigma**2

      #And construct the fisher matrix object
      fisher[flavour] = FisherMatrix(matrix=fmatrix,
                                     parameters=data.keys(),  #order is important here!
                                     best_fits=[fiducial[par]['value'] for par in data.keys()],
                                     labels =  [fiducial[par]['label'] for par in data.keys()] )

  #Finally calculate combined matrix for all flavours
  #(only if there is more than one)
  if len(fisher.keys()) > 1:
    fisher[''] = FisherMatrix(matrix=n.array([f.matrix for f in  fisher.itervalues()]).sum(axis=0),
                              parameters=data.keys(),  #order is important here!
                              best_fits=[fiducial[par]['value'] for par in data.keys()],
                              labels =  [fiducial[par]['label'] for par in data.keys()] )

  #Now add our priors for everything  and calculate the covariances
  logging.info("Calculating covariances...")
  for flavour, fisher_matrix in fisher.iteritems():

    # first add the priors
    for par in data.keys():
      if fiducial[par]['prior'] is not None:
        fisher_matrix.addPrior(par, 'from_settings', fiducial[par]['prior'])

    #Calculate the covariance
    fisher_matrix.calculateCovariance()

  #Return all fisher matrices
  return fisher



