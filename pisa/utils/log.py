#
# utils.py
#
# This module sets up the logging system by looking for a "logging.json"
# configuration file. It will search (in this order) the local directory, $PISA
# and finally the package resources. The loggers found in there will be lifted
# to the module namespace.
# 
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# date:   2014-10-17

import logging
import logging.config

#Add a trace level
logging.TRACE= 5 
logging.addLevelName(logging.TRACE, "TRACE")
def trace(self, message, *args, **kws):
    self.log(logging.TRACE, message, *args, **kws) 
logging.Logger.trace = trace
logging.RootLogger.trace = trace
logging.trace = logging.root.trace

#Don't move these up, as "trace" might be used in them
from pisa.utils.jsons import from_json
from pisa.resources.resources import find_resource

#Get the logging configuration
#Will search in local dir, $PISA and finally package resources
logconfig = from_json(find_resource('logging.json'))

#Setup the logging system with this config
logging.config.dictConfig(logconfig)

#Make the loggers public
#In case they haven't been defined, this will just inherit from the root logger
physics = logging.getLogger('physics')
profile = logging.getLogger('profile')


def set_verbosity(verbosity):
    '''Overwrite the verbosity level for the root logger
       Verbosity should be an integer with the levels just below.
    '''
    #define verbosity levels
    levels = {0:logging.WARN,
              1:logging.INFO,
              2:logging.DEBUG,
              3:logging.TRACE}

    #Overwrite the root logger with the verbosity level
    logging.root.setLevel(levels[min(3,verbosity)])
