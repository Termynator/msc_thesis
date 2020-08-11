import os
import sys
import glob

import numpy
import matplotlib

import esim_py
import data

esim = esim_py.EventSimulator(
    contrast_threshold_pos,  # contrast thesholds for positive 
    contrast_threshold_neg,  # and negative events
    refractory_period,  # minimum waiting period (in sec) before a pixel can trigger a new event
    log_eps,  # epsilon that is used to numerical stability within the logarithm
    use_log,  # wether or not to use log intensity
    )


