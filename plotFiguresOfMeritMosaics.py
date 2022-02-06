#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Helper function to generate Figure-of-Merit (FoM) plots. 
Called by 'runToolkit.py' and 'runToolkitExtended.py'.

For the full procedure, see "README.md".

For method details, please see "A toolkit for data-driven discovery of governing equations in 
high-noise regimes" (2022) by C.B. Delahunt and J.N. Kutz.

For these FoM plots to appear in separate windows, first run in the python console:
%matplotlib qt
                  
Copyright (c) 2021 Charles B. Delahunt.  delahunt@uw.edu
MIT License
"""

import numpy as np
from matplotlib import pyplot as plt 
    
def plotFoMSubplot_fn(y, x, ax, titleStr, yLims, varNames, c, thisMark, legendLoc='',
                      idealLineVal=None, offset=0):
    """
    

    Parameters
    ----------
    y : np.array of floats (maybe of ints). The FoM's values
    x : np.array (vector) of ints (the iteration numbers)
    ax: matplotlib axis object
    titleStr : str
    yLims : list of floats, length = 2
    varNames : list of str
    c : list of str (color names)
    thisMark : str (marker type)
    legendLoc : str, default = ''
    idealLineVal : float, default = None-> don't draw a line at the ideal value.
    offset : float (offsets marker locations on x-axis)
     
    Returns
    -------
    None. Renders a subplot

    """
    numVars = len(varNames)
    if np.ndim(y) == 1: # case: a dummy value was put in because evolutions were skipped.
        temp = np.zeros((len(y), numVars))
        for i in range(len(y)):
            temp[i, :] = y[i]  
        y = temp
    if np.ndim(y) == 3:  # weird ju-ju
        y = y[:, 0, :]
    if idealLineVal != None:
        ax.plot(x, idealLineVal * np.ones(x.shape), 'k--')
    for i in range(numVars):
        ax.scatter(x + i * 0.05 + offset, y[:, i],  marker=thisMark, color=c[i], s=12, 
                   label=varNames[i])
    ax.set_title(titleStr, fontsize=12, fontweight='bold')
    if len(legendLoc) > 0:
        ax.legend(fontsize=12, loc=legendLoc)
    ax.set_ylim(yLims)
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.grid(b=True, which='both', axis='x')
    
# --------- End of plotFoMSubplot_fn---------------------

def plotFiguresOfMeritMosaics_fn(d):
    """  
    Function to create Figure-of-Merit (FoM) mosaics after a set of runs. 'toolkitRunWrapper' 
    generates, for each training trajectory, two things: a sequence of progressively sparser models 
    created by linear fits to that trajectory's derivatives with progressively culledfunctionals; 
    and sequences of FoMs for these models, related to how well predicted trajectories evolved by 
    these models match the training trajectory and the validation trajectories. 
    This function plots these FoM sequences in a set of subplots. This enables the user to see 
    which models best balance good predicted trajectories and sparse discovered libraries. It also 
    allows the user to spot potentially valuable functionals (eg by a sudden drop in FoMs when that 
    functional was culled); for this latter purpose, the user must also consult the text history
    outputted by 'tookitRunWrapper', which records which functionals were culled at each iteration.
    
    This function is typically used as follows. 
    1. It is called at the end of 'toolkitRunWrapper'. The user then examines the mosaics
    to select a 'best' model (noted by iteration number) for each training trajectory.
    enters 
    2. 'plotSelectedIterations' is run with these best model numbers as input. This (a) plots 
    predicted trajectories of the 'best' models; and (b) writes to console an 
    'initialFunctionsToUseArray', which is the union of the 'best' models' discovered libraries. 
    3. This array is entered as a 'user entry' in 'toolkitRunWrapper', which is then run again 
    with the constrained union library. 
    4. Repeat steps 1 and 2, to decide final best models.
    
    Mosaic format: For each train trajectory, plot one mosaic 5 x 4: first two columns = home 
    trajectory FoM sequences, last 2 columns = results on each val trajectory, using the home 
    trajectory model.
    
    Parameters
    ----------
    d : dict of FoM sequences

    Returns
    -------
    None. Generates mosaics of plots, one for each training trajectory.

    """ 
    # Unpack d, the dataDict: 
    seed = d['seed']   
    numTrajTrain = d['numTrajTrain']
    variableNames = d['variableNames'] 
    functionList = d['functionList'] 
    
    indsValAll = d['indsValAll'] 
    historyWhichIterAll = d['historyWhichIterAll'] 
    historyCoeffArrayAll = d['historyCoeffArrayAll'] 
    historyMinHistCorrelationForEvolutionsAll = d['historyMinHistCorrelationForEvolutionsAll'] 
    historyInBoundsFoMAll = d['historyInBoundsFoMAll'] 
    historyInEnvelopeFoMAll = d['historyInEnvelopeFoMAll'] 
    historyStdDevFoMAll = d['historyStdDevFoMAll'] 
    historyHistogramCorrelationFoMAll = d['historyHistogramCorrelationFoMAll'] 
    historyXDotDiffEnvelopeAll = d['historyXDotDiffEnvelopeAll'] 
    historyMinHistCorrelationForEvolutionsValAll = \
        d['historyMinHistCorrelationForEvolutionsValAll'] 
    historyInBoundsFoMValAll = d['historyInBoundsFoMValAll']
    historyInEnvelopeFoMValAll = d['historyInEnvelopeFoMValAll']
    historyFftCorrelationFoMAll = d['historyFftCorrelationFoMAll']
    historyXDotHistogramCorrelationFoMAll = d['historyXDotHistogramCorrelationFoMAll']
    historyStdDevFoMValAll = d['historyStdDevFoMValAll']
    historyHistogramCorrelationFoMValAll = d['historyHistogramCorrelationFoMValAll']
    historyFftCorrelationFoMValAll = d['historyFftCorrelationFoMValAll']
    historyXDotHistogramCorrelationFoMValAll = d['historyXDotHistogramCorrelationFoMValAll']
    historyXDotDiffEnvelopeValAll = d['historyXDotDiffEnvelopeValAll']
    
    markerList = ['o', 'v', 's', 'P', 'd']  
    
    for traj in range(numTrajTrain):
        indsVal = indsValAll[traj]  # to determine which symbol to use
        
        # 1. FoM for results on xTrain:
        tag = 'Train ' + str(traj) + '. ' 
        thisMark = 'o'
        colorList = ['r', 'b', 'g', 'm', 'c']
        
        fig, axs = plt.subplots(nrows=6, ncols=3, sharex=True, figsize=(9, 12))
        
        xVals = np.array(historyWhichIterAll[traj])  # same for all subplots. Val runs only 
        # (confirm): xVals may be one too long for salvaged crashed runs (since last iteration not 
        # recorded due to the crash). This is checked in the Val section below.
        
        # Make the various subplots. For each, define the y-values (FoM values) and some params,
        # then call the subplot function.
        
        # [0, 0]
        r = 0  # row in mosaic
        c = 0  # col in mosaic
        historyCoeffArray = historyCoeffArrayAll[traj]
        this = np.array(historyCoeffArray)  # numCulls x numVariables x numFunctionals
        this = np.sum(np.abs(this) > 1e-5, axis=2)  # numCulls x numVariables
        
        titleStr = tag + '(seed ' + str(seed) + '). Number of functionals per variable'
        yLims = [-0.1, len(functionList) + 0.1]
        legendLoc = 'upper right'
        idealLineVal = None
        plotFoMSubplot_fn(this, xVals, axs[r, c], titleStr, yLims, variableNames, colorList, 
                          thisMark, legendLoc, idealLineVal)
        
        # FoMs for derivative estimates:
         
        # [1, 0]:
        r = 1
        c = 0
        historyXDotDiffEnvelope = historyXDotDiffEnvelopeAll[traj]
        this = np.array(historyXDotDiffEnvelope)
         
        titleStr = tag + 'xDots 80th percentile of errors, ideal = 0'
        yLims = [-0.1, 1.5] # This plot did not have a defined yLim range
        legendLoc = ''
        idealLineVal = 0
        plotFoMSubplot_fn(this, xVals, axs[r, c], titleStr, yLims, variableNames, colorList, 
                          thisMark, legendLoc, idealLineVal)
         
        # [2, 0]:
        r = 2
        c = 0
        historyXDotHistogramCorrelationFoM = historyXDotHistogramCorrelationFoMAll[traj]
        this = np.array(historyXDotHistogramCorrelationFoM)
         
        titleStr = tag + 'xDot histogram correlation, ideal = 1'
        yLims = [-0.1, 1.5]
        legendLoc = ''
        idealLineVal = 1
        plotFoMSubplot_fn(this, xVals, axs[r, c], titleStr, yLims, variableNames, colorList, 
                          thisMark, legendLoc, idealLineVal)
             
        # [3, 0]:
        r = 3
        c = 0
        thisValList = np.array(historyXDotDiffEnvelopeValAll[traj])
        for v in range(numTrajTrain - 1):
            thisMark = markerList[indsVal[v]]
            this = thisValList[:, v,]
            
            titleStr = tag + 'xDots 80th percentile of errors, ideal = 0'
            yLims = [-0.1, 1.5]
            legendLoc = ''
            idealLineVal = 0
            offset = -0.1 + v * 0.2  # to offset the different validation run results
            plotFoMSubplot_fn(this, xVals, axs[r, c], titleStr, yLims, variableNames, colorList, 
                              thisMark, legendLoc, idealLineVal, offset)
             
        # [4, 0]:
        r = 4
        c = 0
        thisValList = np.array(historyXDotHistogramCorrelationFoMValAll[traj])
        for v in range(numTrajTrain - 1):
            thisMark = markerList[indsVal[v]]
            this = thisValList[:, v,] 
            
            titleStr = tag + 'xDot histogram correlation, ideal = 1'
            yLims = [-0.1, 1.5]
            legendLoc = ''
            idealLineVal = 1
            offset = -0.1 + v * 0.2  # to offset the different validation run results
            plotFoMSubplot_fn(this, xVals, axs[r, c], titleStr, yLims, variableNames, colorList, 
                              thisMark, legendLoc, idealLineVal, offset)
              
        #%% FoMs for the predicted Training trajectory:
        # [0, 1]:
        r = 0
        c = 1
        historyMinHistCorrelationForEvolutions = historyMinHistCorrelationForEvolutionsAll[traj] 
        this = np.array(historyMinHistCorrelationForEvolutions)
        
        titleStr = tag + 'Histogram correlation between evolutions, ideal = 1'
        yLims = [-0.1, 1.5]
        legendLoc = ''
        idealLineVal = 1
        plotFoMSubplot_fn(this, xVals, axs[r, c], titleStr, yLims, variableNames, colorList, 
                          thisMark, legendLoc, idealLineVal)
         
        # [1, 1]:
        r = 1
        c = 1
        historyInBoundsFoM = historyInBoundsFoMAll[traj]
        this = np.array(historyInBoundsFoM) 
        
        titleStr = tag + 'In-bounds fraction, ideal = 1'
        yLims = [-0.1, 1.1]
        legendLoc = ''
        idealLineVal = 1
        plotFoMSubplot_fn(this, xVals, axs[r, c], titleStr, yLims, variableNames, colorList, 
                          thisMark, legendLoc, idealLineVal)
         
        # [2, 1]:
        r = 2
        c = 1
        historyInEnvelopeFoM = historyInEnvelopeFoMAll[traj]
        this = np.array(historyInEnvelopeFoM)
        
        titleStr = tag + 'In-envelope fraction, ideal = 1'
        yLims = [-0.1, 1.1]
        legendLoc = ''
        idealLineVal = 1
        plotFoMSubplot_fn(this, xVals, axs[r, c], titleStr, yLims, variableNames, colorList, 
                          thisMark, legendLoc, idealLineVal)
         
        # [3, 1]:
        r = 3
        c = 1
        historyStdDevFoM = historyStdDevFoMAll[traj]
        this = np.array(historyStdDevFoM)
        this[this > 2] = 1
        this[this < -2] = -1  # not necessary
        
        titleStr = tag + 'Relative error of std dev, ideal = 0'
        yLims = [-0.1, 2.1]
        legendLoc = ''
        idealLineVal = 0
        plotFoMSubplot_fn(this, xVals, axs[r, c], titleStr, yLims, variableNames, colorList, 
                          thisMark, legendLoc, idealLineVal)
         
        # [4, 1]:
        r = 4
        c = 1
        historyHistogramCorrelationFoM = historyHistogramCorrelationFoMAll[traj]
        this = np.array(historyHistogramCorrelationFoM)
         
        titleStr = tag + 'Histogram correlation, ideal = 1'
        yLims = [-0.1, 1.5]
        legendLoc = ''
        idealLineVal = 1
        plotFoMSubplot_fn(this, xVals, axs[r, c], titleStr, yLims, variableNames, colorList, 
                          thisMark, legendLoc, idealLineVal)
         
        # [5, 1]:
        r = 5
        c = 1
        historyFftCorrelationFoM = historyFftCorrelationFoMAll[traj]
        this = np.array(historyFftCorrelationFoM)
         
        titleStr = tag + 'FFT correlation, ideal = 1'
        yLims = [-0.1, 1.5]
        legendLoc = ''
        idealLineVal = 1
        plotFoMSubplot_fn(this, xVals, axs[r, c], titleStr, yLims, variableNames, colorList, 
                          thisMark, legendLoc, idealLineVal)
        
        #%% FoMs for the predicted Validation trajectories: 
            
        # Plot all val trajectories in one plot, using shapes to distinguish them (colors stand
        # for variables, as with train trajectory results).            
        tag = 'Val ' 
        
        # When a run crashes and we wish to salvage it, some histories will be shorter, so reduce
        # xVals by 1 in this case:
        if len(np.array(historyMinHistCorrelationForEvolutionsValAll[traj])[:,0,0]) < len(xVals):
            xVals = xVals[0:-1]
            
        # skip the first box (col 3, row 0) since the functional count is same as for home traj.
        # [0, 2]:
        r = 0
        c = 2
        thisValList = np.array(historyMinHistCorrelationForEvolutionsValAll[traj])
        # numIters x numValTrajectories x numVars array
        for v in range(numTrajTrain - 1):
            thisMark = markerList[indsVal[v]]
            this = np.array(thisValList[:, v, ])
             
            titleStr = tag + 'Histogram correlation between evolutions, ideal = 1'
            yLims = [-0.1, 1.5]
            legendLoc = ''
            idealLineVal = 1
            offset = -0.1 + v * 0.2  # to offset the different validation run results
            plotFoMSubplot_fn(this, xVals, axs[r, c], titleStr, yLims, variableNames, colorList, 
                              thisMark, legendLoc, idealLineVal, offset)
         
        # [1, 2]:
        r = 1
        c = 2
        thisValList = np.array(historyInBoundsFoMValAll[traj])
        for v in range(numTrajTrain - 1):
            thisMark = markerList[indsVal[v]]
            this = thisValList[:, v,] 
             
            titleStr = tag + 'In-bounds fraction, ideal = 1'
            yLims = [-0.1, 1.1]
            legendLoc = ''
            idealLineVal = 1
            offset = -0.1 + v * 0.2  # to offset the different validation run results
            plotFoMSubplot_fn(this, xVals, axs[r, c], titleStr, yLims, variableNames, colorList, 
                              thisMark, legendLoc, idealLineVal, offset)
            
        # [2, 2]:
        r = 2
        c = 2
        thisValList = np.array(historyInEnvelopeFoMValAll[traj])
        for v in range(numTrajTrain - 1):
            thisMark = markerList[indsVal[v]]
            this = thisValList[:, v,]
             
            titleStr = tag + 'In-envelope fraction, ideal = 1'
            yLims = [-0.1, 1.1]
            legendLoc = ''
            idealLineVal = 1
            offset = -0.1 + v * 0.2  # to offset the different validation run results
            plotFoMSubplot_fn(this, xVals, axs[r, c], titleStr, yLims, variableNames, colorList, 
                              thisMark, legendLoc, idealLineVal, offset)
             
        # [3, 2]:
        r = 3
        c = 2
        thisValList = np.array(historyStdDevFoMValAll[traj])
        for v in range(numTrajTrain - 1):
            thisMark = markerList[indsVal[v]]
            this = thisValList[:, v,]
            this[this > 2] = 1
            this[this < -2] = -1  # not necessary
            
            titleStr = tag + 'Relative error of std dev, ideal = 0'
            yLims = [-0.1, 1.1]
            legendLoc = ''
            idealLineVal = 0
            offset = -0.1 + v * 0.2  # to offset the different validation run results
            plotFoMSubplot_fn(this, xVals, axs[r, c], titleStr, yLims, variableNames, colorList, 
                              thisMark, legendLoc, idealLineVal, offset)
             
        # [4, 2]:
        r = 4
        c = 2
        thisValList = np.array(historyHistogramCorrelationFoMValAll[traj])
        for v in range(numTrajTrain - 1):
            thisMark = markerList[indsVal[v]]
            this = thisValList[:, v,]
            
            titleStr = tag + 'Histogram correlation, ideal = 1'
            yLims = [-0.1, 1.5]
            legendLoc = ''
            idealLineVal = 1
            offset = -0.1 + v * 0.2  # to offset the different validation run results
            plotFoMSubplot_fn(this, xVals, axs[r, c], titleStr, yLims, variableNames, colorList, 
                              thisMark, legendLoc, idealLineVal, offset)
             
        # [5, 2]:
        r = 5
        c = 2
        thisValList = np.array(historyFftCorrelationFoMValAll[traj])        
        for v in range(numTrajTrain - 1):
            thisMark = markerList[indsVal[v]]
            this = thisValList[:, v,] 
            
            titleStr = tag + 'FFT correlation, ideal = 1'
            yLims = [-0.1, 1.5]
            legendLoc = ''
            idealLineVal = 1
            offset = -0.1 + v * 0.2  # to offset the different validation run results
            plotFoMSubplot_fn(this, xVals, axs[r, c], titleStr, yLims, variableNames, colorList, 
                              thisMark, legendLoc, idealLineVal, offset)
        
        plt.tight_layout(pad=0.5)
        fig.show()  # Show mosaic for this home trajectory.

#%%
# MIT license:
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
# associated documentation files (the "Software"), to deal in the Software without restriction, including 
# without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to 
# the following conditions: 
# The above copyright notice and this permission notice shall be included in all copies or substantial 
# portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN 
# AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
