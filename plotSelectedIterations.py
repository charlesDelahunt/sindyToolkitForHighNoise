#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 
Run and plot evolutions for models at chosen iterations, selected from FoM mosaics outputted by a 
previous run by 'toolkitRunWrapper.py'. 
The key User Entries are:
    iterationNumbersToPlot: tuple of ints, one int for each training trajectory.
    savedHistoriesFile: the .pickle outputted by 'toolkitRunWrapper.py'
 
For method details, please see "A toolkit for data-driven discovery of governing equations in 
high-noise regimes" (2022) by C.B. Delahunt and J.N. Kutz.

For these plots to appear in separate windows, run in the python console:
%matplotlib qt
                  
Copyright (c) 2021 Charles B. Delahunt.  delahunt@uw.edu
MIT License
"""

import numpy as np
import pickle

from toolkitSupportFunctions import printModel_fn, evolveModel_fn, \
    calculateDerivativesFromModel_fn, replaceNonLibraryFunctionals_fn, \
    findSpansOfFunctionals_fn, findSpansOfFunctionalsLeaveOneOut_fn, \
    findClosestLinearlyEquivalentVersion_fn
from plotVariousTimeSeriesFunction import plotVariousTimeseries_fn

#%%

""" ------------------------- USER ENTRIES --------------------------------------------- """
     
# Choose one best iteration per training trajectory based on FoM mosaics of the initial run:       
iterationNumbersToPlot = (35, 35, 35)  #!!!!!#  

# The saved file from the initial run (i.e. 'pickleFilename' from toolkitRunWrapper):
savedHistoriesFile = 'runResults_lorenz_initialRun_0000.pickle'  #!!!!!#  
 
# Important  parameter for oracle assessment:
rSqThresh = 0.95 #!!!!!#  If the R squared of a functional's linear dependence is above this, it 
# is considered good enough to use in substitutions to find a version of the discovered model 
# closer to 'true'.
 
tolRatio = 0.98  # Stopping criterion for finding closest equivalent model (oracle assessment)
stepSize = 0.25  # when changing coeffs, move the chosen fnal this fraction towards true

verbose = False  # whether to give diagnostic text output to console 
 
# CAUTION! 'plotLinearDependenceThreshold' < 1 can make LOTS of plots if the library is big, due to
# many strong linear dependencies: 
plotLinearDependenceThreshold = 1.1 # 0.95  # Controls which linearly dependent functional 
# fits or not. > 1 disables. 

showTimeSeriesPlotsFlag = [1, 1, 1, 1, 1, 1, 1, 1, 1]  # 1 -> plot, 0 -> ignore.  
# Key:
# 0 = original training sets, clean and noisy.
# 1 = derivatives of training sets, true and predicted.
# 2 = predicted trajectories on training sets, as both train and as val trajectories.
# 3 = test set predicted derivatives.
# 4 = test set predicted trajectories.
# 5 = 3-D plots, as follows:
#   5a = original training sets, clean and noisy.
#   5b = smoothed trajectories of training sets. 
#   5c = predicted trajectories of training sets.
#   5d = predicted trajectories of training sets, as both train and as val trajectories.
# 6 = 3-D plots of predicted test set trajectories.
# 7 = histograms of FFT power, trajectory values, and derivative values. For each variable.

# After calculating the linearly equivalent models closest to the 'true' model, maybe plot their 
# time-series to see if the behavior is the same as the starting models:
plotClosestEquivalentModelTimeseriesFlag = False

""" ------------------------- END USER ENTRIES --------------------------------------------- """


""" ------------------------------ BEGIN MAIN ---------------------------------------------- """
 
#%% Load and unpack the results saved from the initial run:
with open(savedHistoriesFile, 'rb') as f:
    h = pickle.load(f)

# 'seed' is used (a) for repeatability if wished; and (b) to ensure unique saved result filenames.
seed = np.random.randint(0, high = 1e4)
np.random.seed(seed)
outputFilename = 'outputFileForSelectAndCombineBestModels_' + str(seed)
print('Sending text to ' + outputFilename + ', based on initial runs recorded in ' + \
      h['outputFilename'])

# Unpack 'h':    
historyCoeffArrayAll = h['historyCoeffArrayAll']
historyProtectedFnsArrayAll = h['historyProtectedFnsArrayAll']
historyWhichIterAll = h['historyWhichIterAll']   
historyWhichCullAll = h['historyWhichCullAll']
historyInBoundsFoMAll = h['historyInBoundsFoMAll'] 
historyInEnvelopeFoMAll = h['historyInEnvelopeFoMAll'] 
historyStdDevFoMAll = h['historyStdDevFoMAll'] 
historyMeanFoMAll = h['historyMeanFoMAll'] 
historyMedianFoMAll = h['historyMedianFoMAll']
historyFftCorrelationFoMAll = h['historyFftCorrelationFoMAll'] 
historyFftPowerAll = h['historyFftPowerAll'] 
historyHistogramCorrelationFoMAll = h['historyHistogramCorrelationFoMAll']  
historyHistogramsAll = h['historyHistogramsAll']
historyHistogramBinsAll = h['historyHistogramBinsAll'] 
historyMinHistCorrelationForEvolutionsAll = h['historyMinHistCorrelationForEvolutionsAll']
historyXDotInBoundsFoMAll = h['historyXDotInBoundsFoMAll'] 
historyXDotDiffEnvelopeAll = h['historyXDotDiffEnvelopeAll']
historyXDotHistogramCorrelationFoMAll = h['historyXDotHistogramCorrelationFoMAll']
historyXDotHistogramsAll = h['historyXDotHistogramsAll'] 
historyXDotHistogramBinsAll = h['historyXDotHistogramBinsAll'] 
 
# for xVal: 
historyCoeffArrayValAll = h['historyCoeffArrayValAll'] 
historyProtectedFnsArrayValAll = h['historyProtectedFnsArrayValAll']
historyWhichIterValAll = h['historyWhichIterValAll']
historyWhichCullValAll = h['historyWhichCullValAll'] 
historyInBoundsFoMValAll = h['historyInBoundsFoMValAll'] 
historyInEnvelopeFoMValAll = h['historyInEnvelopeFoMValAll']
historyStdDevFoMValAll = h['historyStdDevFoMValAll']
historyMeanFoMValAll = h['historyMeanFoMValAll'] 
historyMedianFoMValAll = h['historyMedianFoMValAll']
historyFftCorrelationFoMValAll = h['historyFftCorrelationFoMValAll']
historyFftPowerValAll = h['historyFftPowerValAll']
historyHistogramCorrelationFoMValAll = h['historyHistogramCorrelationFoMValAll']
historyHistogramsValAll = h['historyHistogramsValAll']
historyHistogramBinsValAll = h['historyHistogramBinsValAll']
historyMinHistCorrelationForEvolutionsValAll = h['historyMinHistCorrelationForEvolutionsValAll']
historyXDotDiffEnvelopeValAll = h['historyXDotDiffEnvelopeValAll']
historyXDotInBoundsFoMValAll = h['historyXDotInBoundsFoMValAll']
historyXDotHistogramCorrelationFoMValAll = h['historyXDotHistogramCorrelationFoMValAll'] 
historyXDotHistogramsValAll = h['historyXDotHistogramsValAll']
historyXDotHistogramBinsValAll = h['historyXDotHistogramBinsValAll'] 

# Original system info: 
systemInfoStr = h['systemInfoStr']
numTrajTrain = h['numTrajTrain']
variableNames = h['variableNames']
functionList = h['functionList']
trueSystemCoeffArray = h['trueSystemCoeffArray']
trueSystemLibraryArray = h['trueSystemLibraryArray']
numVars = len(variableNames)
simInitCondsAll = h['simInitCondsAll']
LAll = h['LAll'] 
recipes = h['recipes'] 
fomTimepointInds = h['fomTimepointInds'] 
derivAndFnalSmoothType = h['derivAndFnalSmoothType'] 
hammDerivs = h['hammDerivs'] 
trueLib = h['trueLib'] 
pointWeightsForFomAll = h['pointWeightsForFomAll'] 
LNoisyAll = h['LNoisyAll'] 
dt = h['dt']
marginInSecs = h['marginInSecs']
margin = int(marginInSecs / dt)
numSecsInTrain = h['numSecsInTrain']
numSecsInTest = h['numSecsInTest']
numTrajTest = h['numTrajTest']
tTrain = h['tTrain']
xTrainNoisyOriginalAll = h['xTrainNoisyOriginalAll']
xTrainCleanAll = h['xTrainCleanAll']
xTrainAll = h['xTrainAll']
xDotTrainUnsmoothedAll = h['xDotTrainUnsmoothedAll']
xDotTrainAll = h['xDotTrainAll']
xDotTrainTrueAll = h['xDotTrainTrueAll']
indsValAll = h['indsValAll']
tTest = h['tTest']
xTestAll = h['xTestAll']
xDotTestTrueAll = h['xDotTestTrueAll'] 
x0TestAll = h['x0TestAll'] 
LTestAll = h['LTestAll'] 

xTrainFftPowerAll = h['xTrainFftPowerAll']
xTrainHistogramAll = h['xTrainHistogramAll']
xTrainHistogramBinsAll = h['xTrainHistogramBinsAll']
xDotTrainHistogramAll = h['xDotTrainHistogramAll']
xDotTrainHistogramBinsAll = h['xDotTrainHistogramBinsAll']

# A few parameters we'll need:
windowLengthForFomEnvelopes = h['windowLengthForFomEnvelopes']
maxFnValRatio = h['maxFnValRatio']
minNumStartIndsToUse = h['minNumStartIndsToUse']
   
# First populate coeffArrayAll:
coeffArrayAll = []
for traj in range(len(iterationNumbersToPlot)):
    i2 = iterationNumbersToPlot[traj] 
    coeffArrayAll.append(historyCoeffArrayAll[traj][i2])

# Make a dict of several parameters to feed into functions:
params = dict()
params['functionList'] = functionList 
params['variableNames'] = variableNames  
params['outputFilename'] = outputFilename  
params['plottingThreshold'] = plotLinearDependenceThreshold  
params['windowLengthForFomEnvelopes'] = windowLengthForFomEnvelopes 
params['dt'] = dt  
params['margin'] = margin  
params['maxFnValRatio'] =  maxFnValRatio 
params['minNumStartIndsToUse'] = minNumStartIndsToUse
params['tolRatio'] = tolRatio
params['functionList'] = functionList
params['rSqThresh'] = rSqThresh
params['stepSize'] = stepSize
params['verbose'] = verbose      
#%% Remind ourselves of the target system:
print(systemInfoStr)

# And print the selected models for each training traj:
for i in range(h['numTrajTrain']):
    print('\n' + 'Discovered model for training trajectory ' + str(i) + ', from iteration ' + \
          str(iterationNumbersToPlot[i]) + ':')
    thisCoeffArray = historyCoeffArrayAll[i][iterationNumbersToPlot[i]]
    mS = printModel_fn(thisCoeffArray, h['variableNames'], h['functionList'])
    print(mS)

#%% Create the union of the sparse libraries. The console printout can be copied and pasted into
# the 'initialFunctionsToUseArray' variable in User Entries of 'toolkitRunWrapper.py', to 
# constrain the starting library in a new run to a best subset of functionals, based on the best 
# models from the first run.

unionFunctionsToUseArray = np.zeros(coeffArrayAll[0].shape, dtype=bool)
for j in range(numTrajTrain):
    unionFunctionsToUseArray = np.logical_or(unionFunctionsToUseArray, 
                                             coeffArrayAll[j].astype(bool))  
print('\n' + 'unionFunctionsToUseArray, from models specified by iterationNumbersToPlot = ' + \
      str(iterationNumbersToPlot) + ':' + '\n')

# Format the array (by adding commas) for easy copy-and-paste back into 'runToolkit.py':
unionArrayPrintStr = str(unionFunctionsToUseArray).replace('False ','False, ')
unionArrayPrintStr = unionArrayPrintStr.replace('True ', 'True, ').replace('False\n', 'False,\n')
unionArrayPrintStr = unionArrayPrintStr.replace('True\n','True,\n').replace(']\n', '],\n')    
print(unionArrayPrintStr)  

#%% Plot predicted time-series of various kinds for Train, Val, and Test. Two steps:
# 1. evolve each model on training, val, and test trajectories  
# 2. call the plotting function 

#%% 1. Evolutions:
# For each trajectory, use the model trained on that trajectory to (a) evolve the training
# trajectory and predict derivatives; (b) do the same on each validation trajectory (ie the
# other training trajectories); (c) optionally, do the same on each test trajectory.

# Note: All of the time-series below start at 'margin' and stop at '-margin'.
# We use three lists, indexed by which training trajectory: one list for 'home trajectory'
# evolutions, one list for Val evolutions, and one list for test trajectory evolutions.  Note 
# that each entry in the Val and Test lists may contain more than one result, if there are 
# >= 3 training trajectories (therefore >= 2 val trajectories) or >= 2 Test trajectories.
 
# Initialize some storage:
xTrainEvolvedAll = []
xDotTrainPredictedAll = []
xDotTrainSmoothedAll = []

xValEvolvedAll = []  # Each entry will be a list of (numTrajTrain - 1) evolved val trajectories
# using the j'th model.
xDotValPredictedAll = []  # Each entry will be a list of (numTrajTrain - 1) arrays of xDot
# predictions, each array found by applying the j'th model to one of the val trajectories.
xDotValSmoothedAll = []  # ditto

xTestEvolvedAll = []  # Each entry will be a list of test trajectories (len = numTrajTest)
# as evolved by one of the models.
xDotTestPredictedAll = []  # Ditto

for j in range(numTrajTrain):
    # update to the correct model, ie the result of processing the j'th trajectory:
    modelCoefs = coeffArrayAll[j]
    modelActiveLib = np.abs(modelCoefs) > 1e-5

    this = printModel_fn(modelCoefs, variableNames, functionList)
    print('\n' + 'traj = ' + str(j) + ': \n' + this)

    # 6a. Evolve the j'th training trajectory:
    x = xTrainAll[j]
    t = tTrain
    initConds = simInitCondsAll[j].copy()
    L = LAll[j]

    # Evolve starting from initConds:
    xEvolvedFromMargin = evolveModel_fn(modelCoefs, recipes, initConds,
                                                  t[fomTimepointInds])
    xDotPredictedFromMargin = calculateDerivativesFromModel_fn(x[fomTimepointInds, :],
                                                               modelCoefs,
                                                               L[fomTimepointInds, :])
    xTrainEvolvedAll.append(xEvolvedFromMargin)
    xDotTrainPredictedAll.append(xDotPredictedFromMargin)
    # Make a smoothed version of xDotPredicted, if we are not already smoothing with hamming:
    xDotSmoothedFromMargin = xDotPredictedFromMargin.copy()
    if derivAndFnalSmoothType != 'hamming':
        for i in range(xDotSmoothedFromMargin.shape[1]):
            temp = xDotPredictedFromMargin[:, i]
            xDotSmoothedFromMargin[:, i] = \
                np.convolve(temp - temp[0], hammDerivs, mode = 'same') + temp[0]
    xDotTrainSmoothedAll.append(xDotSmoothedFromMargin)

    # Evolve the Validation trajectories using this j'th model:
    xValEvolvedThese = []
    xDotValPredictedThese = []

    tVal = tTrain[fomTimepointInds]
    indsVal = indsValAll[j]
    for k in range(numTrajTrain - 1):  # go through the validation trajectories
        xVal = xTrainAll[indsVal[k]]
        LVal = LAll[indsVal[k]]  # values of the functionals on this trajectory
        initConds = simInitCondsAll[indsVal[k]].copy()

        xValEvolvedFromMargin = evolveModel_fn(modelCoefs, recipes,
                                                         initConds, tVal)
        xDotValPredictedFromMargin = \
            calculateDerivativesFromModel_fn(xVal[fomTimepointInds, :], modelCoefs,
                                             LVal[fomTimepointInds, :])

        xValEvolvedThese.append(xValEvolvedFromMargin)
        xDotValPredictedThese.append(xDotValPredictedFromMargin)

    xValEvolvedAll.append(xValEvolvedThese)
    xDotValPredictedAll.append(xDotValPredictedThese)

    # Evolve the Test Set trajectory with this model:
    xTestEvolvedThese = []
    xDotTestPredictedThese = []

    for k in range(numTrajTest):
        x0Test = x0TestAll[k]
        xTest = xTestAll[k]
        LTest = LTestAll[k]
        # evolve the x, y, z data:
        xTestEvolved = evolveModel_fn(modelCoefs, recipes, x0Test, tTest)
        xDotTestPredicted = calculateDerivativesFromModel_fn(xTest, modelCoefs, LTest)
        xTestEvolvedThese.append(xTestEvolved)
        xDotTestPredictedThese.append(xDotTestPredicted)

    xTestEvolvedAll.append(xTestEvolvedThese)
    xDotTestPredictedAll.append(xDotTestPredictedThese)

# Train, Val and Test trajectories have now been evolved by each model (one model per training
# trajectory).

# Add evolved timeseries to 'h', which is the argin dict for the plotting function: 
h['iterationNumbersToPlot'] = iterationNumbersToPlot
h['xTrainEvolvedAll'] = xTrainEvolvedAll 
h['xDotTrainPredictedAll'] = xDotTrainPredictedAll
h['xDotTrainSmoothedAll'] = xDotTrainSmoothedAll 
h['xValEvolvedAll'] = xValEvolvedAll
h['xDotValPredictedAll'] = xDotValPredictedAll
h['xDotValSmoothedAll'] = xDotValSmoothedAll 
h['xTestEvolvedAll'] = xTestEvolvedAll 
h['xDotTestPredictedAll'] = xDotTestPredictedAll 

#%% 2. Plot various time-series of these evolutions: 
 
plotVariousTimeseries_fn(showTimeSeriesPlotsFlag, h, [])

#%% Linear dependence operations:
    
# Do the following, for each variable in each model. Use the 'home' trajectories (linear 
# dependence of functionals depends on the trajectory). 'Rsq' denotes R squared value.

# 1. Print the Rsq of each library functional, as linearly fit by the discovered libraries.
# 2. Print the Rsq of each discovered functional relative to the other discovered functionals, in 
#    leave-one-out fashion.
# 3. Print the Rsq of each library functional, as linearly fit by the 'true' libraries.
# 4. Print the Rsq of each 'true' library functional, relative to the other 'true' library 
#    functionals, in leave-one-out fashion.
# 5. Given a discovered model, find the linearly equivalent version that is closest to the 
#    'true' system. For oracle assessment.
# 6. Evolve and plot this closest model.
 
#%% Linear redundancy relative to the discovered sparse libraries:
    
print('--------------------------------------------------' + '\n' + \
      'Linear combinations for rejected functionals in terms of discovered functionals:')
         
for traj in range(numTrajTrain):
    functionalsInLibraryArray = coeffArrayAll[traj].astype(bool)    
    if np.sum(functionalsInLibraryArray.flatten()) > 0:
        L = LAll[traj][fomTimepointInds, :] 
        LNoisy = LNoisyAll[traj][fomTimepointInds, :] 
        pointWeightsForFom = pointWeightsForFomAll[traj]
        includeConstantFlag = False
# 1.    Assess the linear redundancy of the full library, relative to the discovered sparse 
        # libraries of each model, by looking at Rsq of linear fits over the different 
        # trajectories. 
        # This gives an idea of which rejected functionals could have plausibly given the same 
        # results as the discovered functionals, due to linear dependence. 
        rSq, beta0, betas = \
            findSpansOfFunctionals_fn(functionalsInLibraryArray, L.copy(), LNoisy, 
                                      pointWeightsForFom, includeConstantFlag, params)
            
# 2.    Also see whether any retained functionals are in the span of the other retained 
        # functionals (in leave-one-out fashion). 
        # This gives an idea of redundancy within each discovered model's sparse libraries.
        rSqLoo, beta0Loo, betaLoo = \
            findSpansOfFunctionalsLeaveOneOut_fn(functionalsInLibraryArray, L.copy(), LNoisy, 
                                                 pointWeightsForFom, includeConstantFlag, params)
    else:
        print('All variables eliminated.')  
    print('model # ' + str(traj) + ' Rsq values of library functionals:')
    print(str(np.round(rSq, 2))) 
    
#%% Linear redundancy relative to the 'true' libraries:

print('--------------------------------------------------' + '\n' + \
      'Linear combinations for "untrue" functionals in terms of "true" functionals:') 
 
numFnals = len(functionList)
for traj in range(numTrajTrain):
    print('\n' + 'train traj ' + str(traj) + ':' )
# 3. Print out linear combos for each 'untrue' functional in terms of 'true' libraries, as  
    # well as the Rsq of these fits. The goal is to assess what amount of linear dependence
    # our full library (including the discovered functionals) has relative to the 'true' libraries.
    L = LAll[traj][fomTimepointInds, :]
    LNoisy = LNoisyAll[traj][fomTimepointInds, :]
    pointWeightsForFom = pointWeightsForFomAll[traj]
    includeConstant = False
    rSq, betaZeros, betas = \
            findSpansOfFunctionals_fn(trueSystemLibraryArray, L.copy(), LNoisy, 
                                      pointWeightsForFom, includeConstantFlag, params)   
    for v in range(numVars):
        print(variableNames[v] + "':")
        for i in range(numFnals):
            if rSq[v, i] < 1: #  rSq[v, i] > 0.0 and 
                betaStr = functionList[i] + ' = '
                for j in range(len(betas[v, i])):
                    signStr = ' + '
                    if np.sign(betas[v, i][j]) < 0:
                        signStr = ' - ' 
                    betaStr = betaStr + signStr + str(np.abs(np.round(betas[v, i][j], 2))) + \
                        trueLib[v][j] 
                print(betaStr + '   Rsq = ' + str(np.round(rSq[v, i], 2)))
    
# 4. Also, print out linear combos for each 'true' library functional as a leave-one-out, to 
    # assess the redundancy in the 'true' library:
    rSqLoo, beta0Loo, betasLoo = \
        findSpansOfFunctionalsLeaveOneOut_fn(trueSystemLibraryArray, L.copy(), LNoisy, 
                                             pointWeightsForFom, includeConstant, params)  
    print('\n' + 'leave-one-out:')
    for v in range(numVars):
        print(variableNames[v] + "':")
        libFnalInds = np.where(trueSystemLibraryArray[v, :])[0]  # the indices of true functionals
        for i in range(len(libFnalInds)):
            looInd = libFnalInds[i]   #  index of the left-out fnal
            keptLibInds = libFnalInds[libFnalInds != looInd]  # indices of the kept fnals
            indicesLessI = list(range(0, i)) + list(range(i + 1,len(libFnalInds)))
            keptLibNames = np.array(trueLib[v])[indicesLessI]
            if True: # rSqLoo[v, looInd] > 0.0:  # good enough fit to be worth printing out
                betaStr = functionList[looInd] + ' = '  
                for j in range(len(keptLibInds)): 
                    signStr = ' + '
                    if np.sign(betasLoo[v, looInd][j]) < 0:
                        signStr = ' - ' 
                    betaStr = betaStr + signStr + \
                        str(np.abs(np.round(betasLoo[v, looInd][j], 2))) + \
                        keptLibNames[j]
                print(betaStr + '   rSq = ' + str(np.round(rSqLoo[v, libFnalInds[i]], 2)))
    
print('----------------------------------------------------')

#%% Oracle assessment of discovered model:
    
# Given a discovered model with coefficents, find the closest equivalent to 'true' model using 
# in-span combinations with Rsq > some threshold. First substitute 'true' functionals for 'untrue'
# functionals using 'betas'; then use 'betasLoo' to find the combination that has lowest max 
# relative error per variable.
# This block is only for evaluation of discovered models. It assumes oracle knowledge of the true 
# libraries and coefficients. 
# NOTE: We ignore 'betaZeros', ie we assume that the in-span fits have no constant term, because
# the discovered model can have a constant term, so one of the possible fitting functionals is a 
# constant.
  
#%% 5. Calculate closest legit fit to 'true' model:
 
linComboCoeffArrays = []  # To store results

# Note that each trajectory has slightly different linear dependencies since these are based on 
# the trajectory values. 

print('\n' + 'Calculate closest legit fit to "true" model. ' + \
      'Use linear combinations with R squared > ' + str(rSqThresh))

rawCoeffArrayAll = coeffArrayAll.copy()  # so that the next block (plotting transformed models)
# does not erase the original raw versions.

for traj in range(numTrajTrain):
    print('\n' + 'train trajectory ' + str(traj) + ':')
    newCoeffs = rawCoeffArrayAll[traj].copy()  # to store the transformed model coeffs
    currentModelStr = printModel_fn(newCoeffs, variableNames, functionList, precision=2)
    L = LAll[traj][fomTimepointInds, :]
    LNoisy = LNoisyAll[traj][fomTimepointInds, :]
    pointWeightsForFom = pointWeightsForFomAll[traj]
    
    # Print current model and errors: 
    print('Starting model:')
    print(currentModelStr)  
    print('Starting errors: ')
    for v in range(numVars):
        tC = trueSystemCoeffArray[v, :]   # tC = True Coeffs
        tF = trueSystemLibraryArray[v, :] 
        coeffErr = ((newCoeffs[v, tF] - tC[tF]) / tC[tF])
        print(variableNames[v] + "', lib = " + \
              str(np.array(functionList)[tF]).replace('[','').replace(']','') +  '.    ' + \
              'Errors = ' + \
              str(np.round(100*coeffErr)).replace('[','').replace(']','').replace('.','% '))
    
    # Now start the transforms:
    params['plottingThreshold'] = 2  # > 1 disables plots of linearly dependent functional fits.
    finalErrors = []
    trueFunctionals = []
    rSq, betaZeros, betas = \
            findSpansOfFunctionals_fn(trueSystemLibraryArray, L.copy(), LNoisy, 
                                      pointWeightsForFom, includeConstantFlag, params)  
    rSqLoo, beta0Loo, betasLoo = \
        findSpansOfFunctionalsLeaveOneOut_fn(trueSystemLibraryArray, L.copy(), LNoisy, 
                                             pointWeightsForFom, includeConstantFlag, params) 
    # Note that betasLoo is an array of objects, where each object is a vector of beta 
    # coefficients from a LOO fit.
    
    for v in range(numVars):
        if verbose:
            print('\n' + variableNames[v] + "'" + ':')
        dC = newCoeffs[v, :]   # dC = Discovered Coeffs for variable v (vector)
        dF = dC.astype(bool)  # boolean version
        tC = trueSystemCoeffArray[v, :]   # tC = True Coefficents for variable v
        tF = trueSystemLibraryArray[v, :]   # boolean version, same as tC.astype(bool)
        
        # Substitute out non-'true'-library functionals using the in-span fits where possible,
        # i.e. if their Rsquared values are high enough:
        nC = replaceNonLibraryFunctionals_fn(dC, tF, rSq[v, :], betas[v, :], functionList, 
                                             rSqThresh, verbose) 
        # 'nC' means 'new Coefficients', a linearly equivalent version of dC
        
        # If nC now contains any true library functionals (not including a possible constant 
        # functional), find the least-error linear combination. The fussing is to exclude the 
        # constant function from the check, since we cannot substitute for it.
        constantInd  = np.where(np.array(functionList) == '1')[0][0]  
        nCTemp = nC.copy()
        nCTemp[constantInd] = False
        if np.sum(np.logical_and(nCTemp.astype(bool), tF)) > 0:  # i.e. nC contains at least one 
        # true functional (see 'tF' above).
            # Do iterative transforms, on true library functionals only, that reduce the biggest 
            # coefficient error:
            nC, coeffErrR = findClosestLinearlyEquivalentVersion_fn(nC, tC, rSqLoo[v, :], 
                                                                    betasLoo[v, :], params)
            
            # Update newCoeffs for this variable:            
            newCoeffs[v, :] = nC
        
        finalErrors.append(np.abs((nC[tF] - tC[tF]) / tC[tF]))
        trueFunctionals.append(tF)  # for all variables
    
        # End of 'for v in range(numVars)' loop
        
    linComboCoeffArrays.append(newCoeffs)
    
    newModelStr = printModel_fn(newCoeffs, variableNames, functionList, precision=2)
    
    print('Final model: ')
    print(newModelStr)
    
    print('Final errors: ')
    for v in range(numVars):
        coeffErr = finalErrors[v]
        tF = trueSystemLibraryArray[v, :]
        print(variableNames[v] + "', lib = " + \
              str(np.array(functionList)[tF]).replace('[','').replace(']','') +  \
             '.      Errors = ' + \
              str(np.round(100*coeffErr)).replace('[','').replace(']','').replace('.','% '))

# Print true model again to compare.
print(systemInfoStr) 
                   
#%% 5. Evolve and plot these new models. In principle they are equivalent to the original models, 
# but check whether the new evolved time-series match the original ones. 
# This section of code largely repeats the evolution code above (lines 165 - 275). 
    
coeffArrayAll = linComboCoeffArrays.copy()

# For each trajectory, use the model trained on that trajectory to (a) evolve the training
# trajectory and predict derivatives; (b) do the same on each validation trajectory (ie the
# other training trajectories); (c) optionally, do the same on each test trajectory.

# Note: All of the time-series below start at 'margin' and stop at '-margin'.
# We use three lists, indexed by which training trajectory: one list for 'home trajectory'
# evolutions, one list for Val evolutions, and one list for test trajectory evolutions.  Note 
# that each entry in the Val and Test lists may contain more than one result, if there are 
# >= 3 training trajectories (therefore >= 2 val trajectories) or >= 2 Test trajectories.
 
# Initialize some storage:
xTrainEvolvedAll = []
xDotTrainPredictedAll = []
xDotTrainSmoothedAll = []

xValEvolvedAll = []  # Each entry will be a list of (numTrajTrain - 1) evolved val trajectories
# using the j'th model.
xDotValPredictedAll = []  # Each entry will be a list of (numTrajTrain - 1) arrays of xDot
# predictions, each array found by applying the j'th model to one of the val trajectories.
xDotValSmoothedAll = []  # ditto

xTestEvolvedAll = []  # Each entry will be a list of test trajectories (len = numTrajTest)
# as evolved by one of the models.
xDotTestPredictedAll = []  # Ditto

for j in range(numTrajTrain):
    # update to the correct model, ie the result of processing the j'th trajectory:
    modelCoefs = coeffArrayAll[j]
    modelActiveLib = np.abs(modelCoefs) > 1e-5

    this = printModel_fn(modelCoefs, variableNames, functionList)
    print('\n' + 'traj = ' + str(j) + ': \n' + this)

    # 6a. Evolve the j'th training trajectory:
    x = xTrainAll[j]
    t = tTrain
    initConds = simInitCondsAll[j].copy()
    L = LAll[j]

    # Evolve starting from initConds:
    xEvolvedFromMargin = evolveModel_fn(modelCoefs, recipes, initConds,
                                                  t[fomTimepointInds])
    xDotPredictedFromMargin = calculateDerivativesFromModel_fn(x[fomTimepointInds, :],
                                                               modelCoefs,
                                                               L[fomTimepointInds, :])
    xTrainEvolvedAll.append(xEvolvedFromMargin)
    xDotTrainPredictedAll.append(xDotPredictedFromMargin)
    # Make a smoothed version of xDotPredicted, if we are not already smoothing with hamming:
    xDotSmoothedFromMargin = xDotPredictedFromMargin.copy()
    if derivAndFnalSmoothType != 'hamming':
        for i in range(xDotSmoothedFromMargin.shape[1]):
            temp = xDotPredictedFromMargin[:, i]
            xDotSmoothedFromMargin[:, i] = \
                np.convolve(temp - temp[0], hammDerivs, mode = 'same') + temp[0]
    xDotTrainSmoothedAll.append(xDotSmoothedFromMargin)

    # Evolve the Validation trajectories using this j'th model:
    xValEvolvedThese = []
    xDotValPredictedThese = []

    tVal = tTrain[fomTimepointInds]
    indsVal = indsValAll[j]
    for k in range(numTrajTrain - 1):  # go through the validation trajectories
        xVal = xTrainAll[indsVal[k]]
        LVal = LAll[indsVal[k]]  # values of the functionals on this trajectory
        initConds = simInitCondsAll[indsVal[k]].copy()

        xValEvolvedFromMargin = evolveModel_fn(modelCoefs, recipes,
                                                         initConds, tVal)
        xDotValPredictedFromMargin = \
            calculateDerivativesFromModel_fn(xVal[fomTimepointInds, :], modelCoefs,
                                             LVal[fomTimepointInds, :])

        xValEvolvedThese.append(xValEvolvedFromMargin)
        xDotValPredictedThese.append(xDotValPredictedFromMargin)

    xValEvolvedAll.append(xValEvolvedThese)
    xDotValPredictedAll.append(xDotValPredictedThese)

    # Evolve the Test Set trajectory with this model:
    xTestEvolvedThese = []
    xDotTestPredictedThese = []

    for k in range(numTrajTest):
        x0Test = x0TestAll[k]
        xTest = xTestAll[k]
        LTest = LTestAll[k]
        # evolve the x, y, z data:
        xTestEvolved = evolveModel_fn(modelCoefs, recipes, x0Test, tTest)
        xDotTestPredicted = calculateDerivativesFromModel_fn(xTest, modelCoefs, LTest)
        xTestEvolvedThese.append(xTestEvolved)
        xDotTestPredictedThese.append(xDotTestPredicted)

    xTestEvolvedAll.append(xTestEvolvedThese)
    xDotTestPredictedAll.append(xDotTestPredictedThese)

# Train, Val and Test trajectories have now been evolved by each model (one model per training
# trajectory).

# Add evolved timeseries to 'h2', which is the argin dict for the plotting function. 
h2 = h.copy()  # Make a copy, since we need to overwrite the previous evolution time-series
h2['xTrainEvolvedAll'] = xTrainEvolvedAll 
h2['xDotTrainPredictedAll'] = xDotTrainPredictedAll
h2['xDotTrainSmoothedAll'] = xDotTrainSmoothedAll 
h2['xValEvolvedAll'] = xValEvolvedAll
h2['xDotValPredictedAll'] = xDotValPredictedAll
h2['xDotValSmoothedAll'] = xDotValSmoothedAll 
h2['xTestEvolvedAll'] = xTestEvolvedAll 
h2['xDotTestPredictedAll'] = xDotTestPredictedAll 

#%% 2. Plot various time-series of these evolutions: 

if plotClosestEquivalentModelTimeseriesFlag:
    plotVariousTimeseries_fn(showTimeSeriesPlotsFlag, h2, [])

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



    

