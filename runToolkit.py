#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 
This is a wrapper to run the SINDy high noise toolkit on various dynamical systems.
This is the first script to run. For the full procedure, see "README.md".

This is the short version of 'runToolkit'. Extra threads and parameters for various experiments
are removed. The extended version of the wrapper is 'runToolkitExtended.py'.

For method details, please see "A toolkit for data-driven discovery of governing equations in 
high-noise regimes" (2022) by C.B. Delahunt and J.N. Kutz.

User entries can be divided as follows:
	a. System details (system type, noise level, extra variables)
	b. Important algorithm parameters (these are marked in the script with "#!!!!!#"): 
	   polynomialLibrarydegree, useFullLibraryFlag (True), initialFunctionsToUseArray, 
       hammingWindowLengthForData, balancedCullNumber, percentileOfImputedValuesForWeights,	
       numEvolutionsForTrainFom, restoreBasedOnFoMsFlag, fomChangesDict. See comments in the 
       script body for usage details.
	c. Unimportant algorithm parameters: Many others, which in general do not need tuning. 

If the script hangs on ivp_solve (used to evolve the models to look at the predicted trajectories)
while processing the final training trajectory, you can often simply kill the script, then run the 
cells corresponding to:
    #%% 6. Now save this trajectory's history lists
    #%% 7. Save the results, for use by 'plotSelectedIterations.py'
    #%% 8. Plot figures of merit mosaics
       
Comment: I don't know how to suppress lsoda-- warnings to console from ivp_solve (they are 
         not printed to file). 
         
For these plots to appear in separate windows, run in the python console:
%matplotlib qt
                  
Copyright (c) 2021 Charles B. Delahunt.  delahunt@uw.edu
MIT License
"""

import sys 
import numpy as np
from scipy.integrate import odeint
from sklearn.linear_model import LinearRegression
from numpy.linalg import lstsq  # since this accepts complex inputs and targets.
from multiprocessing import Pool
import time
import pickle

import warnings
warnings.filterwarnings("ignore", 
                        message='Casting complex values to real discards the imaginary part')
# To ignore console output from using only real part of FFTs. 

from toolkitSupportFunctions import lorenz_fn, dampedHarmonicOscillatorLinear_fn, \
dampedHarmonicOscillatorCubic_fn, threeDimLinear_fn, hopfNormalForm2D_fn, hopfNormalForm3D_fn, \
generateModelStrAndTrueArrays_fn, generateTrueFunctionalAndCoeffArrays_fn, \
generatePolynomialLibrary_fn, calculateLibraryFunctionalValues_fn, makeHammingWindow_fn, \
calculateSlopeAndStd_fn, addGaussianNoise_fn, addWhiteNoise_fn, calcWeightArray_fn, \
cullAndAssessWhetherToRerun_fn, smoothData_fn, parseTimepointsByMagnitudesOfVariables_fn, \
calculateDerivativesFromData_fn, calculateDerivativesFromModel_fn, estimateDerivatives_fn, \
combineSegmentCoeffs_fn, \
printWeightedCoeffModel_fn, evolveModel_fn, generateVariableEnvelopes_fn, \
calculateFiguresOfMerit_fn, findSpansOfFunctionalsLeaveOneOut_fn, \
evolveModelAndCalcFoms_fn, appendToLocalHistograms_fn, \
printModel_fn, calculateFftForRegression_fn
 
from plotFiguresOfMeritMosaics import plotFiguresOfMeritMosaics_fn
  
#%% 

""" USER ENTRIES """

"""Important note: 
There appear to be very many tunable parameters, but in fact most of them don't matter much and 
can be left as-is. The following parameters are most important. They are marked by "#!!!!!#" in 
the code.

System and noise details (various parameters)

polynomialLibraryDegree      # size of library
useFullLibraryFlag           # whether doing an initial (True) or second (False) run
initialFunctionsToUseArray   # constrains the starting library, iff 'useFullLibraryFlag' = False

hammingWindowLengthForData  # controls the amount of low pass filtering 

balancedCullNumber          # imposes some uniformity on the number of terms per variable
percentileOfImputedValuesForWeights # maybe

numEvolutionsForTrainFom    # whether to check for model stability (can affect speed)
doParallelEvolutionsFlag    # this may or may not work on a particular computer (speed only)

restoreBasedOnFoMsFlag      # whether to restore functionals 
fomChangesDict              # which FoMs (and how big a change to these FoMs) can trigger 
                              restorations. 

    """
#%% Define any constraints on the initial library, in terms of functionals to ignore or functionals
# to permanently protect from culling.

# Library:
# For polynomial library (the only library used in this code). All possible monomials <= degree
# will be generated: 
polynomialLibraryDegree = 3  #!!!!!#

# Is this the initial run?
useFullLibraryFlag = True  #!!!!!# 

# Constrain initial library of functionals if wished: 
# If 'useFullLibraryFlag' = True, the full library is used, and 'initialFunctionsArray' (below) 
# gets automatically reset to empty = [], so any values entered are ignored. This is the typical 
# case for an first run.
# If 'useFullLibraryFlag' = False, then 'initialFunctionsArray' defines the subset of the full
# library that the procedure will start with. So 'False' entries mark pre-culled functionals.
# Each row is ordered according to how the library of functionals is ordered, as in 
# 'functionList'. Each row corresponds to a variable, ordered as in 'variableNames'.

# This constrained library (i.e. 'useFullLibraryFlag' = False) can be used in two ways:
# 1. To combine the discovered sparse libraries of multiple training trajectories from a previous 
# run to get a "best of" starting library. To do this, paste in the console printout of 
# 'unionFunctionsToUseArray' outputted by 'plotSelectedIterations.py'. This is a standard step in 
# the toolkit method. 
# 2. It can also be used at any time to remove certain functionals from a full library, e.g. 
# because they are physically impossible.

initialFunctionsToUseArray = \
    np.array([[False,  True,  True, False, False, False, False, False, False, False, False, False,
  False, False, False, False, False, False, False, False],
 [False,  True,  True, False, False, False,  True, False, False, False, False, False,
  False, False, False, False, False, False, False, False],
 [ True, False, False,  True,  True, False, False,  True, False, False, False, False,
  False, False, False, False, False, False, False, False]])  #!!!!!#

# Permanently protected functions: 
# Specify any functionals that should never be culled. Format is either: 1. boolean array 
# with same dimensions and ordering as 'functionsToUseArray' (see that variable's definition 
# below); or 2. np.array([]), if nothing is permanently protected.
permanentlyProtectedFunctions = np.array([])

#---------------------

# 'seed' is used (a) for repeatability if wished; and (b) to ensure unique saved result filenames.
seed = np.random.randint(0, high = 1e4)
np.random.seed(seed)

outputFilenamePrefix = 'outputFile'  # We'll append system name, initial or second run, and seed
pickleFilenamePrefix = 'runResults'  # We'll append the above items plus .pickle

#--------------------------------------

#%% System details:
    
modelSystem = 'lorenz'  # 'lorenz' # 'harmOscLinear' # 'harmOscCubic'  # 'threeDimLinear'
                             # 'hopfNormal2D'  # 'hopfNormal3D'
# NOTE: Parameters for each system are set in a section near the end of user entries.

numTrajTrain = 3  # number of training trajectories (also act as validation)
numTrajTest = 2  # number of test trajectories

# time step:
dt = 0.002

# noise parameters:
noiseFactorsActiveVars = 1 * np.array([1.35, 1.35, 0.5])  #  can be different for each variable.
# Magnitude of noise = noiseFactors*stdDev of clean data;
# Note 1: if there are n < 3 active vars, the first n elements of this vector are used.
# Note 2: high noise means we may want to shrink the data for FoM purposes only, to give a more 
# informative 'in-envelope' FoM. See 'shrinkFomEnvelopeFactor' below.

# Noise factors give the following percentage noise levels:
# Lorenz:
#       [0.7, 0.7, 0.25] -> 50% noise per variable
#       [1.35, 1.35, 0.5] -> 100% in each variable
#       [2, 2, 1 ]-> 150, 150, 200% in x, y, z
#       [3, 3, 1]  -> 210, 210, 200% noise in x, y, z
#       [3.5, 3.5, 1.2] -> 250%,  but solve_ivp hangs up at some point in every run
#       [4.1, 4.1, 1.5] -> 300%, but solve_ivp hangs up at some point in every run
# HarmOscLinear:  [1.5, 1.5] -> 105% in each variable (a bit too much for method)
#                   [1.0, 1.0] -> 70%
# HarmOscCubic: [1.5, 1.5] -> 105% in each variable  (too much for method)
# linear 3D:  [1, 1, 0.65] -> 70% in each variable if length = 10 secs.
#             [1, 1, 0.75] -> 70 in each variable if length = 16
#            [0.72, 0.72, 0.54] -> 50% in each var if length = 16 secs.
# hopf 3D: [0.42, 0.42, 0.3] -> 30% (z trajectories vary)
#          [0.7, 0.7, 0.5] -> 50% (z trajectories vary)

numExtraVars = 0  # These are noise variables added to the system time-series.
# Whenever a variable is eliminated, the procedure starts over with the remaining variables and
# the full functional libraries. So if (as results have shown so far) the procedure reliably 
# knocks out noise variables first, this can be set = 0, w.l.o.g.

# Noise parameters:
noiseType = 'white' # 'white', 'uniform' or 'normal'.
# Parameters to put noise on extra variables:
noiseFactorsExtraVars = np.array([1, 1, 1, 1, 1])  # Base value. Up to 5 extra variables.
extraVarsNoiseFactors = 0.5  # Used for gaussian noise. 1 means use average of non-zero
# variables' noise values.
# The next two values control the amount of noise in extra vars when using white noise, because
# the magnitude of the constant value decides the std dev of the FFT.
lowMeanVal = 5   # for the mean vals of extra vars
highMeanVal = 10

# Variable are named below, based on the chosen system. The active variables are x, y, and maybe  
# z; and any extra noise variables will be w0, w1, w2 etc

#------------------------------------------

# Plotting parameters: 
showFiguresOfMeritMosaicsFlag = True  # usually true. This gives us the history of models as they
# got sparser via culling.
showTimeSeriesPlotsAtEndOfRunFlag = False  # Usually False here, but reset to True when
# 'plotSelectedIterations.py' is run (because then we have chosen a promising model from the
# sequence generated here).

""" Now specify algorithm parameters: """

#%%  Smoothing parameters:

# smoothing window parameters:

# Hamming parameters:  (By far the best smoothing method)
hammingWindowLengthForData = 200 #!!!!!# Better too small than too big (ie it's better to retain 
# high frequency noise than to have a too-strong lowpass filter).
# Lorenz: 100 - 200; harmOcsCubic: 300

hammingWindowLengthForDerivsAndFnals = 20  # 50  # Even though raw data gets smoothed, this still
# matters. It does a last bit of smoothing (non-obvious).
 
marginInSecs = 1 # the number of seconds to throw out at the start and end. A convenience to 
# prevent endpoint artifacts in the smoothing windows, so 0.5 * window length (in secs) suffices.
 
#%% Parameters for linear regression at each iteration, using the current library functions, to  
# update the coefficients. This step does not cull coefficients. Regress on xDot, for each variable
# separately, using the pre-processed trajectories:

regressOnFftAlpha = 0.5  # What fraction of regression weight is on the fft of xDotTrain as
# target vs on xDotTrain as target. (fft regression is an experiment)
fftRegressionTarget = 'magnitude'  # 'complex', 'realOnly', 'magnitude', 'power'  # Regress on the 
# full (real + imag) fft, real part only, magnitude, or power spectrum.
# 'complex' works worse. The other 3 have similar effect.

parseTimepointsByVarValuesFlag = True  # Says whether to use the 'maxFnValRatio' parameter.
maxFnValRatio = 20  # This is used to ignore certain {f, t} as follows: Let f, g be library
# functions and t a timepoint, such that f(t) is small and g(t) is big. If f(t) / g(t) < some
# value tied to maxVarValueRatio, then ignore f at timepoint t during # regression. We count up
# how many fns are available at each timepoint, and ignore timepoints that have too many rejected
# fn. Low parameter value is more restrictive. This parameter has an effect, but I don't know how
# to optimize.

printDiagnosticsForParsingFlag = False  # print how many timepoints each variable's
# regression uses. This is an optional diagnostic to be sure that the exclusion of timepoints 
# based on 'maxFnValRatio' is working.
printWeightedCoefficientModelFlag =  True 

# Weight timepoints for the regressions:
weightTimepointsFlag = True
# 3 ways to scale or clip point weights (can all be used if wished):
scaleByLogFlag = True  # True works best

# Parameters related to doing several regressions, over different collections of timepoints, and 
# deciding the final coeffs via a median-type method:
numRegressionSegments = 15
overlapFraction = 1  # Gives the number of points per segment, via:
# (1 + 2*overlapFraction)*(numTotalPoints / numRegressionSegments).

minNumSegmentResultsToUse = 9 # We remove outlier coeffs down to this number, then take a median.
minNumStartIndsToUse = 1000
snrThreshold = 0.4  # Used to remove segments when combining coefficents from the segment 
# regressions. Not a sensitive parameter if not too small.

# Diagnostic to show spread of coeffs over various segments and to see how segment coeff estimates
# are combining: (To disable: -1 -> print nothing)
printCoeffsForEachSegmentThresh = -1 # 9  # Diagnostic. If this var has fewer library fns than this
# threshold, print the coeffs over each of its fns from each segment, eg to see spread of coeff
# estimates.


numDtStepsForDeriv = 2   # how many dt's to use for euler or 4th order stepping.
numEvolveStepsForEvolutionTarget = 50  # If regressionTarget = 'evolution', how many steps to
# 'evolve' by stringingtogether 4th order derivatives at a sequence of points.


#%% Culling step parameters:

extraNumFnsCulledPerIter = 1   # Invoked for culls done after the first cull (in each
# iteration) which removes zero functions.
minAllowedWeightedCoeff = 0.001  # if a *weighted* coeff is smaller than this, cull the functional.

almostZeroRoundDownThreshold = 1e-5  # In various spots we kill off functionals that have very
# small |coeff|, to handle machine noise.

balancedCullNumber = 3  #!!!!!# To prevent any one variable being culled down to nothing while other
# variables retain large libraries. Set = 100 (ie big) to disable.

stopIfVarIsCulledFlag = True  # prevents wasted time, but it gets automatically set to False if 
# there are extra noise variables, because we expect the system to first kill the noise variables 
# then re-start on the remaining variables.

cullUsingInSpanFlag = True  # As part of iteration, for each variable: test whether any
# active functional is within the span of the remaining active functionals (leave one out). If yes,
# cull it. Break ties by choosing the functional with the lowest weighted coefficient.
inSpanCullThreshold = 0.95 # If Rsq value of fit to a functional is above this, can cull.
# set > 1 to disable.
percentileThresholdForInSpanCulling = 0.2  # weighted coeff must be below this percentile
# (among functionals of that var) to be culled.
printDiagnosticOfInSpanCullFlag = True  # True -> print Rsq values each iteration
#----------------------------------------------------------
# Weight the current coeffs during culling to account for large differences in values taken on by
# the functions they multiply:

# Define a percentile for use in setting functional weights. The goal is to compensate for the
# larger number of functionals with high degree (resulting from combinatorics):
percentileOfImputedValuesForWeights = 10  #!!!!!# a low number shifts preference towards 
# lower-valued functionals, which may effectively mean lower-degree functionals.
# The goal is to offset the combinatorially larger number of high-degree functionals (eg using the
# median would favor high-degree functionals because there are so many).
coeffWeightsCutoffFactor = 10  #  Must be >= 1, where  1 -> No weighting of coeffs. This defines
# the allowable differences in scale for typical functional values. A high value -> big contrast 
# in weights for various functionals. Coeff weights will be clipped at [centralValue/this,
# centralValue*this]
# Note: not weighting the coeffs disadvantages high-degree functionals.

#-------------------------------------------------------------
#%% Figures of Merit (FoM):

useTrueInitialValueForSimFlag = False  # If True, we start the final train set simulation from the
# true, clean data values. If False, we use an estimate (realistic use-case situation). 

windowLengthForFomEnvelopes = 30  # Length of window for max-min envelopes
shrinkFomEnvelopeFactor = 0.1 / max(noiseFactorsActiveVars)
# If the data is very noisy, 'inEnvelopeFoM' means nothing, so shrink the envelope. If the data
# is not as all noisy, increase the envelope size. The goal is to make the 'inEnv' fom meaningful.

# For fft power:
numFftPoints = 50

numEvolutionsForTrainFom = 3  #!!!!!# To check for stability. Downside: unstable evolutions can take a
# long time to run. So ideally add bail-out option to odeint
numEvolutionsForValFom = numEvolutionsForTrainFom

doParallelEvolutionsFlag = False  # This should be faster, but in fact it gets stuck inside
# the parallel loop when using pysindy's built-in odeint. Have not tried directly using solve_ivp.

doNotEvolveAfterInSpanCullFlag = True  # To save time, since in-span culling mostly happens in
# very early stages.
maxNumFunctionalsToEvolve = 100  # Because with very many functionals, solve_ivp can hang,
# ruining the run.  A high number #!!!!!#effectively disables this.

#--------------------------------------------------------------------
# Parameters to restore and protect culled functionals:
restoreBasedOnFoMsFlag = True #!!!!!#
startAfterWhichIter = 8  # do not restore until part-way in
numItersProtection = 3 # Once restored, protect from culling for this many iterations.

# The FoMs to monitor, and a vector that contains (0) their triggering value drops (expressed as a
# fraction of previous iteration's value); and (1) the minimum value for the previous iteration
# (the point is to only restore functionals if the previous result was decent).
# This is only relevant if 'restoreBasedOnFoMsFlag' = True
fomChangesDict = {'inBounds':[0.8, 0.8], 'evolutionsCorrelation':[0.8, 0.98],
                  'histogramCorrelation':[0.8, 0.6], 'inEnvelopeFoM':[0.8, 0.6],
                  'fftPower':[0.8, 0.7],'stdDevFoM':[0.3, 0.8]}  #!!!!!# For restoring purposes,
# stdDevFoM will be subtracted from one, so we look for values close to 0 changing to values far
# from 0. Note that the first arg for stdDevFoM is a threshold on Subtraction, not Ratio (the
# other change detectors use Ratio).
# NOTE: 'results', defined during the culling phase, must have keys that match the keys in
# 'fomChangesDict'. If they do not, an error message will print to console, but still manually 
# comparing 'results' and 'fomChangesDict' before running is best.

#--------------------------------------------------

# Initialize lists to save FoMs from each iteration. This is necessary to do here, so we can then 
# define 'results', the histories relevant to restoring functionals.
historyCoeffArray = []  # to save coeff matrices from each step.
historyProtectedFnsArray = []
historyWhichIter = []
historyWhichCull = []
historyInBoundsFoM = []
historyInEnvelopeFoM = []
historyStdDevFoM = []
historyMeanFoM = []
historyMedianFoM = []
historyFftCorrelationFoM = []
historyFftPower = []
historyHistogramCorrelationFoM = []
historyHistograms = []
historyHistogramBins = []
historyMinHistCorrelationForEvolutions = []
historyXDotDiffEnvelope = []
historyXDotInBoundsFoM = []
historyXDotHistogramCorrelationFoM = []
historyXDotHistograms = []
historyXDotHistogramBins = []

# for xVal:
historyCoeffArrayVal= []  # to save coeff matrices from each step.
historyProtectedFnsArrayVal = []
historyWhichIterVal = []
historyWhichCullVal = []
historyInBoundsFoMVal = []
historyInEnvelopeFoMVal= []
historyStdDevFoMVal = []
historyMeanFoMVal = []
historyMedianFoMVal = []
historyFftCorrelationFoMVal = []
historyFftPowerVal = []
historyHistogramCorrelationFoMVal = []
historyHistogramsVal = []
historyHistogramBinsVal = []
historyMinHistCorrelationForEvolutionsVal = []
historyXDotDiffEnvelopeVal = []
historyXDotInBoundsFoMVal = []
historyXDotHistogramCorrelationFoMVal = []
historyXDotHistogramsVal = []
historyXDotHistogramBinsVal = []

# Once iterations start, we'll define 'results' for each trajectory. The keys must
# match the keys in 'fomChangesDict'. 

# MAYBE TO DO: Weighting currently uses *global* values of x, y, etc to weight the coefficients. 
# Most likely there is a better way. If using sequential segments, we could weight according to 
# each segment (for random segments, this is about the same as global weight estimates).

cullingRulesDict = {'setConstantDerivEstimatesToZero':True,
                    'extraNumFnsCulledPerIter':extraNumFnsCulledPerIter,
                    'minAllowedWeightedCoeff':minAllowedWeightedCoeff}

#%% Dynamics Eqn params and initial values, by modelSystem:
    
# 'modelParams' is formatted as a tuple (of length 1) to please 'solve_ivp'.
if modelSystem == 'lorenz':
    modelParams = ({'p0': 10, 'p1': 28, 'p2': np.round(-8/3, 2), 'numExtraVars': numExtraVars},)
    # default.  Must be a tuple to pass into odeint as args, so wrap the dict as a length 1 tuple
    #lorenzParams = ({'p0': 5, 'p1': 40, 'p2': np.round(-1, 2), 'numExtraVars': numExtraVars},)
    #lorenzParams = ({'p0': 3, 'p1': 50, 'p2': np.round(-2, 2), 'numExtraVars': numExtraVars},)
    x0TrainAll = [[-8, 8, 27], [5, -7, 29], [-2, 7, 21]]
    x0TestAll =  [[8, 7, 15], [-6, 12, 25]]
    variableNames = ['x','y','z']
    numSecsInTrain = 14    # Length of training time-series. Note: we throw out the first second
    # and last seconds except for use at boundaries

if modelSystem == 'harmOscLinear':
    modelParams = ({'p0': 0.1, 'p1': 2,  'numExtraVars': numExtraVars},)
    x0TrainAll = [[2, 0], [4, 1], [7, 1]]   # see Brunton paper S.I. fig 2
    x0TestAll =  [[3, 2], [6, 3]]
    variableNames = ['x','y']
    numSecsInTrain = 30   # Length of training time-series.

if modelSystem == 'harmOscCubic':
    modelParams = ({'p0': 0.1, 'p1': 2,  'numExtraVars': numExtraVars},)
    x0TrainAll = [[2, 0], [4, 1], [7, 1]]  # see Brunton paper S.I. fig 2
    x0TestAll =  [[3, 2], [6, 3]]
    variableNames = ['x','y']
    numSecsInTrain =  30    # Length of training time-series.

if modelSystem == 'threeDimLinear':
    modelParams = ({'p0': 0.1, 'p1': 2, 'p2': 0.3, 'numExtraVars': numExtraVars},)
    # NOTE: p0 originally = 0.1
    x0TrainAll = [[2, 0, 1], [4, -1, 2], [3, 3, 3]]    # see Brunton paper S.I. fig 3
    x0TestAll =  [[3, 1, 1], [9, 1, 3]]
    variableNames = ['x','y','z']
    numSecsInTrain = 16 # 30    # Length of training time-series.

if modelSystem == 'hopfNormal2D':
    modelParams = ({'p0': 0.2, 'p1': 1, 'p2': -1, 'numExtraVars': numExtraVars},)
    # p0 in [-0.2, 0.6]. CAUTION: these are unsure
    x0TrainAll = [[1, 0.75], [0.9, -0.1], [0.25, 1]]
    x0TestAll =  [[0.1, -0.75], [0.5, -0.5]]
    variableNames = ['x','y']
    numSecsInTrain = 16    # Length of training time-series.

if modelSystem == 'hopfNormal3D':
    modelParams = ({'p0': 0.2, 'p1': 1, 'p2': -1, 'p3': 0.5, 'numExtraVars': numExtraVars},)
    # p0 in [-0.2, 0.6]. CAUTION: these are unsure
    x0TrainAll = [[1, 0.75, 0], [0.9, -0.1, 2], [0.25, 1, 1]] # or z = 0 or z = -160: see brunton
    # paper S.I, eg fig 13
    x0TestAll =  [[0.1, -0.75, 1], [0.5, -0.5, -1]]
    variableNames = ['x','y','z']
    numSecsInTrain = 25   # length in Brunton paper unknown.

""" ------------------------- END USER ENTRIES --------------------------------------------- """

#%%

""" ------------------------------ BEGIN MAIN ---------------------------------------------- """

""" One-time items: """

initialRunTag = '_initialRun_'
if not useFullLibraryFlag:
    initialRunTag = '_secondRun_'
outputFilename = outputFilenamePrefix + '_' + modelSystem + initialRunTag + str(seed)
pickleFilename = pickleFilenamePrefix + '_' + modelSystem + initialRunTag + str(seed) + '.pickle'
 
print('')
print('See ' + outputFilename + ' for detailed progress of run.')
print('Load ' + pickleFilename + ' to get complete run data after completion. \n ')

if useFullLibraryFlag:
    initialFunctionsToUseArray = []

# if results.keys() != fomChangesDict.keys():
#     print('Caution: results keys and fomChangesDict keys must match. Fix and restart.')

if regressOnFftAlpha == 0:
    fftRegressionTarget = 'none'  # just for book-keeping

if numExtraVars > 0:  # Case: there are noise variables. We expect to kill these then start over 
# on the remaining (active) variables, so order the restarts:
    stopIfVarIsCulledFlag = False
    
culledFunctionIndices = []
numCullsWithNoChange = 0
newRemovedFunctionStr = '[]'

#%% check that the padding + window will fit inside the margin:
margin = int(np.round(marginInSecs / dt))    # margin = number of timepoints in the boundary
# regions at start and at end.
if hammingWindowLengthForData/2 > margin:
    print('Error: hammingWindowLength and/or padFactor are too large: ' + \
          str(np.round(hammingWindowLengthForData/2)) + ' > ' + str(np.round(margin)))

numTimepointsForFoM = int((numSecsInTrain - 2*marginInSecs) / dt)

#%% Make names of the variables for printing:
# start with variableNames
for i in range(numExtraVars):
    variableNames.append('w' + str(i)) # w3, w4, etc (dummy vars)
variableNames = np.array(variableNames)

#%% Define train and test trajectories:

# Note re test trajectories: Since these are used as a holdout set, so we do less processing
# than for xTrain. We really just need enough to calculate a model's FoMs on the test set.
# Test trajectories get the following processing: temporarily add noise, solely to calculate an
# envelope for the 'inEnvelope' FoM; collect stats like std dev, for use in FoMs; calculate
# xDotTest for the xDot FoMs. All tTest timepoints are used for FoMs on test trajectories.

tTrain = np.arange(0, numSecsInTrain, dt)
numSecsInTest = numSecsInTrain
tTest = np.arange(0, numSecsInTest, dt)

#%% If wished, append extra distractor variables, with arbitrary initial conditions, to each
# train trajectory:
initCond = []
for j in range(numTrajTrain):
    this = x0TrainAll[j]  # a vector
    for i in range(numExtraVars):
        this.append(np.random.uniform(low = lowMeanVal, high = highMeanVal))
    initCond.append(this)
x0TrainAll = np.array(initCond)    # make into array for ease later

# Ditto for test trajectories:
initCond = []
for j in range(numTrajTest):
    this = x0TestAll[j]
    for i in range(numExtraVars):
        this.append(0)
    initCond.append(this)
x0TestAll = np.array(initCond)

numVars = len(x0TrainAll[0])    # useful param, equals number of variables

#%% Generate the model system time-series for train and for test, as well as a string describing
# the system:
# xTrain and xTest will contain rows for extra variables (if numExtraVars > 0), with constant
# values given by x0Train (or x0Test).
xTrainCleanAll = []

for j in range(numTrajTrain):
    if modelSystem == 'lorenz':
        xTrainCleanAll.append(odeint(lorenz_fn, x0TrainAll[j], tTrain, modelParams))
    if modelSystem == 'harmOscLinear':
        xTrainCleanAll.append(odeint(dampedHarmonicOscillatorLinear_fn, x0TrainAll[j], tTrain,
                                     modelParams))
    if modelSystem == 'harmOscCubic':
        xTrainCleanAll.append(odeint(dampedHarmonicOscillatorCubic_fn, x0TrainAll[j], tTrain,
                                     modelParams))
    if modelSystem == 'threeDimLinear':
        xTrainCleanAll.append(odeint(threeDimLinear_fn, x0TrainAll[j], tTrain, modelParams))
    if modelSystem == 'hopfNormal2D':
        xTrainCleanAll.append(odeint(hopfNormalForm2D_fn, x0TrainAll[j], tTrain, modelParams))
    if modelSystem == 'hopfNormal3D':
        xTrainCleanAll.append(odeint(hopfNormalForm3D_fn, x0TrainAll[j], tTrain, modelParams))

xTestAll = []
for j in range(numTrajTest):
    if modelSystem == 'lorenz':
        xTestAll.append(odeint(lorenz_fn, x0TestAll[j], tTest, modelParams))
    if modelSystem == 'harmOscLinear':
        xTestAll.append(odeint(dampedHarmonicOscillatorLinear_fn, x0TestAll[j], tTest,
                               modelParams))
    if modelSystem == 'harmOscCubic':
        xTestAll.append(odeint(dampedHarmonicOscillatorCubic_fn, x0TestAll[j], tTest, modelParams))
    if modelSystem == 'threeDimLinear':
        xTestAll.append(odeint(threeDimLinear_fn, x0TestAll[j], tTest, modelParams))
    if modelSystem == 'hopfNormal2D':
        xTestAll.append(odeint(hopfNormalForm2D_fn, x0TestAll[j], tTest, modelParams))
    if modelSystem == 'hopfNormal3D':
        xTestAll.append(odeint(hopfNormalForm3D_fn, x0TestAll[j], tTest, modelParams))

#%% Add noise to training trajectories:

# a. Shorten the noiseFactors vector to match number of active variables (it may have been
# defined too long, for convenience):
noiseFactorsActiveVars = noiseFactorsActiveVars[0:len(variableNames)]

# b. Add elements for extra variables:
if numExtraVars > 0:
    noiseFactorsExtraVars = noiseFactorsExtraVars[0:numExtraVars]
    noiseFactors = np.hstack((noiseFactorsActiveVars, noiseFactorsExtraVars))
else:
    noiseFactors = noiseFactorsActiveVars

# c. Add noise to each variable:
xTrainAll = []
for j in range(numTrajTrain):
    if noiseType == 'white':
        xTrainAll.append(addWhiteNoise_fn(xTrainCleanAll[j], noiseFactors))
    else:
        xTrainAll.append(addGaussianNoise_fn(xTrainCleanAll[j], noiseFactors, noiseType))
xTrainNoisyOriginalAll = xTrainAll.copy()   # noisy original, for plotting later

# Test trajectories. Noisy test trajectories are only used to generate envelopes for FoM use:
xTestNoisyAll = []
for j in range(numTrajTest):
    if noiseType == 'white':
        xTestNoisyAll.append(addWhiteNoise_fn(xTestAll[j], noiseFactors))
    else:
        xTestNoisyAll.append(addGaussianNoise_fn(xTestAll[j], noiseFactors, noiseType))

# Calculate how much noise has been added:
noiseStd = np.zeros(numVars)
cleanStd = np.zeros(numVars)
cleanRange = np.zeros(numVars)
noiseLevel = np.zeros(numVars)  # will equal std(noise) / std(cleanTrajectory), per defn in
# Schaeffer 2017.

# Calculate statistics of the added noise, for each training trajectory:
for t in range(numTrajTrain - 1, -1, -1):
    for i in range(numVars):
        addedNoise = xTrainNoisyOriginalAll[t][:,i] - xTrainCleanAll[t][:, i]
        noiseStd[i] = np.std(xTrainNoisyOriginalAll[t][:,i] - xTrainCleanAll[t][:, i])
        cleanStd[i] = np.std(xTrainCleanAll[t][:, i])
        cleanRange[i] = max(xTrainCleanAll[t][:, i]) - min(xTrainCleanAll[t][:, i])
        noiseLevel[i] = int(np.round(100 * noiseStd[i] / cleanStd[i]))
    print('traj # ' + str(t) + ': Clean data std dev = ' + \
          str(np.round(cleanStd, 2)) + '\n' + 'noise envelope std dev = ' + \
          str(np.round(noiseStd, 2)) + '\n' +
          'noise level sigma/sigma = ' + str(noiseLevel))
# Final values are for trajectory 0, since we're iterating backwards. We print them all out to
# check if they vary a lot.
print('')

#%%  For reference only, calculate the true derivatives from the clean data:
xDotTrainTrueAll = []
for j in range(numTrajTrain):
    xDotTrainTrueAll.append(calculateDerivativesFromData_fn(xTrainCleanAll[j], tTrain))
xDotTestTrueAll = []
for j in range(numTrajTest):
    xDotTestTrueAll.append(calculateDerivativesFromData_fn(xTestAll[j], tTrain))  # clean, since
    # xTest is clean

# The following is for plotting only.
xDotTrainOfNoisyDataAll = []
for j in range(numTrajTrain):
    xDotTrainOfNoisyDataAll.append(calculateDerivativesFromData_fn(xTrainAll[j], tTrain))  # from
    # the noisy, unsmoothed data.

#%% Print some facts about the target system:

# Get a string and lists of true functionals for the system:
modelStr, trueLib, trueLibCoeffs = generateModelStrAndTrueArrays_fn(modelSystem, modelParams[0])
# argin 'modelParams' is formatted as a tuple (of length 1) to please 'solve_ivp'.

# Noise level and type:
# sys.stdout is used to send some text to console and some to file.
systemInfoStr = '\n' + modelSystem + ' system (true model): ' + \
          '\n' + 'Seed = ' + str(seed) + '\n' + str(numExtraVars) + ' dummy variables, ' + \
          noiseType + ' noise at ' + str(noiseFactors) + ' * stdDev' + \
          ' \n' + modelStr + '\n' + \
          'Clean data range = ' + str(np.round(cleanRange,2)) + '\n' + \
          'Clean data std dev = ' + str(np.round(cleanStd, 2)) + '\n' + \
          'Noise envelope std dev = ' +  str(np.round(noiseStd, 2)) + '\n' + \
          'Noise level as percentage = ' +  str(noiseLevel.astype(int))
console = sys.stdout
with open(outputFilename, 'a') as file:    
    print(systemInfoStr, file=file)
    sys.stdout = console
    file.close()
    
    
# The system trajectories are now created. 
# Next do various preparations before model fitting:

#%% Generate Hamming filter windows:
# For use on Data; also calculate some windowing values:
hammData = makeHammingWindow_fn(hammingWindowLengthForData)
halfData = int(np.floor(hammingWindowLengthForData) / 2)

# For use on Derivatives:
hammDerivs = makeHammingWindow_fn(hammingWindowLengthForDerivsAndFnals)

# For use on library functions:
hammForFunctions = makeHammingWindow_fn(hammingWindowLengthForDerivsAndFnals)

#%% Calculate global and local max and min for each variable, used to calculate FoMs of the models
# at each iteration:
fomTimepointInds = np.array(range(margin, margin + numTimepointsForFoM))
maxPhaseShift = int(0.1 / dt)  # to reduce FoM error for otherwise-correct evolutions that are out
# of phase.
# For each trajectory, get local measures of max and min, per variable:
varLocalMaxTrainAll = []
varLocalMinTrainAll = []
for j in range(numTrajTrain):
    this = xTrainAll[j]
    varLocalMax, varLocalMin = \
        generateVariableEnvelopes_fn(this[fomTimepointInds, :], windowLengthForFomEnvelopes, 
                                     dt, shrinkFomEnvelopeFactor)
    varLocalMaxTrainAll.append(varLocalMax)
    varLocalMinTrainAll.append(varLocalMin)

varLocalMaxTestAll = []
varLocalMinTestAll = []
for j in range(numTrajTest):
    this = xTestNoisyAll[j]
    varLocalMaxTest, varLocalMinTest = \
        generateVariableEnvelopes_fn(this, windowLengthForFomEnvelopes, dt, 
                                     shrinkFomEnvelopeFactor)
    varLocalMaxTestAll.append(varLocalMax)
    varLocalMinTestAll.append(varLocalMin)

#%% Calculate timepoint weights to use in linear regression.
# Do this for each variable, for each time-point. Procedure: Use the noisy data. For each Point,
# (a) fit a line to the points in a window around the Point; (b) tilt the points so that the
# fitted line has slope 0; (c) define a distribution using the points in the window; (d) the
# weight is 1 / z-score of the Point relative to the distribution.
# Also, within the loop maybe clip or shrink the raw initial data to save flops. This combination
# of actions makes this section a bit messy (and clip and shrink don't work anyway).

print('calculating weights and smoothing.')
pointWeightsAll = []
smoothedXTrainAll = []
eulerSlopesAll = []  # In case we chose to use this as a derivative estimate. Could probably be
# refactored out.
pointWeightsForFomAll = []

for j in range(numTrajTrain):
    thisXTrain = xTrainAll[j].copy()
    thisPointWeights =  np.ones(thisXTrain.shape)
    thisSmoothedXTrain= np.zeros(thisXTrain.shape)
    thisEulerSlopes = np.zeros(thisXTrain.shape)
    for i in range(numVars):  # loop over variables.
        temp = thisXTrain[:, i]
        slopeX, stdX, meanX = \
            calculateSlopeAndStd_fn(temp, dt, hammingWindowLengthForDerivsAndFnals)
        # argouts 2 and 3 are used for weights.

        # A. Update pointWeights array:
        if weightTimepointsFlag:
            weightRow = np.ones(temp.shape)
            weightRow[np.abs(temp - meanX) > 1e-9] = stdX / np.abs(temp - meanX)  # 1 / z-score.
            # Timepoints with super low z-scores remain at maxWeight.
            # Modify to basic weights if wished: 
            if scaleByLogFlag:
                weightRow =  np.log(weightRow + 1)  # base e, could be another base
            # Insert this row into weights array:
            thisPointWeights[:, i] = weightRow

        # B. Update eulerSlopes (for possible future use as derivative estimates):
        thisEulerSlopes[:, i] = slopeX
 

    # Also smooth with spline or hamming if wished:
    # 3. Hamming is done here to keep the logical flow clear: 
    thisXTrain = smoothData_fn(thisXTrain, hammData)

    xTrainAll[j] = thisXTrain  # replace original noisy xTrain with smoothed version.
    pointWeightsAll.append(thisPointWeights)
    smoothedXTrainAll.append(thisXTrain)

    # Define point weights for FoM points, summing to 1:
    thisPointWeightsForFom = thisPointWeights[fomTimepointInds, :].copy()
    pointWeightsForFomAll.append(thisPointWeightsForFom / \
                              np.tile(np.sum(thisPointWeightsForFom, axis=0),
                                      (thisPointWeightsForFom.shape[0], 1)))

# Note that smoothing updates xTrain itself.

#%% Using point weights, select starting points for solve_ivp evolutions: use the highest weight
# points close to 'margin', which is the point at the end of a short opening segment that we
# ignore (to avoid transient effects). The goal is to legally maximize the chance that the
# evolution will start accurately. This is maybe a frivolous detail, and we could just use
# simInitConds = xTrain[margin,:]:
simInitCondsAll = []
for j in range(numTrajTrain):
    thisXTrain = xTrainAll[j]
    thisPointWeights = pointWeightsAll[j]
    if useTrueInitialValueForSimFlag:
        simInitCondsAll.append(xTrainCleanAll[j][margin, :].copy())
    else:
        temp1 = margin * np.ones(thisXTrain[margin, :].shape)
        for i in range(numVars):
            temp2 = thisPointWeights[margin - 2:margin + 3, i]
            ind = np.where(temp2 == max(temp2))[0][0] + margin - 2  # the 2nd [0] in case of a tie.
            temp1[i] = thisXTrain[ind, i]
        simInitCondsAll.append(temp1)

#%% Now that raw data is smoothed, calculate the Derivatives using the pre-processed data, and
# maybe smooth them. Also calculate some stats needed for FoMs on xDot.
print('Smoothing initial derivatives')
xDotTrainAll = []
xDotTrainUnsmoothedAll = []
for j in range(numTrajTrain):
    thisXDot = calculateDerivativesFromData_fn(xTrainAll[j], tTrain)
    thisXDot = smoothData_fn(thisXDot, hammDerivs)
    xDotTrainAll.append(thisXDot)
    xDotTrainUnsmoothedAll.append(xDotTrainAll[j].copy())  # For possible plotting later.

# For FoMs, calculate local max and mins (FFT power spectrum is calculated later):
xDotLocalMaxAll = []
xDotLocalMinAll = []
for j in range(numTrajTrain):
    thisXDotTrain = xDotTrainAll[j]
    temp1, temp2 = generateVariableEnvelopes_fn(thisXDotTrain[fomTimepointInds, :],
                                                windowLengthForFomEnvelopes, dt, 
                                                shrinkFomEnvelopeFactor)
    xDotLocalMaxAll.append(temp1)
    xDotLocalMinAll.append(temp2)

xDotLocalMaxTestAll = []
xDotLocalMinTestAll = []
for j in range(numTrajTest):
    thisXDotTest = xDotTestTrueAll[j]
    temp1, temp2 = generateVariableEnvelopes_fn(thisXDotTest, windowLengthForFomEnvelopes, dt,
                                                shrinkFomEnvelopeFactor)
    xDotLocalMaxTestAll.append(temp1)
    xDotLocalMinTestAll.append(temp2)

#%% Create an initial library of functionals. Also do some prep work with it (4 steps):

# Step 1. Generate the library:
functionList, recipes = generatePolynomialLibrary_fn(variableNames, polynomialLibraryDegree)

# Make the true sparse functional array for the system:
trueSystemLibraryArray, trueSystemCoeffArray = \
    generateTrueFunctionalAndCoeffArrays_fn(trueLib, trueLibCoeffs, functionList)

# Get the index of the non-constant functions for later use:
constantFnIndex = np.where(np.array(functionList) == '1')[0]
temp = np.array(range(len(functionList)))
nonConstantIndices = temp[temp != constantFnIndex]  # To mark non-constant functionals.

# Initialize boolean arrays to use as modelActiveLib, one for each train trajectory. Ditto
# for float arrays for modelCoefs. These will be useful when selecting a group of
# models (one for each training trajectory) and plotting their evolutions of all trajectories.
functionsToUseArrayAll = []

# Use a full all-ones array, or use a custom initial functions array:
if useFullLibraryFlag: # test for emptiness
    initialFunctionsToUseArray = np.ones((len(variableNames), len(functionList)), dtype = bool)
for j in range(numTrajTrain):
    functionsToUseArrayAll.append(initialFunctionsToUseArray)

coeffArrayAll = []
for j in range(numTrajTrain):
    coeffArrayAll.append(np.zeros((len(variableNames), len(functionList))))

#%% Step 2. Calculate values of the feature library, for use during regressions:
LAll = []
LNoisyAll = []
for j in range(numTrajTrain):  # the library functions, calculated at each time point.
    # 'L' is numSamples x numFunctions. Each row is the value of all functions at one timepoint. Each
    # column is the value of one function at all timepoints.
    LAll.append(calculateLibraryFunctionalValues_fn(xTrainAll[j], recipes))
    LNoisyAll.append(calculateLibraryFunctionalValues_fn(xTrainNoisyOriginalAll[j], recipes))

LTestAll = []
for j in range(numTrajTest):
    LTestAll.append(calculateLibraryFunctionalValues_fn(xTestAll[j], recipes))

# Smooth the library functions:
print('smoothing library of functionals...')
for j in range(numTrajTrain):
    LAll[j] = smoothData_fn(LAll[j], hammForFunctions)  # 'L' stands
# for 'library', as in 'the values at each timepoint of all the functionals in the library'.

#%% Step 3. Make weights for coeffs, for use in culling functions. Base this on mean and std of
# xTrain values. The weightVector is big if the functional values can get big, and it's small if
# the functional values tend to stay small.
imputedSizeOfFunctionalsAll = []# 'imputedSizeOfFunctionals' is a positive, mu + sigma estimate of
# functional magnitude. It's used at each regression to estimate a weightVector for the relevant
# functionals.
if coeffWeightsCutoffFactor > 1:
    skip = 5  # To save compute time.
    for j in range(numTrajTrain):
        functionalVals = LAll[j][margin:-margin:skip, :]
        medianFunctionalVals = np.median(functionalVals, axis=0)
        imputedSizeOfFunctionalsAll.append(np.abs(medianFunctionalVals + \
            np.std(functionalVals, axis=0) * np.sign(medianFunctionalVals)))
else:
    imputedSizeOfFunctionalsAll.append(np.ones((1, len(functionList))))
 
# Print functional library:
console = sys.stdout
with open(outputFilename, 'a') as file:
    print('Function library: \n' + str(functionList) + '\n',  file=file)
    sys.stdout = console
    file.close()

# We are done preparing the functional library.

#%% Collect some stats about the post-smoothed time-series, for use with FoMs later:
stdDevTrainAll = []
meanTrainAll = []
medianTrainAll = []
xTrainFftPowerAll = []
xTrainHistogramAll = []
xTrainHistogramBinsAll = []

for j in range(numTrajTrain):
    thisXTrain = xTrainAll[j]
    stdDevTrainAll.append(np.std(thisXTrain[fomTimepointInds, :], axis=0))
    meanTrainAll.append(np.mean(thisXTrain[fomTimepointInds, :], axis=0))
    medianTrainAll.append(np.median(thisXTrain[fomTimepointInds, :], axis=0))
    # FFT power spectrum:
    thisXTrainFftPower = np.zeros((numFftPoints, numVars))
    for i in range(numVars):
        x = thisXTrain[fomTimepointInds, i].copy()
        x = x - np.mean(x)
        xP = pow(np.real(np.fft.fft(x)), 2) # power spectrum of fft
        thisXTrainFftPower[:, i] = xP[0:numFftPoints] / np.sum(xP[0:numFftPoints])  # crop,
        # normalize
    xTrainFftPowerAll.append(thisXTrainFftPower)
    # histograms of time-series:
    numHistogramBins = 100
    thisXTrainHistogram = np.zeros((numHistogramBins, numVars))
    thisXTrainHistogramBins = np.zeros((numHistogramBins, numVars))
    for i in range(numVars):
        temp = np.histogram(thisXTrain[fomTimepointInds, i].flatten(), bins=numHistogramBins)
            # Range of histogram is automatically set to (min, max)
        thisXTrainHistogram[:, i] = temp[0]
        thisXTrainHistogramBins[:, i] = temp[1][0:-1]
    xTrainHistogramAll.append(thisXTrainHistogram)
    xTrainHistogramBinsAll.append(thisXTrainHistogramBins)

# Ditto for xTest:
stdDevTestAll = []
meanTestAll = []
medianTestAll = []
xTestFftPowerAll = []
xTestHistogramAll = []
xTestHistogramBinsAll = []

for j in range(numTrajTest):
    thisXTest = xTestAll[j]
    stdDevTestAll.append(np.std(thisXTest, axis=0))
    meanTestAll.append(np.mean(thisXTest, axis=0))
    medianTestAll.append(np.median(thisXTest, axis=0))
    # FFT power spectrum:
    thisXTestFftPower = np.zeros((numFftPoints, numVars))
    for i in range(numVars):
        x = thisXTest[:, i].copy()
        x = x - np.mean(x)
        xP = pow(np.real(np.fft.fft(x)), 2) # power spectrum of fft
        thisXTestFftPower[:, i] = xP[0:numFftPoints] / np.sum(xP[0:numFftPoints])  # crop,normalize
    xTestFftPowerAll.append(thisXTestFftPower)
    # histograms of time-series:
    numHistogramBins = 100
    thisXTestHistogram = np.zeros((numHistogramBins, numVars))
    thisXTestHistogramBins = np.zeros((numHistogramBins, numVars))
    for i in range(numVars):
        temp = np.histogram(thisXTest[:, i].flatten(), bins=numHistogramBins)  # range is
            # automatically set to (min, max)
        thisXTestHistogram[:, i] = temp[0]
        thisXTestHistogramBins[:, i] = temp[1][0:-1]
    xTestHistogramAll.append(thisXTestHistogram)
    xTestHistogramBinsAll.append(thisXTestHistogramBins)

#%% Calculate calculate FFTs of xDotTrain and of the functional values L, for possible use in
# regressions; This can be the full fft (real + imag), real-only, or power (determined by flag).
# Also calculate FFT power spectrum of xDotTrain (not needed for L) for use in FoMs.

# Training trajectories:
fftXDotTrainTargetAll = []
fftPowerXDotTrainAll = []
fftLibFnsAll = []
xDotTrainHistogramAll = []
xDotTrainHistogramBinsAll = []
for j in range(numTrajTrain):
    thisXDotTrain = xDotTrainAll[j]
    thisL = LAll[j]
    thisFftXDotTrainTarget = np.zeros((numFftPoints, numVars), dtype=complex)  # truncated FFT,
                                                                             # used for regression
    thisFftPowerXDotTrain = np.zeros((numFftPoints, numVars))  # FFT power spectrum, used for FoMs
    for i in range(numVars):
        # 1. Some form of FFT:
        thisX = thisXDotTrain[fomTimepointInds, i].copy()
        fftTarget = calculateFftForRegression_fn(thisX, numFftPoints, fftRegressionTarget)
        thisFftXDotTrainTarget[:, i] = fftTarget
        # 2. FFT power spectrum for FoMs:
        xP = pow(np.real(np.fft.fft(thisX)), 2)
        thisFftPowerXDotTrain[:, i] = xP[0:numFftPoints] / np.sum(xP[0:numFftPoints])

    # Change the data type away from complex if needed:
    if fftRegressionTarget != 'complex':
        thisFftXDotTrainTarget = thisFftXDotTrainTarget.astype(float)
    # Append:
    fftXDotTrainTargetAll.append(thisFftXDotTrainTarget)
    fftPowerXDotTrainAll.append(thisFftPowerXDotTrain)

    # FFT vectors for library functionals:
    thisFftLibFns = np.zeros((numFftPoints, thisL.shape[1]), dtype=complex)  # To regress against
                                                                         # fftXDotTrainTarget.
    for i in range(thisL.shape[1]):
        thisX = thisL[fomTimepointInds, i].copy()
        if functionList[i] == '1':
            fftTarget = np.zeros(numFftPoints)
        else:
            fftTarget = calculateFftForRegression_fn(thisX, numFftPoints, fftRegressionTarget)
        thisFftLibFns[:, i] = fftTarget

    # Change the data type away from complex if needed:
    if fftRegressionTarget != 'complex':
        thisFftLibFns = thisFftLibFns.astype(float)
    # Append:
    fftLibFnsAll.append(thisFftLibFns)

    # Generate histograms:
    thisXDotTrainHistogram = np.zeros((numHistogramBins, numVars))
    thisXDotTrainHistogramBins = np.zeros((numHistogramBins, numVars))
    for i in range(numVars):
        temp = np.histogram(thisXDotTrain[fomTimepointInds, i].flatten(), bins=numHistogramBins)
        thisXDotTrainHistogram[:, i] = temp[0]
        thisXDotTrainHistogramBins[:, i] = temp[1][0:-1]
    xDotTrainHistogramAll.append(thisXDotTrainHistogram)
    xDotTrainHistogramBinsAll.append(thisXDotTrainHistogramBins)

#%% Ditto for xTest:
# NOTE: These variables are currently not referred to again, since FoMs are not currently
# done on xTest, and no regression is done on xTest.
# So maybe this section can be refactored out.

fftXDotTestTargetAll = []
fftPowerXDotTestAll= []
fftLibFnsTestAll= []
xDotTestHistogramAll = []
xDotTestHistogramBinsAll = []

for j in range(numTrajTest):
    thisXDotTest = xDotTestTrueAll[j]
    thisFftXDotTestTarget = np.zeros((numFftPoints, numVars), dtype=complex)  # truncated FFT,
    # used for regression
    thisFftPowerXDotTest = np.zeros((numFftPoints, numVars))  # FFT power spectrum, used for FoMs
    for i in range(numVars):
        # 1. Some form of FFT for regressions:
        thisX = thisXDotTest[:, i].copy()
        fftTarget = calculateFftForRegression_fn(thisX, numFftPoints, fftRegressionTarget)
        thisFftXDotTestTarget[:, i] = fftTarget
        # 2. Power spectrum for FoMs:
        xP = pow(np.real(np.fft.fft(thisX)), 2)
        thisFftPowerXDotTest[:, i] = xP[0:numFftPoints] / np.sum(xP[0:numFftPoints])
    if fftRegressionTarget != 'complex':
        thisFftXDotTestTarget = thisFftXDotTestTarget.astype(float)
    fftXDotTestTargetAll.append(thisFftXDotTestTarget)
    fftPowerXDotTestAll.append(thisFftPowerXDotTest)

    thisLTest = LTestAll[j]
    thisFftLibFnsTest = np.zeros((numFftPoints, thisLTest.shape[1]), dtype=complex)  # To regress
    # against.
    # fftXDotTrainTarget
    for i in range(thisLTest.shape[1]):
        thisX = thisLTest[:, i].copy()
        fftTarget = calculateFftForRegression_fn(thisX, numFftPoints, fftRegressionTarget)
        thisFftLibFnsTest[:, i] = fftTarget
    if fftRegressionTarget != 'complex':
        thisFftLibFnsTest = thisFftLibFns.astype(float)
    fftLibFnsTestAll.append(thisFftLibFnsTest)

    # Generate xDot histograms for FoMs:
    thisXDotTestHistogram = np.zeros((numHistogramBins, numVars))
    thisXDotTestHistogramBins = np.zeros((numHistogramBins, numVars))
    thisXDotTest = xDotTestTrueAll[j]
    for i in range(numVars):
        temp = np.histogram(thisXDotTest[:, i].flatten(), bins=numHistogramBins)  # range is
            # automatically set to (min, max)
        thisXDotTestHistogram[:, i] = temp[0]
        thisXDotTestHistogramBins[:, i] = temp[1][0:-1]
    xDotTestHistogramAll.append(thisXDotTestHistogram)
    xDotTestHistogramBinsAll.append(thisXDotTestHistogramBins)

# The bulk of preparation work is now done.
# Next, start the culling iterations.

#%% Initializations for the fit-and-cull iterations:

# Run algorithm on each training trajectory separately, using the other training trajectories for
# "validation set" FoMs. Ideally this can be parallelized efficiently.
# (During all iterations):
# 1. Update coeffs with linear regression on multiple subsets of timepoints.
# 2a. Cull variables and function types using reliability of coeff estimates.
# 2b. Cull variables and function types using size of coeff estimates (weighted for fn value size).
# Repeat 1-2 as necessary.

# To store results on 'home' training trajectory:
xTrainEvolvedAll = []
xDotTrainPredictedAll = []  # results of a model on its training trajectory
xDotTrainFirstSmoothedAll = []
xDotTrainSmoothedAll = []

# To store results on validation trajectories:
indsValAll = []
xValEvolvedAll = []
xDotValComputedAll = []
xDotValPredictedAll = []
xDotValSmoothedAll = []

# FoM storage. We make a list of lists, where each inner list corresponds to the results for one
# training trajectory. For Val, the contents of each inner list is another list, of the FoMs for
# each val trajectory.
# Build the outer list here. Then the inner lists are populated for each training trajectory.
historyCoeffArrayAll = [[] for i in range(numTrajTrain)]
historyProtectedFnsArrayAll = [[] for i in range(numTrajTrain)]
historyWhichIterAll = [[] for i in range(numTrajTrain)]
historyWhichCullAll = [[] for i in range(numTrajTrain)]
historyInBoundsFoMAll = [[] for i in range(numTrajTrain)]
historyInEnvelopeFoMAll = [[] for i in range(numTrajTrain)]
historyStdDevFoMAll = [[] for i in range(numTrajTrain)]
historyMeanFoMAll = [[] for i in range(numTrajTrain)]
historyMedianFoMAll = [[] for i in range(numTrajTrain)]
historyFftCorrelationFoMAll = [[] for i in range(numTrajTrain)]
historyFftPowerAll = [[] for i in range(numTrajTrain)]
historyHistogramCorrelationFoMAll = [[] for i in range(numTrajTrain)]
historyHistogramsAll = [[] for i in range(numTrajTrain)]
historyHistogramBinsAll = [[] for i in range(numTrajTrain)]
historyMinHistCorrelationForEvolutionsAll = [[] for i in range(numTrajTrain)]
historyXDotDiffEnvelopeAll = [[] for i in range(numTrajTrain)]
historyXDotInBoundsFoMAll = [[] for i in range(numTrajTrain)]
historyXDotHistogramCorrelationFoMAll = [[] for i in range(numTrajTrain)]
historyXDotHistogramsAll = [[] for i in range(numTrajTrain)]
historyXDotHistogramBinsAll = [[] for i in range(numTrajTrain)]

# for xVal:
historyCoeffArrayValAll = [[] for i in range(numTrajTrain)]  # to save coeff matrices of each step.
historyProtectedFnsArrayValAll = [[] for i in range(numTrajTrain)]
historyWhichIterValAll = [[] for i in range(numTrajTrain)]
historyWhichCullValAll = [[] for i in range(numTrajTrain)]
historyInBoundsFoMValAll = [[] for i in range(numTrajTrain)]
historyInEnvelopeFoMValAll = [[] for i in range(numTrajTrain)]
historyStdDevFoMValAll = [[] for i in range(numTrajTrain)]
historyMeanFoMValAll = [[] for i in range(numTrajTrain)]
historyMedianFoMValAll = [[] for i in range(numTrajTrain)]
historyFftCorrelationFoMValAll = [[] for i in range(numTrajTrain)]
historyFftPowerValAll = [[] for i in range(numTrajTrain)]
historyHistogramCorrelationFoMValAll = [[] for i in range(numTrajTrain)]
historyHistogramsValAll = [[] for i in range(numTrajTrain)]
historyHistogramBinsValAll = [[] for i in range(numTrajTrain)]
historyMinHistCorrelationForEvolutionsValAll = [[] for i in range(numTrajTrain)]
historyXDotDiffEnvelopeValAll = [[] for i in range(numTrajTrain)]
historyXDotInBoundsFoMValAll = [[] for i in range(numTrajTrain)]
historyXDotHistogramCorrelationFoMValAll = [[] for i in range(numTrajTrain)]
historyXDotHistogramsValAll = [[] for i in range(numTrajTrain)]
historyXDotHistogramBinsValAll = [[] for i in range(numTrajTrain)]

#%% Loop through training trajectories, applying the training procedure to each trajectory 
# in turn. This could be parallelized, with care taken as to what order results get saved in to 
# the '*All' lists.
for traj in range(numTrajTrain):

    print('\n' + '--------------------------------------------- \n' + \
          'Starting iterations on training trajectory ' + str(traj) + ': \n')

    #%% Initialize a bunch of variables to save results for this trajectory:
    whichIter = 0
    whichCull = 0 # to track iterations of cull-and-rerun.
    iterateAgainFlag = True    # to start the while loop
    functionsToUseArray = functionsToUseArrayAll[traj]
    maxNumCulls = np.sum(functionsToUseArray.flatten())
    if restoreBasedOnFoMsFlag:
        maxNumCulls = int(1.3 * maxNumCulls)  # Allow a few extra.

    xTrain = xTrainAll[traj]
    pointWeights = pointWeightsAll[traj]
    L = LAll[traj]
    imputedSizeOfFunctionals = imputedSizeOfFunctionalsAll[traj]
    fftXDotTrainTarget = fftXDotTrainTargetAll[traj]
    fftLibFns = fftLibFnsAll[traj]
    simInitConds = simInitCondsAll[traj]
    xTrainFftPower = xTrainFftPowerAll[traj]
    varLocalMin = varLocalMinTrainAll[traj]
    varLocalMax = varLocalMaxTrainAll[traj]
    pointWeightsForFom = pointWeightsForFomAll[traj]
    xDotTrain = xDotTrainAll[traj]
    fftPowerXDotTrain = fftPowerXDotTrainAll[traj]
    xDotLocalMin = xDotLocalMinAll[traj]
    xDotLocalMax = xDotLocalMaxAll[traj]

    indsVal = np.array(range(numTrajTrain))
    indsVal = indsVal[indsVal != traj]
    xVal = [xTrainAll[i] for i in indsVal]
    simInitCondsVal = [simInitCondsAll[i] for i in indsVal]
    xValFftPower = [xTrainFftPowerAll[i] for i in indsVal]
    varLocalMinVal = [varLocalMinTrainAll[i] for i in indsVal]
    varLocalMaxVal = [varLocalMaxTrainAll[i] for i in indsVal]
    pointWeightsForFomVal = [pointWeightsForFomAll[i] for i in indsVal]
    xDotVal = [xDotTrainAll[i] for i in indsVal]
    fftPowerXDotVal = [fftPowerXDotTrainAll[i] for i in indsVal]
    xDotLocalMinVal = [xDotLocalMinAll[i] for i in indsVal]
    xDotLocalMaxVal = [xDotLocalMaxAll[i] for i in indsVal]

    # Initialize history arrays for this training trajectory:
    historyCoeffArray = []  # to save coeff matrices from each step.
    historyProtectedFnsArray = []
    historyWhichIter = []
    historyWhichCull = []
    historyInBoundsFoM = []
    historyInEnvelopeFoM = []
    historyStdDevFoM = []
    historyMeanFoM = []
    historyMedianFoM = []
    historyFftCorrelationFoM = []
    historyFftPower = []
    historyHistogramCorrelationFoM = []
    historyHistograms = []
    historyHistogramBins = []
    historyMinHistCorrelationForEvolutions = []
    historyXDotDiffEnvelope = []
    historyXDotInBoundsFoM = []
    historyXDotHistogramCorrelationFoM = []
    historyXDotHistograms = []
    historyXDotHistogramBins = []

    # for xVal:
    historyCoeffArrayVal= []  # to save coeff matrices from each step.
    historyProtectedFnsArrayVal = []
    historyWhichIterVal = []
    historyWhichCullVal = []
    historyInBoundsFoMVal = []
    historyInEnvelopeFoMVal= []
    historyStdDevFoMVal = []
    historyMeanFoMVal = []
    historyMedianFoMVal = []
    historyFftCorrelationFoMVal = []
    historyFftPowerVal = []
    historyHistogramCorrelationFoMVal = []
    historyHistogramsVal = []
    historyHistogramBinsVal = []
    historyMinHistCorrelationForEvolutionsVal = []
    historyXDotDiffEnvelopeVal = []
    historyXDotInBoundsFoMVal = []
    historyXDotHistogramCorrelationFoMVal = []
    historyXDotHistogramsVal = []
    historyXDotHistogramBinsVal = []

    indsValAll.append(indsVal)  # for easy recall

    # Initialize 'results' for this trajectory.
    # The keys must match the keys in 'fomChangesDict'.
    results = {'inBounds':historyInBoundsFoM,
               'evolutionsCorrelation':historyMinHistCorrelationForEvolutions,
               'histogramCorrelation':historyHistogramCorrelationFoM,
               'inEnvelopeFoM':historyInEnvelopeFoM,
               'fftPower':historyFftPower,
               'stdDevFoM':historyStdDevFoM}

    # Check that the dict keys match those specified in User Entries, if we are restoring 
    # functionals:    
    if results.keys() != fomChangesDict.keys() and restoreBasedOnFoMsFlag:
        print('Caution: results keys and fomChangesDict keys must match. Fix and restart.')
        
    # Make a counter. If a variable is culled, we want to start the process over with a restored
    # 'functionsToUseArray' (except for functional that use the culled variable):
    numRemovedVariables = 0

    preCullFunctionsToUseArray = functionsToUseArray  # Initialize
    protectedFunctionsArray = np.zeros(functionsToUseArray.shape, dtype=int)  # initialize
    # Permantently protect any indicated functionals:
    if permanentlyProtectedFunctions.shape[0] > 0:
        protectedFunctionsArray[permanentlyProtectedFunctions] = 1000
    restorableFnsArray = np.zeros(functionsToUseArray.shape, dtype=bool)

    # Initialize the threshold to increment:
    cullableFunctionFlag = False
    restoreFnFlag = False
    skipCullByCoeffSizeDueToInSpanCullFlag = False

    modelActiveLib = functionsToUseArray
    modelCoefs = functionsToUseArray.astype(float)

    # Initializations are done.

    #%% Now start iterations for this trajectory:
    while iterateAgainFlag == True:

        # decrement the counts of protected iterations per {var, fn}:
        protectedFunctionsArray -= 1
        protectedFunctionsArray[protectedFunctionsArray < 0] = 0

        console = sys.stdout
        with open(outputFilename, 'a') as file:
            print('--------------------------------------------', file=file)
            print('Iteration ' + str(whichIter) + ', trajectory ' + str(traj) + ':', file=file)
            sys.stdout = console
            file.close()

        #######################################################

        #%% 1. Regress (each iteration).
        # If 'basic' lin reg, this will determine only the coeffs that go into
        # assessWhetherToCull...(), not the eligible-for-cull fns.
        # If 'ridge' regr, eligible-for-cull functions may change as well. Note that ridge
        # regression does not work as well, so it can perhaps be refactored out.
        # Coefficient estimates are typically much better after regression. These coefficients
        # then get examined by the culling function.
        # To regress, we also need to have culled some new function (2nd condition in 'if' below).

        postRegressionCoeffs = modelCoefs * modelActiveLib
        # These coeffs are from the previous regression (from the last cullAndRerun iteration),
        # with coeffs for any newly culled functions zeroed out. We'll generate new coefficients
        # in this iteration.

        # Make a copy of xTrain, for ease:
        xT = xTrain.copy()  

        #%% 1a. Define some timepoints to regress on, 'startIndsToUse', so-called because they are
        # the first timepoint to use in eg euler or 4th order approximation of the derivative.
        # Perhaps remove any timepoints where the active functionals have wildly big ratios.

        if parseTimepointsByVarValuesFlag:  # Case (but True works best): Ignore timepoints where
        # the difference between library functions is too great ('maxFnValRatio') for too many
        # library functions. Done separately for each variable, since they have different active
        # libraries.
            startIndsToUse = \
                parseTimepointsByMagnitudesOfVariables_fn(L, functionsToUseArray, 
                                                          nonConstantIndices, margin, 
                                                          minNumStartIndsToUse, maxFnValRatio)
            if printDiagnosticsForParsingFlag:
                console = sys.stdout
                with open(outputFilename, 'a') as file:
                    print('\n' + 'numTimepoints for each var = ' + \
                      str(np.sum(startIndsToUse,axis=1).flatten()), file=file)
                    sys.stdout = console
                    file.close()

        else:  # Case: random selection of points, without constraint, to regress at (de facto
        # may be almost all points). Note: This case is not currently used.
            numSamplesToDraw = min(len(fomTimepointInds), len(tTrain) - 2*margin)
            startInds = np.sort(np.random.choice(range(margin, len(tTrain) - margin), 
                                                 numSamplesToDraw, replace = False))
            startIndsToUse = np.zeros((xT.shape[0] - 2*margin, numVars), dtype=bool)
            for v in range(startIndsToUse.shape[1]):
                startIndsToUse[startInds, v] = True

        # 'startIndsToUse' is an array of booleans, showing timepoints to use, one column per
        # variable. The first entry is for timepoint 'margin', and the last is for 
        # {len(tTrain) - margin}.

        # Define a regressor:
        linReg = LinearRegression(fit_intercept=False) # Note that the constant functional is in
        # our library. So we don't want the linear regression to add an intercept term.

        #%% 1b. Regress on each variable using each segment in turn.
        # So do numVars * numRegressionSegments regressions. For each variable, each active
        # functional will end up with numRegressionSegments coefficient estimates.
        # Then compare and combine these sets of coefficients (roughly, take their median).

        # Initialize storage for new coefficients:
        postRegressionCoeffsBySeg = -100 * np.ones((len(variableNames), len(functionList),
                                                    numRegressionSegments))
        
        # 1b(i) Removed in abridged wrapper.
        
        # The outer loop is over variables, the inner loop is over segments.
        for v in range(len(variableNames)):

            pointWtsForThisVar = pointWeights[:, v]

            if np.sum(functionsToUseArray[v, :]) == 0:  # Case: this variable has no usable
            # functionals in its library.
                pass  # postRegressionCoeffs[v, :] remains equal to 0

            else:  # Case: This variable has a non-zero library. Do regressions on each segment:
                startIndsVarI = startIndsToUse[v, ].copy()  # booleans
                # Convert to indices of timepoints:
                startIndsVarI  = np.where(startIndsVarI == True)[0] + margin

                for seg in range(numRegressionSegments):
                    # 1b(ii) Pick a random subset of startIndsVarI for this segment: 
                    numPointsPerSegment = \
                        int(len(startIndsVarI) / numRegressionSegments * \
                            (1 + 2 * overlapFraction))
                    startIndIndices = np.sort(np.random.choice(range(len(startIndsVarI)),
                                              numPointsPerSegment, replace=False))
                    startInds = startIndsVarI[startIndIndices]
                     

                    # Check that we have both valid startInds and active functionals:
                    functionInds = np.where(functionsToUseArray[v, :] == True)[0]
                    if len(startInds) > 0 and len(functionInds) > 0:  # Case: there are some
                    # startInds in this segment and some functionals left for this variable.

                        # 1b(iii) Define the regression target and timepoint weights: 
                        y, weights = \
                            estimateDerivatives_fn(xT[:, v], startInds, numDtStepsForDeriv,
                                                   pointWtsForThisVar, dt)
                        # Extract the relevant functions (columns):
                        X = L[startInds, :].copy()
                        X = X[:, functionInds]
                         
                        if weightTimepointsFlag:
                            sampleWeights = weights
                        else:
                            sampleWeights = np.ones(weights.shape) / len(weights)  # uniform, sum
                            # to 1

                        # Also perhaps include fft(xDotTrain) as targets. There are many
                        # fewer regression points in the fft target: 50 vs. up to 5000 for the
                        # usual raw derivs, so use the sample weights to balance this out.
                        if regressOnFftAlpha > 0:
                            rawYAlpha = 1 - regressOnFftAlpha
                            yFftOfXDot = fftXDotTrainTarget[:, v]
                            y = np.hstack((y, yFftOfXDot ))  # Stack the targets
                            XForFft = fftLibFns[:, functionInds].copy()
                            XForFft[np.where(np.isnan(XForFft))] = 0  # since fft of 1 == nan
                            X = np.vstack((X, XForFft))

                            sampleWeights = \
                                np.hstack((rawYAlpha * sampleWeights, regressOnFftAlpha * \
                                           np.ones(numFftPoints) / numFftPoints))

                        # 1b(iv) Finally ready to do the regression:
                        # Special case: If we are regressing on complex FFT, we need to use lstsq:
                        if fftRegressionTarget == 'complex' and regressOnFftAlpha > 0:
                            w = sampleWeights.reshape(-1, 1)
                            betas = lstsq(np.tile(np.sqrt(w), (1, X.shape[1])) * X,
                                          np.sqrt(w) * y.reshape(-1, 1), rcond=-1)[0]
                            if whichCull > 20 and seg == 0:
                                print('whichIter = ' + str(whichIter) + ' ' + variableNames[v] + \
                                      ', betas = ' + str(np.round(betas, 2).transpose()))
                            betas = np.real(betas)  # NOTE: This may be important and harmful.
                            postRegressionCoeffsBySeg[v, functionInds, seg] = betas.flatten()
                        else: # case: all real values in X and y
                            linReg = linReg.fit(X, y, sample_weight=sampleWeights)
                            postRegressionCoeffsBySeg[v, functionInds, seg] = linReg.coef_

        # The 3-D postRegressionCoeffsBySeg array is now populated, with meaningful values and/or
        # -100s. Regressions are done for all variables and segments.

        #%% 2. Process the collections of coeffs from the different segments to get a single set of
        # coeffs, as follows: For each {variable, library function} pair, if
        # stdDev(coeffs) / median(coeffs) is low, set coeff = median. Else remove the biggest
        # outlier and repeat. This is a fancier version of taking the median.
        # The main work is done in 'combineSegmentCoeff_fn'

        postRegressionStdOverMedian = np.zeros(postRegressionCoeffs.shape)  # Initialize an array
        # to contain noisiness values for each {variable, function} pair. Irrelevant if we do not
        # cull based on noisiness of coefficient estimates.

        prs = postRegressionCoeffsBySeg
        for v in range(len(variableNames)):
            for j in range(len(functionList)):
                if functionsToUseArray[v, j]:  # Case: this functional is in use for this var (it
                # has not been culled), so we need to calculate a single coeff.
                    printDiagnosticsFlag = \
                        np.sum(prs[v, :, 0] != -100) <= printCoeffsForEachSegmentThresh and \
                            v < 4 # ie few enough functionals in this variable, and few enough
                            # variables overall, that we can print the outcome.

                    # Decide what coeff value to carry forward:
                    coeff, stdOverMedian = \
                        combineSegmentCoeffs_fn(prs[v, j, :], printDiagnosticsFlag,
                                                variableNames[v], functionList[j], snrThreshold, 
                                                minNumSegmentResultsToUse, outputFilename)

                    # Assign the final coeffs for this {variable, function} pair:
                    postRegressionCoeffs[v, j] = coeff
                    postRegressionStdOverMedian[v, j] = stdOverMedian # In case we cull functions
                    # based on high variability.

        # postRegressionCoeffs and postRegressionStdOverMedian are now both complete.
 
        #%% Book-keeping section:
        # Now all variables have been fitted (maybe twice if we culled based on std/mean) and
        # have new coeffs assigned. Replace the coeffs of the current model:
        functionsToUseArray = np.abs(postRegressionCoeffs) > almostZeroRoundDownThreshold
        modelActiveLib = functionsToUseArray
        modelCoefs = postRegressionCoeffs * functionsToUseArray  # To avoid weird
        # "almost zero" coefs

        # Print current model:
        this = printModel_fn(modelCoefs, variableNames, functionList)
        console = sys.stdout
        with open(outputFilename, 'a') as file:
            print('\n' + ' Linear Regression (' + \
                  str(numDtStepsForDeriv) + '*dt), target = estimated derivatives ' + \
                      ', on results of whichIter = ' + str(whichIter) + '\n' + this + '\n',
                      file=file)
            sys.stdout = console
            file.close()

        # Diagnostic: If wished, print a version with weighted coeffs:
        if printWeightedCoefficientModelFlag:
            if coeffWeightsCutoffFactor > 1:  # Recall coeffWeightsCutoffFactor = 1 -> no weights.
                weightArray = \
                    calcWeightArray_fn(modelCoefs, modelActiveLib,
                                       imputedSizeOfFunctionals, coeffWeightsCutoffFactor, 
                                       percentileOfImputedValuesForWeights)
            else:  # Case: we're not weighting the coeffs.
                weightArray = modelActiveLib.astype(int)
            printWeightedCoeffModel_fn(modelCoefs, functionsToUseArray, weightArray, 
                                       variableNames, functionList, outputFilename)

        # To catch nans in coeffs:
        modelCoefs[np.where(np.isnan(modelCoefs))] = 0

        #%% 3. Calculate Figures of Merit for this iteration, post-regression (pre-cull):
        # First evolve the data from an estimated initial condition. Then calculate:
        # 1. percentage of time each evolved variable stays within its noise envelope;
        # 2. percentage of time each evolved variable stays within its max-min bounds;
        # 3, 4, 5. std dev, mean, median of each evolved time-series;
        # 6. fft power spectrum correlation; 7. histogram correlation;
        # 8. stability of evolutions, by doing multiple evolutions and comparing histograms);
        # 9. comparing xDot evolutions. We want these generally close but not exact (since the LP
        #    hamming filter gives incorrect derivatives).

        # print('whichIter = ' + str(whichIter) + ', simInitConds = ' + \
        #       str(np.round(simInitConds, 2)))

        # Define the time window we wish to test for matching:
        fomTimepoints = tTrain[fomTimepointInds]

        # 3a. Calculate FoMs for xTrain, UNLESS we culled based on in-span (ie culled using 
        # linear dependence of active functionals), in which case we skip the FoMs (to avoid the
        # cost of time-series evolutions):
        if (skipCullByCoeffSizeDueToInSpanCullFlag and doNotEvolveAfterInSpanCullFlag) or \
            max(np.sum(functionsToUseArray, axis=1)) > maxNumFunctionalsToEvolve:
            print('iter ' + str(whichIter) + \
                  '. Skip FoM evolutions because we culled an in-span functional. \n')
        else:
            print('iter ' + str(whichIter) + '. Evolving models for FoMs...\n')
            # If doing multiple evolutions (to check model stability) there's some extra work:
            if numEvolutionsForTrainFom > 1: # Case: we're doing multiple evolutions of the model
            # to test for model stability. Maybe use parallel processing.
                localHistograms = []
                if doParallelEvolutionsFlag:  # Case: do parallel evolutions
                    preTime = time.time()
                    pool = Pool(processes=numEvolutionsForTrainFom)
                    for i in range(numEvolutionsForTrainFom):
                        # pool.apply_async(foo_pool, args = (i, ), callback = log_result)
                        pool.apply_async(evolveModelAndCalcFoms_fn,
                                         args=(modelCoefs, recipes, simInitConds.copy(), 
                                               fomTimepoints, xTrainFftPower, varLocalMin, 
                                               varLocalMax, maxPhaseShift),
                                          callback=appendToLocalHistograms_fn)

                    pool.close()
                    pool.join()
                    #print(str(localHistograms))

                    postTime = time.time()
                    #print(int(postTime - preTime))

                else: # Case: do not do parallel evolutions
                    numEvolutionsDone = 0
                    preTime = time.time()
                    while numEvolutionsDone < numEvolutionsForTrainFom:
                        # Evolve this model over just the FoM timepoints:
                        xTrainEvolved = evolveModel_fn(modelCoefs, recipes,
                                                       simInitConds.copy(), fomTimepoints)
                        # Calculate the FoMs to get the histogram for this evolution:
                        fomDict = \
                            calculateFiguresOfMerit_fn(xTrainEvolved, xTrain[fomTimepointInds, :],
                                                       xTrainFftPower, varLocalMin, varLocalMax,
                                                       stdDevTrainAll[traj], meanTrainAll[traj],
                                                       medianTrainAll[traj], fomTimepoints,
                                                       maxPhaseShift)
                        localHistograms.append(fomDict['histograms'])
                        numEvolutionsDone += 1

                    postTime = time.time()
                    #print(int(postTime - preTime))

                # Note: The evolution that will be used for most of the FoMs is the last one
                # that occurred (the final version of 'fomDict').
                # If we did multiple evolutions, compare their histograms. Our goal is to see if
                # the evolutions diverged from each other (stability). All the bins (for each 
                # variable) are the same due to 'calculateFiguresOfMerit_fn'.
                minHistCorr = 10*np.ones(numVars)
                this = localHistograms[0]    # numBins x numVars
                for i in range(1, numEvolutionsForTrainFom):
                    that = localHistograms[i]
                    for j in range(numVars):
                        histCorr = np.dot(this[:, j], that[:, j]) / np.dot(this[:, j], this[:, j])
                        minHistCorr[j] = min(minHistCorr[j], histCorr)
            else:  # case: We do just one evolution (so we're not testing for model stability)
                
                minHistCorr = np.ones(numVars)
                xTrainEvolved = evolveModel_fn(modelCoefs, recipes, simInitConds.copy(), 
                                               fomTimepoints)
                fomDict = calculateFiguresOfMerit_fn(xTrainEvolved, xTrain[fomTimepointInds, :],
                                                     xTrainFftPower, varLocalMin, varLocalMax,
                                                     stdDevTrainAll[traj], meanTrainAll[traj],
                                                     medianTrainAll[traj], fomTimepoints, 
                                                     maxPhaseShift)

            # 3b. xDot FoMs: These do not depend on a particular evolution, just on the
            # coefficients of the current model:
            xDotTrainPredicted = \
                calculateDerivativesFromModel_fn(xTrain[fomTimepointInds, :], modelCoefs,
                                                 L[fomTimepointInds, :])
            # (i) Compare this iteration's xDot predictions to the "true" via the 80th %ile of
            # their difference, normalized:
            dummyForXDotFom = np.ones(meanTrainAll[traj].shape) # dummy argin for xDot calcFoM_fn
            maxXDotVals = np.max(np.abs(xDotTrain[fomTimepointInds, :]), axis=0)  # for normalizing
            xDotDiff = xDotTrain[fomTimepointInds, :] - xDotTrainPredicted
            xDotDiffEnvelope = np.zeros(numVars)
            for i in range(numVars):
                xDotDiffEnvelope[i] = np.percentile(np.abs(xDotDiff[:, i]), 80) / maxXDotVals[i]

            # (ii) More xDot FoMs:
            xDotFomDict = \
                calculateFiguresOfMerit_fn(xDotTrainPredicted, xDotTrain[fomTimepointInds, :],
                                           fftPowerXDotTrain, xDotLocalMin, xDotLocalMax,
                                           dummyForXDotFom, dummyForXDotFom, dummyForXDotFom, 
                                           fomTimepoints, 1)

        # 3c. Save the FoMs from this iteration (using FoMs of the last of the evolutions).
        # If we culled an in-span functional, all the FoMs except number of functionals will
        # repeat from the previous time.

        # Special case: If we have been skipping evolutions at high functional counts (to avoid
        # hanging), 'fomDict' may not yet be defined. In this case, define it here:
        if not ('fomDict' in locals()):
            dummyVal = -1 * np.ones((1, numVars))
            fomDict = dict()
            fomDict['inBoundsFoM'] = dummyVal
            fomDict['inEnvelopeFoM'] = dummyVal
            fomDict['stdDevFoM'] = dummyVal
            fomDict['meanFoM'] = dummyVal
            fomDict['medianFoM'] = dummyVal
            fomDict['fftCorrelationFoM'] = dummyVal
            fomDict['fftPower'] = dummyVal
            fomDict['histogramCorrelationFoM'] = dummyVal
            fomDict['histograms'] = dummyVal
            fomDict['histogramBins'] = dummyVal
            minHistCorr = dummyVal
            xDotFomDict = dict()
            xDotFomDict['inBoundsFoM'] = dummyVal
            xDotFomDict['histogramCorrelationFoM'] = dummyVal
            xDotFomDict['histograms'] = dummyVal
            xDotFomDict['histogramBins'] = dummyVal
            xDotDiffEnvelope = dummyVal

        # Now save this iter's foms:
        historyWhichIter.append(whichIter)
        historyWhichCull.append(whichCull)
        historyCoeffArray.append(modelCoefs.copy())  # also gives count of active fnals
        historyInBoundsFoM.append(fomDict['inBoundsFoM'])
        historyInEnvelopeFoM.append(fomDict['inEnvelopeFoM'])
        historyStdDevFoM.append(fomDict['stdDevFoM'])
        historyMeanFoM.append(fomDict['meanFoM'])
        historyMedianFoM.append(fomDict['medianFoM'])
        historyFftCorrelationFoM.append(fomDict['fftCorrelationFoM'])
        historyFftPower.append(fomDict['fftPower'])
        historyHistogramCorrelationFoM.append(fomDict['histogramCorrelationFoM'])
        historyHistograms.append(fomDict['histograms'])
        historyHistogramBins.append(fomDict['histogramBins'])
        historyMinHistCorrelationForEvolutions.append(minHistCorr)
        historyXDotDiffEnvelope.append(xDotDiffEnvelope)
        historyXDotInBoundsFoM.append(xDotFomDict['inBoundsFoM'])
        historyXDotHistogramCorrelationFoM.append(xDotFomDict['histogramCorrelationFoM'])
        historyXDotHistograms.append(xDotFomDict['histograms'])
        historyXDotHistogramBins.append(xDotFomDict['histogramBins'])

        #----------------------------------------------------------
        # 3d. Calculate FoMs for xVal, for each val trajectory in turn. An FoM from each val
        # trajectory get saved in a list, which is then appended to the relevant history List:
        # Note: This section largely repeats the section for FoMs on xTrain, but further
        # refactoring would be a nuisance.
        localInBoundsFoMVal = []  # 'local...' because it is for this iteration only
        localInEnvelopeFoMVal = []
        localStdDevFoMVal = []
        localMeanFoMVal = []
        localMedianFoMVal = []
        localFftCorrelationFoMVal = []
        localFftPowerVal = []
        localHistogramCorrelationFoMVal = []
        localHistogramsVal = []
        localHistogramBinsVal = []
        localMinHistCorrelationForEvolutionsVal = []
        localXDotDiffEnvelopeVal = []
        localXDotInBoundsFoMVal = []
        localXDotHistogramCorrelationFoMVal = []
        localXDotHistogramsVal = []
        localXDotHistogramBinsVal = []

        for k in range(numTrajTrain - 1):
            xVal = xTrainAll[indsVal[k]]  # we will evolve over timepointsForFoMs
            LVal = LAll[indsVal[k]]  # values of the functionals on this trajectory
            initConds = simInitCondsAll[indsVal[k]]
            varLocalMinVal = varLocalMinTrainAll[indsVal[k]]
            varLocalMaxVal = varLocalMaxTrainAll[indsVal[k]]
            xValFftPower = xTrainFftPowerAll[indsVal[k]]
            #xValInitCond = simInitCondsAll[indsVal[k]]  # this starts at 'margin'
            thisStdDev = stdDevTrainAll[indsVal[k]]
            thisMean = meanTrainAll[indsVal[k]]
            thisMedian = medianTrainAll[indsVal[k]]
            xDotVal = xDotTrainAll[indsVal[k]]
            xDotLocalMaxVal = xDotLocalMaxAll[indsVal[k]]
            xDotLocalMinVal = xDotLocalMinAll[indsVal[k]]
            fftPowerXDotVal = fftPowerXDotTrainAll[indsVal[k]]
            dummyForXDotFom = np.ones(thisMean.shape)  # dummy argin for xDot calcFoM_fn

            if skipCullByCoeffSizeDueToInSpanCullFlag and doNotEvolveAfterInSpanCullFlag or \
                max(np.sum(functionsToUseArray, axis=1)) > maxNumFunctionalsToEvolve:
                pass
            else:
                # 3d(i) Evolve val trajectories or trajectory, for FoMs:
                if numEvolutionsForValFom > 1:
                    localHistograms = []
                    if doParallelEvolutionsFlag:
                        pool = Pool(processes=numEvolutionsForValFom)
                        for i in range(numEvolutionsForValFom):
                            pool.apply_async(evolveModelAndCalcFoms_fn,
                                             args=(modelCoefs, recipes,
                                                   initConds.copy(), fomTimepoints, xValFftPower,
                                                   varLocalMinVal, varLocalMaxVal, maxPhaseShift),
                                             callback=appendToLocalHistograms_fn)
                        pool.close()
                        pool.join()
                    else:
                        numEvolutionsDone = 0
                        while numEvolutionsDone < numEvolutionsForValFom:
                            # Evolve this model over just the FoM timepoints:
                            xValEvolved = evolveModel_fn(modelCoefs, recipes, initConds,
                                                         fomTimepoints)
                            xDotValPredicted = \
                                calculateDerivativesFromModel_fn(xVal[fomTimepointInds, :],
                                                                 modelCoefs,
                                                                 LVal[fomTimepointInds, :])

                            # Calculate and print the FoMs:
                            fomDictVal = \
                                calculateFiguresOfMerit_fn(xValEvolved, xVal[fomTimepointInds, :],
                                                           xValFftPower, varLocalMinVal,
                                                           varLocalMaxVal, thisStdDev, thisMean,
                                                           thisMedian, fomTimepoints, 
                                                           maxPhaseShift)
                            localHistograms.append(fomDict['histograms'])
                            numEvolutionsDone += 1

                    # Compare histograms of the different evolutions: All the bins (for each
                    # variable) are the same due to 'calculateFiguresOfMerit_fn'.
                    minHistCorrVal = 10*np.ones(numVars)
                    this = localHistograms[0]    # numBins x numVars
                    for i in range(1, numEvolutionsForValFom):
                        that = localHistograms[i]
                        for j in range(numVars):
                            histCorrVal = np.dot(this[:, j],
                                                 that[:, j]) / np.dot(this[:, j], this[:, j])
                            minHistCorrVal[j] = min(minHistCorrVal[j], histCorrVal)  # the FoM
                else:  # case: We're doing just one evolution (not checking model stability)
                    minHistCorrVal = np.ones(numVars)  # set the stability FoM == 1
                    # Do one evolution of xVal to calculate FoMs:
                    xValEvolved = evolveModel_fn(modelCoefs, recipes, initConds, fomTimepoints)

                # 3d(ii) Calculate and print the FoMs:
                fomDictVal = \
                    calculateFiguresOfMerit_fn(xValEvolved, xVal[fomTimepointInds, :],
                                               xValFftPower, varLocalMinVal, varLocalMaxVal,
                                               thisStdDev, thisMean, thisMedian, fomTimepoints, 
                                               maxPhaseShift)
                # 3d(iii). xDot FoMs on Validation:
                xDotValPredicted = \
                    calculateDerivativesFromModel_fn(xVal[fomTimepointInds, :], modelCoefs,
                                                     LVal[fomTimepointInds, :])
                maxXDotVals = np.max(np.abs(xDotVal), axis=0)  # for normalizing
                xDotDiffVal = xDotVal[fomTimepointInds, :] - xDotValPredicted
                xDotDiffEnvelopeVal = np.zeros(numVars)
                for i in range(numVars):
                    xDotDiffEnvelopeVal[i] = \
                        np.percentile(np.abs(xDotDiffVal[:, i]), 80) / maxXDotVals[i]
                xDotFomDictVal = \
                    calculateFiguresOfMerit_fn(xDotValPredicted, xDotVal[fomTimepointInds, :],
                                               fftPowerXDotVal, xDotLocalMinVal, xDotLocalMaxVal,
                                               dummyForXDotFom, dummyForXDotFom, dummyForXDotFom,
                                               fomTimepoints, 1)
            # 3d(iv). Save FoMs for this val trajectory:
            if not ('fomDictVal' in locals()):
                dummyVal = -1 * np.ones((1, numVars))
                fomDictVal = dict()
                fomDictVal['inBoundsFoM'] = dummyVal
                fomDictVal['inEnvelopeFoM'] = dummyVal
                fomDictVal['stdDevFoM'] = dummyVal
                fomDictVal['meanFoM'] = dummyVal
                fomDictVal['medianFoM'] = dummyVal
                fomDictVal['fftCorrelationFoM'] = dummyVal
                fomDictVal['fftPower'] = dummyVal
                fomDictVal['histogramCorrelationFoM'] = dummyVal
                fomDictVal['histograms'] = dummyVal
                fomDictVal['histogramBins'] = dummyVal
                minHistCorrVal = dummyVal
                xDotFomDictVal = dict()
                xDotFomDictVal['inBoundsFoM'] = dummyVal
                xDotFomDictVal['histogramCorrelationFoM'] = dummyVal
                xDotFomDictVal['histograms'] = dummyVal
                xDotFomDictVal['histogramBins'] = dummyVal
                xDotDiffEnvelopeVal = dummyVal

            localInBoundsFoMVal.append(fomDictVal['inBoundsFoM'])
            localInEnvelopeFoMVal.append(fomDictVal['inEnvelopeFoM'])
            localStdDevFoMVal.append(fomDictVal['stdDevFoM'])
            localMeanFoMVal.append(fomDictVal['meanFoM'])
            localMedianFoMVal.append(fomDictVal['medianFoM'])
            localFftCorrelationFoMVal.append(fomDictVal['fftCorrelationFoM'])
            localFftPowerVal.append(fomDictVal['fftPower'])
            localHistogramCorrelationFoMVal.append(fomDictVal['histogramCorrelationFoM'])
            localHistogramsVal.append(fomDictVal['histograms'])
            localHistogramBinsVal.append(fomDictVal['histogramBins'])
            localMinHistCorrelationForEvolutionsVal.append(minHistCorrVal)
            localXDotDiffEnvelopeVal.append(xDotDiffEnvelopeVal)
            localXDotInBoundsFoMVal.append(xDotFomDictVal['inBoundsFoM'])
            localXDotHistogramCorrelationFoMVal.append(xDotFomDictVal['histogramCorrelationFoM'])
            localXDotHistogramsVal.append(xDotFomDictVal['histograms'])
            localXDotHistogramBinsVal.append(xDotFomDictVal['histogramBins'])

        # 3d(v). All Validation trajectories now have FoMs. Save Val info from this iteration:
        historyWhichIterVal.append(whichIter)
        historyWhichCullVal.append(whichCull)
        historyCoeffArrayVal.append(modelCoefs.copy())
        historyInBoundsFoMVal.append(localInBoundsFoMVal)
        historyInEnvelopeFoMVal.append(localInEnvelopeFoMVal)
        historyStdDevFoMVal.append(localStdDevFoMVal)
        historyMeanFoMVal.append(localMeanFoMVal)
        historyMedianFoMVal.append(localMedianFoMVal)
        historyFftCorrelationFoMVal.append(localFftCorrelationFoMVal)
        historyFftPowerVal.append(localFftPowerVal)
        historyHistogramCorrelationFoMVal.append(localHistogramCorrelationFoMVal)
        historyHistogramsVal.append(localHistogramsVal)
        historyHistogramBinsVal.append(localHistogramBinsVal)
        historyMinHistCorrelationForEvolutionsVal.append(localMinHistCorrelationForEvolutionsVal)
        historyXDotDiffEnvelopeVal.append(localXDotDiffEnvelopeVal)
        historyXDotInBoundsFoMVal.append(localXDotInBoundsFoMVal)
        historyXDotHistogramCorrelationFoMVal.append(localXDotHistogramCorrelationFoMVal)
        historyXDotHistogramsVal.append(localXDotHistogramsVal)
        historyXDotHistogramBinsVal.append(localXDotHistogramBinsVal)

        # Note: we are now done using 'modelForFom' (until next iteration).

        #%% 4. Restore culled functionals:
        # Compare current and past values of selected FoMs. If there is a triggering drop from a
        # sufficiently strong value, and the evolutions all agreed, restore the culled
        # functional and protect it for some future iterations.
        # current dict = {'inBounds':0.8, 'evolutionsCorrelation':0.8, 'histogramCorrelation':0.8}
        restoreFnFlag = False  # initialize
        evolThreshold = fomChangesDict['evolutionsCorrelation'][1]
        if len(historyWhichCull) > 1 and restoreBasedOnFoMsFlag and \
            np.sum(restorableFnsArray.flatten()) > 0 and whichIter > startAfterWhichIter and \
                np.min(historyMinHistCorrelationForEvolutions[-2]) > evolThreshold:
            fomsToMonitor = fomChangesDict.keys()
            # 4a. For each monitored FoM, check whether there was a triggering drop:
            for key in fomsToMonitor:
                this = results[key]
                fomChangeThresh = fomChangesDict[key][0]
                fomMinPreviousVal = fomChangesDict[key][1]
                if key != 'stdDevFoM': # ie most foms
                    restoreFnFlag = (np.min(this[-2]) > fomMinPreviousVal and \
                                     np.min(this[-1] / this[-2]) < fomChangeThresh) or \
                                     restoreFnFlag
                if key == 'stdDevFoM': # use 1 - clipped(abs(std dev value)):
                    temp = np.abs(this[-2])
                    for i in range(len(temp)):
                        temp[i] = min(1, temp[i])
                    adjPreviousVal = 1 - temp
                    temp = np.abs(this[-1])
                    for i in range(len(temp)):
                        temp[i] = min(1, temp[i])
                    adjCurrentVal = 1 - temp
                    restoreFnFlag = \
                        (np.min(adjPreviousVal) > fomMinPreviousVal) and \
                        (np.min(adjPreviousVal - adjCurrentVal) > fomChangeThresh) or \
                        restoreFnFlag  # ie both of first two conditions OR 'restoredFnFlag'
                    # use np.min() because 'this' is a vector, with length = numVars

            #  4b. Restore and protect functionals if indicated:
            if restoreFnFlag:
                restoredFnStr = ''
                for i in np.where(np.sum(restorableFnsArray, axis=1) > 0)[0]:  # ie rows/variables
                # with restored functionals.
                    restoredFnStr = restoredFnStr + variableNames[i] + ': ' + \
                        str(np.array(functionList)[restorableFnsArray[i, :]]) + ' '
                if len(restoredFnStr) > 0:
                    print('iter ' + str(whichIter) + '. Restored functionals: ' + restoredFnStr)
                console = sys.stdout
                with open(outputFilename, 'a') as file:
                    print('Restoring functionals: ' + restoredFnStr, file=file)
                    sys.stdout = console
                    file.close()
                functionsToUseArray = np.logical_or(functionsToUseArray, restorableFnsArray)
                modelActiveLib = functionsToUseArray
                modelCoefs[restorableFnsArray] = 100  # placeholder coefficient.
                protectedFunctionsArray[restorableFnsArray] = numItersProtection + 1  # the '+1'
                # is due to an order-of-events weirdness (that might want attention).

        historyProtectedFnsArray.append(protectedFunctionsArray.copy())

        #%% 5. Culling phase:
        # Assess whether we want to cull and re-run; and if so, what variables and library
        # functions do we keep?

        whichIter += 1
        # If a cull in indicated, prepare for the next iteration of while loop.
        if iterateAgainFlag:
            whichCull = whichCull + 1

        preCullFunctionsToUseArray = functionsToUseArray  # As a record, to check for changes due
        # to cull step.

        # 5a. To keep culling somewhat balanced among variables, set some variables off-limits:
        librarySize = np.sum(functionsToUseArray, axis=1)
        imbalance = np.max(librarySize) - librarySize >= balancedCullNumber  # the index of the
        # variable with the biggest library
        inBalanceVarsArray = np.ones(functionsToUseArray.shape, dtype=bool)
        inBalanceVarsArray[imbalance, :] = False  # False means this var has too few functionals,
        # so it is off-limits for culling. Entire rows are uniformly True or False.

        #%% 5b. In-span culling:
        # Before the usual cull, see if any functionals are in the span of the other active
        # functionals (leave-one-out test of linear dependence). If yes, remove one of them then
        # skip the usual culling step.
        skipCullByCoeffSizeFlag = False  # default
        inSpanCullStr = ''
        skipCullByCoeffSizeDueToInSpanCullFlag = False  # reset for this iteration
        # A dict of params:
        params = dict()
        params['functionList'] = functionList 
        params['variableNames'] = variableNames  
        params['outputFilename'] = outputFilename  
        params['plottingThreshold'] = 2  # > 1 disables plotsof linearly dependent functional fits  
        params['windowLengthForFomEnvelopes'] = windowLengthForFomEnvelopes 
        params['dt'] = dt  
        params['margin'] = margin  
        params['maxFnValRatio'] =  maxFnValRatio 
        params['minNumStartIndsToUse'] = minNumStartIndsToUse
        if cullUsingInSpanFlag and not restoreFnFlag:  # if a functional was restored, skip culling
            # 5b(i). Leave-one-out linear dependence.
            includeConstantFlag = False
            LNoisy = LNoisyAll[traj]
            rSqValsLoo = \
                findSpansOfFunctionalsLeaveOneOut_fn(functionsToUseArray,
                                                     L[fomTimepointInds, :].copy(),
                                                     LNoisy[fomTimepointInds, :].copy(), 
                                                     pointWeightsForFom, includeConstantFlag,
                                                     params)[0]  # Use 1st argout only
                # Note: to avoid plotting candidates, set argin 6 > 1, eg 1.1

            # 5b(ii). Optionally print R-squared values:
            if printDiagnosticOfInSpanCullFlag and max(rSqValsLoo.flatten()) > 0.9:
                console = sys.stdout
                with open(outputFilename, 'a') as file:
                    print('Rsq leave-one-out values: \n' + str(np.round(rSqValsLoo,3)), file=file)
                    sys.stdout = console
                    file.close()

            # 5b(iii). Create a weighted coefficient array:
            weightArray = \
                calcWeightArray_fn(postRegressionCoeffs, functionsToUseArray,
                                   imputedSizeOfFunctionals, coeffWeightsCutoffFactor, 
                                   percentileOfImputedValuesForWeights)
            wtedCoeffs = weightArray * modelCoefs

            # 5b(iv). If in-span candidates exist, cull some based on smallest weighted coef. Each
            # variable is treated separately.
            for i in range(numVars):
                candidateInds = np.where(np.logical_and(rSqValsLoo[i, :] > inSpanCullThreshold,
                                                        rSqValsLoo[i, :] < 1))[0]
                # The '< 1' exempts the constant functional if we set LinearRegression to
                # automatically include a bias term (if not, then the '< 1' has no effect).
                # Two conditions are needed to proceed:
                if len(candidateInds) > 0 and inBalanceVarsArray[i, 0]:
                    # Calculate a max threshold. Don't cull a functional with high weighted coeff:
                    wtedCoeffsThisVar = wtedCoeffs[i, :]
                    wtedCoeffsThisVar = wtedCoeffsThisVar[np.abs(wtedCoeffsThisVar) > \
                                                          almostZeroRoundDownThreshold]
                    maxAllowedWt = np.percentile(np.abs(wtedCoeffsThisVar),
                                                 percentileThresholdForInSpanCulling)
                    # Maybe cull the one with the lowest weighted coefficient:
                    candidateWts = np.abs(wtedCoeffs[i, candidateInds])   # shorter vector
                    cullInd = candidateInds[candidateWts == min(candidateWts)][0]  # The [0] is
                    # due to the RHS being an array otherwise.
                    if np.abs(wtedCoeffs[i, cullInd]) < maxAllowedWt: # Cull if wted coeff is small
                        inSpanCullStr = inSpanCullStr + variableNames[i] + ': ' + \
                            functionList[cullInd] + '. '
                        functionsToUseArray[i, cullInd] = False
                        skipCullByCoeffSizeDueToInSpanCullFlag = True

        modelActiveLib = functionsToUseArray
        modelCoefs = modelCoefs * functionsToUseArray

        # We're done culling in-span functionals. Continue with 'usual' cull based on low coeffs.

        #%% 5c. Culling based on lowest coefficient (standard type of sequential threshold cull).
        # Cull library functions with low weighted coeffs. This generates a modelActiveLib for the
        # next run. The main work is done in 'cullAndAssessWhetherToRerun_fn'.
        if whichCull == 0:  # edge case
            liveVarInds = np.sum(functionsToUseArray,axis=1) > 0
            functionsToUseArray[liveVarInds, :] = True

        if restoreFnFlag:  # Case: Fill in some book-keeping values, and skip the cull
            iterateAgainFlag = True
            outputStr = 'Just restored a functional, so skip coeff-based cull this iteration.'
            minNonZeroWeightedCoeff = -1
            cullableFunctionFlag = False
            restorableFnsArray = np.zeros(functionsToUseArray.shape,dtype=bool)
        else:  # Case: Do a cull. First assign some book-keeping values
            if skipCullByCoeffSizeDueToInSpanCullFlag:  # Case: we already culled in-span fnals
                minNonZeroWeightedCoeff = -1
                cullableFunctionFlag = False
                outputStr ='Removed in-span functional(s) with Rsq > ' + str(inSpanCullThreshold) + \
                    ', ' + inSpanCullStr + \
                           ' Skip coeff-based cull this iteration.'
            else:  # Case: we actually do this cull step
                cullable = protectedFunctionsArray == 0 # any {var, fn} with > 0 entry is protected

                iterateAgainFlag, functionsToUseArray, restorableFnsArray, outputStr, \
                    minNonZeroWeightedCoeff, cullableFunctionFlag = \
                        cullAndAssessWhetherToRerun_fn(modelCoefs.copy(), variableNames,
                                                       functionList, imputedSizeOfFunctionals,
                                                       cullingRulesDict, functionsToUseArray,
                                                       cullable, inBalanceVarsArray, 
                                                       coeffWeightsCutoffFactor,
                                                       percentileOfImputedValuesForWeights)

                # Update the model's boolean active library array, and the coefficient array:
                modelActiveLib = functionsToUseArray
                modelCoefs = modelCoefs * functionsToUseArray
                       
            print('iter ' + str(whichIter) + '. ' + outputStr)
        # To track culls of variables (all functionals except maybe constant removed) and entire
        # functionals (ie the functional culled for all variables):
        removedVariableInds = np.where(np.sum(functionsToUseArray, axis = 1) == 0)[0]  # indices
        # of vars that have all zeros in their function libraries

        #%% 5d. Handling removed variables:
        # If all functionals of a variable have just been culled, the variable is now presumed
        # constant, so all functionals for which that variable is an argin must be culled. This
        # changes the picture such that we don't trust our cull history, so we start over.
        # So if we have just removed a variable, we want to start over with functionsToUseArray
        # restored such that rows == True for all surviving variables, except for functionals using
        # the removed variable as an argin, which are all set to False.
        # If we would stop anyway (ie stopIfVarIsCulledFlag == True), then don't run this section.
        if len(removedVariableInds) > numRemovedVariables and not stopIfVarIsCulledFlag:
            numCullsWithNoChange = 0
            restoredFnArray = np.ones(functionsToUseArray.shape, dtype=bool)
            restoredFnArray[removedVariableInds, :] = False
            # 2. Now zero out all functions that involve zeroed-out variables, ie zero out columns
            # that contain the variable name:
            for i in removedVariableInds:
                varName = variableNames[i]
                for j in range(restoredFnArray.shape[1]):
                    if varName in functionList[j]: # exception for the constant function
                        restoredFnArray[:, j] = False
            functionsToUseArray = restoredFnArray
            modelCoefs = functionsToUseArray.astype(float)
            print('Note: Removed a new variable, re-starting the cull steps.')
            console = sys.stdout
            with open(outputFilename, 'a') as file:
                print('Note: Removed a new variable, re-starting the cull steps.', file=file)
                sys.stdout = console
                file.close()
            whichCull = -1  # It gets incremented before leaving the 'cull' block.

        # 5e. Assorted book-keeping and printouts:
        numRemovedVariables = len(removedVariableInds)
        removedVariableStr =  str(variableNames[removedVariableInds])  # Each printed message
        # lists all variables removed to date.
        # Track functionals that have been removed in all variables, for fun only:
        removedFunctionInds = np.where(np.sum(functionsToUseArray, axis = 0) == 0)[0]
        # See which of these are new:
        newRemovedFnInds = removedFunctionInds.copy()
        for i in range(len(newRemovedFnInds)):
            if removedFunctionInds[i] in culledFunctionIndices:
                newRemovedFnInds[i] = -1
        newRemovedFnInds = newRemovedFnInds[np.where(newRemovedFnInds > -1)[0]]
        newRemovedFunctionStr = str(np.array(functionList)[newRemovedFnInds])

        # Print to console (Optional diagnostic):
        removedVariableStr = removedVariableStr.replace('[','')
        removedVariableStr = removedVariableStr.replace(']','')
        if len(removedVariableStr) > 0:
            variablesRemovedPrintStr = 'Removed variables: ' +  removedVariableStr + '. '
        else:
            variablesRemovedPrintStr = ''
        newRemovedFunctionStr = newRemovedFunctionStr.replace('[','')
        newRemovedFunctionStr = newRemovedFunctionStr.replace(']','')
        if len(newRemovedFunctionStr) > 0:
            functionsRemovedPrintStr = ' Newly-removed functional column(s): ' + \
                newRemovedFunctionStr + '. '
        else:
            functionsRemovedPrintStr = ''

        # Print results of culling:
        console = sys.stdout
        with open(outputFilename, 'a') as file:
            print('\n' + 'Cull ' + str(whichCull) + ' results: ' + \
              variablesRemovedPrintStr + outputStr, file=file)  # + functionsRemovedPrintStr)
            sys.stdout = console
            file.close()

        # update culledFunctionIndices:
        culledFunctionIndices = removedFunctionInds.copy()

        # Culls are now done and changes are recorded.

        # 5f. Update some flags, to decide whether to iterate again:
        if iterateAgainFlag == False:
            numCullsWithNoChange += 1
        else:
            numCullsWithNoChange = 0   # if this cull had some effect, reset the counter.

        # Modify iterateAgainFlag according to other constraints:
        if whichCull > maxNumCulls:
            iterateAgainFlag = False
        if np.sum(functionsToUseArray.flatten()) == 0:
            iterateAgainFlag = False

        # If weighted coeffs are stable and above the max cutoff threshold, and we have not just
        # restored a functional, stop the iterations:
        noCoeffsCutFlag = len(newRemovedFunctionStr) == 0 and ('Also culled' not in outputStr)

        # Maybe stop the run due to a variable being wiped out:
        if stopIfVarIsCulledFlag and len(removedVariableInds) > 0:
            iterateAgainFlag = False
    # End of while iterateAgainFlag loop (started around line 2340). When we exit this loop, we are
    # done with the regress-cull iterations.

    #%% 6. Now save this trajectory's history lists:
    historyCoeffArrayAll[traj] = historyCoeffArray
    historyProtectedFnsArrayAll[traj] = historyProtectedFnsArray
    historyWhichIterAll[traj] = historyWhichIter
    historyWhichCullAll[traj] = historyWhichCull
    historyInBoundsFoMAll[traj] = historyInBoundsFoM
    historyInEnvelopeFoMAll[traj] = historyInEnvelopeFoM
    historyStdDevFoMAll[traj] = historyStdDevFoM
    historyMeanFoMAll[traj] = historyMeanFoM
    historyMedianFoMAll[traj] = historyMedianFoM
    historyFftCorrelationFoMAll[traj] = historyFftCorrelationFoM
    historyFftPowerAll[traj] = historyFftPower
    historyHistogramCorrelationFoMAll[traj] = historyHistogramCorrelationFoM
    historyHistogramsAll[traj] = historyHistograms
    historyHistogramBinsAll[traj] = historyHistogramBins
    historyMinHistCorrelationForEvolutionsAll[traj] = historyMinHistCorrelationForEvolutions
    historyXDotDiffEnvelopeAll[traj] = historyXDotDiffEnvelope
    historyXDotInBoundsFoMAll[traj] = historyXDotInBoundsFoM
    historyXDotHistogramCorrelationFoMAll[traj] = historyXDotHistogramCorrelationFoM
    historyXDotHistogramsAll[traj] = historyXDotHistograms
    historyXDotHistogramBinsAll[traj] = historyXDotHistogramBins

    # for xVal:
    historyCoeffArrayValAll[traj] = historyCoeffArrayVal  # to save coeff matrices of each step.
    historyProtectedFnsArrayValAll[traj] = historyProtectedFnsArrayVal
    historyWhichIterValAll[traj] = historyWhichIterVal
    historyWhichCullValAll[traj] = historyWhichCullVal
    historyInBoundsFoMValAll[traj] = historyInBoundsFoMVal
    historyInEnvelopeFoMValAll[traj] = historyInEnvelopeFoMVal
    historyStdDevFoMValAll[traj] = historyStdDevFoMVal
    historyMeanFoMValAll[traj] = historyMeanFoMVal
    historyMedianFoMValAll[traj] = historyMedianFoMVal
    historyFftCorrelationFoMValAll[traj] = historyFftCorrelationFoMVal
    historyFftPowerValAll[traj] = historyFftPowerVal
    historyHistogramCorrelationFoMValAll[traj]= historyHistogramCorrelationFoMVal
    historyHistogramsValAll[traj] = historyHistogramsVal
    historyHistogramBinsValAll[traj] = historyHistogramBinsVal
    historyMinHistCorrelationForEvolutionsValAll[traj] = historyMinHistCorrelationForEvolutionsVal
    historyXDotDiffEnvelopeValAll[traj] = historyXDotDiffEnvelopeVal
    historyXDotInBoundsFoMValAll[traj] = historyXDotInBoundsFoMVal
    historyXDotHistogramCorrelationFoMValAll[traj] = historyXDotHistogramCorrelationFoMVal
    historyXDotHistogramsValAll[traj] = historyXDotHistogramsVal
    historyXDotHistogramBinsValAll[traj] = historyXDotHistogramBinsVal

    # End of loop "for traj in range(numTrajTrain):"
    
#%% 7. Save the results, for use by 'plotSelectedIterations.py':
# Save two things: original system information, and histories of iteration results.
 
results = dict()

# Original system trajectories and associated parameters:
results['systemInfoStr'] = systemInfoStr
results['numTrajTrain'] = numTrajTrain
results['variableNames'] = variableNames
results['functionList'] = functionList
results['trueSystemCoeffArray'] = trueSystemCoeffArray
results['trueSystemLibraryArray'] = trueSystemLibraryArray 
results['dt'] = dt
results['marginInSecs'] = marginInSecs
results['numSecsInTrain'] = numSecsInTrain
results['numSecsInTest'] = numSecsInTest
results['numTrajTest'] = numTrajTest
results['tTrain'] = tTrain
results['xTrainNoisyOriginalAll'] = xTrainNoisyOriginalAll
results['xTrainCleanAll'] = xTrainCleanAll
results['xTrainAll'] = xTrainAll
results['xDotTrainUnsmoothedAll'] = xDotTrainUnsmoothedAll
results['xDotTrainAll'] = xDotTrainAll
results['xDotTrainTrueAll'] = xDotTrainTrueAll
results['indsValAll'] = indsValAll
results['tTest'] = tTest
results['xTestAll'] = xTestAll
results['xDotTestTrueAll'] = xDotTestTrueAll

results['xTrainFftPowerAll'] = xTrainFftPowerAll
results['xTrainHistogramAll'] = xTrainHistogramAll
results['xTrainHistogramBinsAll'] = xTrainHistogramBinsAll
results['xDotTrainHistogramAll'] = xDotTrainHistogramAll
results['xDotTrainHistogramBinsAll'] = xDotTrainHistogramBinsAll

results['simInitCondsAll'] = simInitCondsAll
results['LAll'] = LAll
results['recipes'] = recipes
results['fomTimepointInds'] = fomTimepointInds
results['derivAndFnalSmoothType'] = 'hamming'
results['hammDerivs'] = hammDerivs
results['x0TestAll'] = x0TestAll
results['LTestAll'] = LTestAll
results['trueLib'] = trueLib
results['pointWeightsForFomAll'] = pointWeightsForFomAll
results['LNoisyAll'] = LNoisyAll

results['outputFilename'] = outputFilename
results['windowLengthForFomEnvelopes'] = windowLengthForFomEnvelopes 
results['maxFnValRatio'] = maxFnValRatio
results['minNumStartIndsToUse'] = minNumStartIndsToUse

# Histories of all iterations in this run:
results['historyCoeffArrayAll'] =  historyCoeffArrayAll
results['historyProtectedFnsArrayAll'] = historyProtectedFnsArrayAll
results['historyWhichIterAll'] =  historyWhichIterAll   
results['historyWhichCullAll'] =  historyWhichCullAll
results['historyInBoundsFoMAll'] = historyInBoundsFoMAll 
results['historyInEnvelopeFoMAll'] = historyInEnvelopeFoMAll 
results['historyStdDevFoMAll'] = historyStdDevFoMAll 
results['historyMeanFoMAll'] = historyMeanFoMAll 
results['historyMedianFoMAll'] =  historyMedianFoMAll
results['historyFftCorrelationFoMAll'] = historyFftCorrelationFoMAll 
results['historyFftPowerAll'] = historyFftPowerAll 
results['historyHistogramCorrelationFoMAll'] = historyHistogramCorrelationFoMAll  
results['historyHistogramsAll'] =  historyHistogramsAll
results['historyHistogramBinsAll'] = historyHistogramBinsAll 
results['historyMinHistCorrelationForEvolutionsAll'] = \
    historyMinHistCorrelationForEvolutionsAll
results['historyXDotInBoundsFoMAll'] = historyXDotInBoundsFoMAll 
results['historyXDotDiffEnvelopeAll'] =  historyXDotDiffEnvelopeAll
results['historyXDotHistogramCorrelationFoMAll'] = historyXDotHistogramCorrelationFoMAll
results['historyXDotHistogramsAll'] = historyXDotHistogramsAll 
results['historyXDotHistogramBinsAll'] = historyXDotHistogramBinsAll 

# for xVal: 
results['historyCoeffArrayValAll'] = historyCoeffArrayValAll 
results['historyProtectedFnsArrayValAll'] =  historyProtectedFnsArrayValAll
results['historyWhichIterValAll'] =  historyWhichIterValAll
results['historyWhichCullValAll'] = historyWhichCullValAll 
results['historyInBoundsFoMValAll'] =  historyInBoundsFoMValAll 
results['historyInEnvelopeFoMValAll'] = historyInEnvelopeFoMValAll 
results['historyStdDevFoMValAll'] =  historyStdDevFoMValAll
results['historyMeanFoMValAll'] = historyMeanFoMValAll 
results['historyMedianFoMValAll'] = historyMedianFoMValAll 
results['historyFftCorrelationFoMValAll'] = historyFftCorrelationFoMValAll 
results['historyFftPowerValAll'] = historyFftPowerValAll 
results['historyHistogramCorrelationFoMValAll'] = historyHistogramCorrelationFoMValAll
results['historyHistogramsValAll'] = historyHistogramsValAll
results['historyHistogramBinsValAll'] =  historyHistogramBinsValAll
results['historyMinHistCorrelationForEvolutionsValAll'] = \
    historyMinHistCorrelationForEvolutionsValAll 
results['historyXDotDiffEnvelopeValAll'] = historyXDotDiffEnvelopeValAll
results['historyXDotInBoundsFoMValAll'] = historyXDotInBoundsFoMValAll
results['historyXDotHistogramCorrelationFoMValAll'] = \
    historyXDotHistogramCorrelationFoMValAll 
results['historyXDotHistogramsValAll'] = historyXDotHistogramsValAll
results['historyXDotHistogramBinsValAll'] = historyXDotHistogramBinsValAll 

# Save dict in a pickle file:
with open(pickleFilename, 'wb') as f:
   pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
 
    
'''  All fit-and-cull iterations are done. Now plot FoMs and maybe time-series: '''
 
# Reminder of where data was saved: 
print('')
print('See ' + outputFilename + ' for detailed progress of run.')
print('Load ' + pickleFilename + ' to get complete run data after completion. \n ') 
   
#%% Remind ourselves of the target system.  
console = sys.stdout
with open(outputFilename, 'a') as file: 
    print(systemInfoStr, file=file)
    sys.stdout = console
    file.close()

sys.stdout.flush() 

#%% 8. Plot figures of merit mosaics:
# These are an important tool to assess the accuracy of models as they become progressively
# more sparse.
# Format: For each train trajectory, plot one mosaic 5 x 4: first two cols = home trajectory
# results, last 2 cols = results on each val trajectory, using the home trajectory model.

if showFiguresOfMeritMosaicsFlag:

    # Pack d, the dataDict which is argin for the FoM mosaic function:
    d = dict()
    d['seed'] = seed
    d['numTrajTrain'] = numTrajTrain
    d['variableNames'] = variableNames
    d['functionList'] = functionList

    d['indsValAll'] = indsValAll
    d['historyWhichIterAll'] = historyWhichIterAll
    d['historyCoeffArrayAll'] = historyCoeffArrayAll
    d['historyMinHistCorrelationForEvolutionsAll'] = historyMinHistCorrelationForEvolutionsAll
    d['historyInBoundsFoMAll'] = historyInBoundsFoMAll
    d['historyInEnvelopeFoMAll'] = historyInEnvelopeFoMAll
    d['historyStdDevFoMAll'] = historyStdDevFoMAll
    d['historyHistogramCorrelationFoMAll'] = historyHistogramCorrelationFoMAll
    d['historyXDotDiffEnvelopeAll'] = historyXDotDiffEnvelopeAll
    d['historyMinHistCorrelationForEvolutionsValAll'] = \
        historyMinHistCorrelationForEvolutionsValAll
    d['historyInBoundsFoMValAll'] = historyInBoundsFoMValAll
    d['historyInEnvelopeFoMValAll'] = historyInEnvelopeFoMValAll
    d['historyFftCorrelationFoMAll'] = historyFftCorrelationFoMAll
    d['historyXDotHistogramCorrelationFoMAll'] = historyXDotHistogramCorrelationFoMAll
    d['historyStdDevFoMValAll'] = historyStdDevFoMValAll
    d['historyHistogramCorrelationFoMValAll'] = historyHistogramCorrelationFoMValAll
    d['historyFftCorrelationFoMValAll'] = historyFftCorrelationFoMValAll
    d['historyXDotHistogramCorrelationFoMValAll'] = historyXDotHistogramCorrelationFoMValAll
    d['historyXDotDiffEnvelopeValAll'] = historyXDotDiffEnvelopeValAll

    # Call the mosaic function:
    plotFiguresOfMeritMosaics_fn(d)

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


