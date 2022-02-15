#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 
Support functions for the sindy toolkit.
Called by 'runToolkit.py', 'runToolkitExtended.py' and by 'plotSelectedIterations.py'.

For the full procedure, see "README.md".

For method details, please see "A toolkit for data-driven discovery of governing equations in 
high-noise regimes" (2022) by C.B. Delahunt and J.N. Kutz.
         
Copyright (c) 2021 Charles B. Delahunt.  delahunt@uw.edu
MIT License
"""


import sys
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.linear_model import LinearRegression 
from numpy.linalg import lstsq  # since this accepts complex inputs and targets. 
from scipy.integrate import solve_ivp  

"""
---------------------------------------------------------------------------------------------
------------------------ Function Defs ------------------------------------------------------
--------------------------------------------------------------------------------------------- """

#%% Functions for various model systems, that return initial conditions for derivatives. 
# These are used to simulate the system via odeint():
 
def lorenz_fn(z, t, p=({'p0': 10, 'p1': 28, 'p2': np.round(-8/3, 2), 'numExtraVars': 0},)):
    """ xDot = -p0*x + p0*y
        yDot =  p1*x - y - x*z
        zDot =  p2*z + x*y  
        Inputs: 
            z: np.vector of floats (initial conditions)
            t: np.vector of floats (timesteps)
            p: dict (system parameters)
        Output:
            derivList: np.vector of floats (initial conditions of derivatives).
            """  
    derivList = [
        p['p0'] * (z[1] - z[0]),
        z[0] * (p['p1'] - z[2]) - z[1],
        z[0] * z[1] + p['p2'] * z[2]
] 
    for i in range(p['numExtraVars']):
        derivList.append(0) 
    
    return derivList 

# End of lorenz attractor fn 
    
# -------------------------------------------------------------------    
def dampedHarmonicOscillatorLinear_fn(z, t, p=({'p0': 0.1, 'p1': 2, 'numExtraVars': 0})):
    """ xDot = -p0*x + p1*y
        yDot = -p1*x - p0*y  
        Inputs: 
            z: np.vector of floats (initial conditions)
            t: np.vector of floats (timesteps)
            p: dict (system parameters)
        Output:
            derivList: np.vector of floats (initial conditions of derivatives).
    """  
    derivList = [
        -p['p0']*z[0] + p['p1']*z[1],
        -p['p1']*z[0] - p['p0']*z[1]
]
    for i in range(p['numExtraVars']):
        derivList.append(0) 
    
    return derivList 

# end of dampedHarmonicOscillatorLinear_fn
# -------------------------------------------------------------------
    
def dampedHarmonicOscillatorCubic_fn(z, t, p=({'p0': 0.1, 'p1': 2, 'numExtraVars': 0})):
    """ xDot = -p0*x^3 + p1*y^3
        yDot = -p1*x^3 - p0*y^3  
        Inputs: 
            z: np.vector of floats (initial conditions)
            t: np.vector of floats (timesteps)
            p: dict (system parameters)
        Output:
            derivList: np.vector of floats (initial conditions of derivatives).
    """  
    derivList = [
        -p['p0']*pow(z[0], 3) + p['p1']*pow(z[1], 3), 
        -p['p1']*pow(z[0], 3) - p['p0']*pow(z[1], 3)
]
    for i in range(p['numExtraVars']):
        derivList.append(0) 
    
    return derivList 

# end of dampedHarmonicOscillatorCubic_fn
#---------------------------------------------------------------------- 
        
def threeDimLinear_fn(z, t, p=({'p0': 0.1, 'p1': 2, 'p2': 0.3, 'numExtraVars': 0})):
    """ xDot = -p0*x - p1*y
        yDot =  p1*x - p0*y 
        zDot = -p2*z
        Inputs: 
            z: np.vector of floats (initial conditions)
            t: np.vector of floats (timesteps)
            p: dict (system parameters)
        Output:
            derivList: np.vector of floats (initial conditions of derivatives).
            
    NOTE: >= 4th order library terms cause failure in Brunton
    """  
    derivList = [
        -p['p0']*z[0] - p['p1']*z[1],
        p['p1']*z[0] - p['p0']*z[1],
        -p['p2']*z[2]
]
    for i in range(p['numExtraVars']):
        derivList.append(0) 
    
    return derivList 

# end of threeDimLinear_fn
# -------------------------------------------------------------------
     
def hopfNormalForm2D_fn(z, t, p=({'p0': 0, 'p1': -1, 'p2': 1, 'numExtraVars': 0})):
    """ Mean field model with zDot == 0:
        xDot = p0*x + p1*y - p2*x*(x^2 + y^2)
        yDot =  -p1*x + p0*y - p2*y*(x^2 + y^2)
        
        where p0 = mu, p1 = omega, p2 = A, p3 = lambda in Brunton paper. 
        
        Note that in the 3D model, zDot = -p2 * (z - x^2 - y^2). In this 2D version we assume 
        lambda is big, so zDot -> 0 rapidly and thus z = x^2 + y^2.
        TO-DO: we need param values for this model. mu is in [-0.2, 0.6]. omega, A, and lambda 
               values are unknown.
        Initial values estimate: x,y = {1, 0.75} or {0,0} (see fig 3 in Brunton paper)
        
        Inputs: 
            z: np.vector of floats (initial conditions)
            t: np.vector of floats (timesteps)
            p: dict (system parameters)
        Output:
            derivList: np.vector of floats (initial conditions of derivatives).
            """  
    derivList = [
        p['p0'] * z[0] - p['p1'] * z[1] + p['p2'] * z[0] * (pow(z[0], 2) + pow(z[1], 2)),
        p['p1'] * z[0] + p['p0'] * z[1] + p['p2'] * z[1] * (pow(z[0], 2) + pow(z[1], 2))  
] 
    for i in range(p['numExtraVars']):
        derivList.append(0)
    
    return derivList 
 
# end of hopfNormalForm2D_fn
# -------------------------------------------------------------------
         
def hopfNormalForm3D_fn(z, t, p=({'p0': 0, 'p1': -1, 'p2': 1, 'p3': 0.5, 'numExtraVars': 0})):
    """ Mean field model with zDot == 0:
        xDot = p0*x + p1*y - p2*x*z 
        yDot =  -p1*x +p0*y - p2*y*z
        zDot = -p3 * (z - x^2 - y^2).
        
        where p0 = mu, p1 = omega, p2 = A, p3 = lambda in Brunton paper.
        In this 3D version of the model, we assume lambda is not too big, so zDot is not == 0.
        
        TO-DO: We need param values for this model. mu is in [-0.2, 0.6]. omega, A, and lambda 
        values are unknown. See tables 10, 11 in Brunton paper S.I. 
        Question: is mu being used in two ways, as a coefficient and as a "bifurcation parameter" 
                  (see eg table 13)?
        Initial values estimate: x,y = {1, 0.75} or {0,0} (see fig 3 in Brunton paper)
        
        Inputs: 
            z: np.vector of floats (initial conditions)
            t: np.vector of floats (timesteps)
            p: dict (system parameters)
        Output:
            derivList: np.vector of floats (initial conditions of derivatives).
            """  
    derivList = [
        p['p0'] * z[0] - p['p1'] * z[1] + p['p2'] * z[0] * z[2], # (pow(z[0], 2) + pow(z[1], 2)),
        p['p1'] * z[0] + p['p0'] * z[1] + p['p2'] * z[1] * z[2], # (pow(z[0], 2) + pow(z[1], 2)),
        -p['p3'] * (z[2] - pow(z[0],2) - pow(z[1], 2))]
    
    for i in range(p['numExtraVars']):
        derivList.append(0)
    
    return derivList 

# end of hopfNormalForm3D_fn
# ------------------------------------------------------------------

def generateModelStrAndTrueArrays_fn(modelSystem, p):
    """
    Build a string describing the model. 
    
    Parameters
    ----------
    modelSystem : str
    p : dict

    Returns
    -------
    modelStr : str.
    trueLib : tuple of lists of str
    trueLibCoeffs : tuple of lists of floats

    """
    if modelSystem == 'lorenz':   
        modelStr = \
        "x' = -" + str(p['p0']) + ' x + ' + str(p['p0']) + ' y' + '\n' + \
        "y' = " + str(p['p1']) + ' x - y - x*z' + '\n' + \
        "z' = " + str(p['p2']) + ' z' + ' + x*y' 
        trueLib = (['x','y'], ['x','y','x*z'], ['z', 'x*y'])
        trueLibCoeffs = ([-p['p0'], p['p0']], [p['p1'], -1, -1], [p['p2'], 1])
    if modelSystem == 'harmOscLinear':  
        modelStr = \
        "x' = -" + str(p['p0']) + ' x + ' + str(p['p1']) + ' y' + '\n' + \
        "y' = -" + str(p['p1']) + " x -" + str(p['p0']) + ' y' 
        trueLib = (['x', 'y'], ['x', 'y'])
        trueLibCoeffs = ([-p['p0'], p['p1']], [-p['p1'], -p['p0']])
    if modelSystem == 'harmOscCubic':  
        modelStr = \
        "x' = -" + str(p['p0']) + ' x^3 + ' + str(p['p1']) + ' y^3' + '\n' + \
        "y' = -" + str(p['p1']) + " x^3 -" + str(p['p0']) + ' y^3' 
        trueLib = (['x^3', 'y^3'], ['x^3', 'y^3'])
        trueLibCoeffs = ([-p['p0'], p['p1']], [-p['p1'], -p['p0']])
    if modelSystem == 'threeDimLinear':        
        modelStr = \
        "x' = -" + str(p['p0']) + ' x - ' + str(p['p1']) + ' y' + '\n' + \
        "y' = " + str(p['p1']) + " x -" + str(p['p0']) + ' y' + '\n' + \
        "z' = -" + str(p['p2']) + " z" 
        trueLib = (['x', 'y'], ['x', 'y'], ['z'])
        trueLibCoeffs = ([-p['p0'], -p['p1']], [p['p1'], -p['p0']], [-p['p2']])
    if modelSystem == 'hopfNormal2D':         
        modelStr = \
        "x' = " + str(p['p0']) + ' x + ' + str(p['p1']) + ' y ' + "+ " + str(p['p2']) + \
                '(x^3 + x*y^2)' + '\n' + \
        "y' = " + str(p['p1']) + ' x + ' + str(p['p0']) + ' y ' + "+ " + str(p['p2']) + \
                '(y*x^2 + y^3)' 
        trueLib = (['x', 'y', 'x^3', 'x*y^2'], ['x', 'y', 'y^3', 'x^2*y'])
        trueLibCoeffs = ([p['p0'], p['p1'], p['p2'], p['p2']], 
                         [p['p1'], p['p0'], p['p2'], p['p2']])
    if modelSystem == 'hopfNormal3D':        
        modelStr = \
        "x' = " + str(p['p0']) + ' x - ' + str(p['p1']) + ' y ' + "+ " + str(p['p2']) + \
                ' x*z' + '\n' + \
        "y' = " + str(p['p1']) + ' x + ' + str(p['p0']) + ' y ' + "+ " + str(p['p2']) + \
                ' y*z' + '\n' + \
        "z' = -" + str(p['p3']) + ' * (z - x^2 - y^2)' 
        trueLib = (['x', 'y', 'x*z'], ['x', 'y', 'y*z'], ['z', 'x^2', 'y^2'])
        trueLibCoeffs = ([p['p0'], p['p1'], p['p2']], 
                         [p['p1'], p['p0'], p['p2']], [-p['p3'], p['p3'], p['p3']])
        
    return modelStr, trueLib, trueLibCoeffs

# End of generateModelStrAndTrueArrays_fn 
# ---------------------------------------------------

def generateTrueFunctionalAndCoeffArrays_fn(trueLib, trueLibCoeffs, functionList):
    """
    Given lists of functional names as str, and the function list, construct a true 
    'functionsToUseArray'

    Parameters
    ----------
    trueLib : list-like of lists of str. len of list-like = numVars, len of each list = num true
              library functionals for that variable
    trueLibCoeffs : list-like of lists of floats. Matches 'trueLib' above.
    functionList : list of str.  The functional names

    Returns
    -------
    trueLibraryArray : np.array of bools, numVars x numFunctionals

    """
    trueLibraryArray = np.zeros((len(trueLib), len(functionList)), dtype=bool)
    trueCoeffArray = np.zeros((len(trueLib), len(functionList)))
    for v in range(len(trueLib)):  # v is the variable index
        theseFnalNames = np.array(trueLib[v])
        theseCoeffs = np.array(trueLibCoeffs[v])
        for f in range(len(functionList)):
            ind = np.where(theseFnalNames == functionList[f])[0]
            if len(ind) > 0:  # ie functionList[f] is a true functional
                trueLibraryArray[v, f] = True  
                trueCoeffArray[v, f] = theseCoeffs[ind]
    
    return trueLibraryArray, trueCoeffArray

# End of generateTrueFunctionalAndCoeffArrays_fn
# ----------------------------------------------------------
    
def generateFunctionStr_fn(v, varInds, variableNames):
    """
    Given a list, generate a string. Used by generatePolynomialLibrary_fn.

    Parameters
    ----------
    v : list of ints
    varInds : list of ints
    variableNames : list of str

    Returns
    -------
    fnStr : str

    """
    fnStr = ''
    for i in varInds:
        if i in v:
            if len(fnStr) > 0:  # case: we need a multiplication sign:
                fnStr += '*'
            fnStr += variableNames[i]
            if np.sum(np.array(v) == i) > 1: # case: we need an exponent:
                fnStr += '^' + str(np.sum(np.array(v) == i))
    return fnStr
# End of generateFunctionStr_fn
#------------------------------------------------------------------------
 
def generatePolynomialLibrary_fn(variableNames, degree):
    """
    Generate a library of polynomials up to a certain degree. Return two things: a list of 
    functional names and a list of recipes for use in ode evolutions.
    NOTE: If there is one variable, and its name is more that one character, this function will 
    fail because 'varInds' will equal the number of characters in the variable name.

    Parameters
    ----------
    variableNames : list-like of str 
    degree : int

    Returns
    -------
    functionList : list of str 
    recipes : list of lists, length = numFunctionals
    """

    varInds = np.arange(len(variableNames))
    recipes = []
    functionList = []
    recipes.append(-1)   # the constant function
    functionList.append('1') 
    
    # Treat degree = 1:
    combos = []  # initialize  
    for i in varInds:
        combos.append(i)    
        recipes.append(i)
        functionList.append(variableNames[i])
        
    deg = 2  # initialize 
    while deg <= degree: 
        combos = [(i, j) for i in varInds for j in combos]  # vector of {int, list} entries, 
        # entry has total len = deg. There are duplicates at >= degree 3, eg (0,(0,1)) and 
        # (1,(0,0)). So for each entry we must (a) make into a single list; and (b) check that it
        # is new before appending it to 'recipes' and 'functionList'.
        # (a) combine int and list:
        keepCombos = []  # to keep the non-duplicates
        for k in range(len(combos)): 
            c = combos[k]    
            this = []   
            for i in c:
                if isinstance(i, (int, np.integer)):
                    this.append(i)
                else:
                    for j in i:
                        this.append(j)
            this = list(np.sort(np.array(this)))
            # 'this' is now a single sorted list of ints.
            # (b) If 'this' is new, append to recipes:
            addFlag = True
            for i in recipes:
                if not isinstance(i, (int, np.integer)):
                    if len(this) == len(i) and np.sum(np.array(this) == np.array(i)) == len(i):
                        addFlag = False
                        break
            if addFlag:
                recipes.append(this)  
                keepCombos.append(this)  
                functionList.append(generateFunctionStr_fn(this, varInds, variableNames))  
                    
        # Update combos with non-duplicate list:
        combos = keepCombos
        deg += 1
    
    return functionList, recipes

# End of generatePolynomialLibrary_fn
#--------------------------------------------------------------

def calculateLibraryFunctionalValues_fn(x, recipes):
    """
    For each functional, calculate its values at the timepoints.

    Parameters
    ----------
    x : np.array of floats, numTimepoints x numVars
    recipes : list of lists. The i'th list gives the indices of variables to be multiplied together 
              to generate the i'th functional.

    Returns
    -------
    fnVals : np.array of floats, numTimepoints x numFunctionals.
    """
    
    fnVals = np.zeros((x.shape[0], len(recipes)))
    for i in range(fnVals.shape[1]):
        r = recipes[i]
        temp = np.ones(x.shape[0])
        if isinstance(r, (int, np.integer)):  # constant or degree 1 monomial
            if r != -1:  # ie not the constant functional 
                temp = x[:, r]
        else:  # >= degree 2
            for j in r:
                temp = temp * x[:, j]
        fnVals[:, i] = temp
        
    return fnVals

# End of calculateLibraryFunctionalValues_fn
#------------------------------------------------------------

#%% Make a hamming window:
def makeHammingWindow_fn(hammingWindowLength, plateauRatio=0):
    """"  Generate a hamming window, perhaps with a flat plateau in the middle (ie a smoothed step
          function).
    Inputs:
        hammingWindowLength: int
        usePlateauHammingFilterFlag: Boolean
        plateauRatio: float 0 to 1
    Outputs:
        hamm: vector with sum = 1
        """
    if plateauRatio > 0:
        # add a plateau in the middle:
        plateauRatio = min(1, plateauRatio)
        ends = int(np.ceil(hammingWindowLength*(1 - plateauRatio)))
        if ends%2 == 1:
            ends = ends + 1   # make even
        rise = int(ends/2)
        ends = np.hamming(ends)    # ends is now a hamming vector
        hamm = np.ones((1, hammingWindowLength))
        hamm = hamm.flatten()
        hamm[0:rise] = ends[0:rise]
        hamm[-rise:] = ends[-rise:]
    else:
        # normal hamming filter
        hamm = np.hamming(hammingWindowLength)
    # Normalize:
    hamm = hamm / np.sum(hamm)

    return hamm

# End of makeHammingWindow_fn
#---------------------------------------------------------

def calculateSlopeAndStd_fn(x, dt, w):
    ''' given a time-series, do two things:
        1. calculate the deriv at each point by simple rise/run (Euler formula)
        2. calculate the std of a window (size2*h) at each point, using the slope from (1) to
           first tilt the data to roughly slope = 1.
    Inputs:
        z: np.vector
        dt: float
        w: int. Window length
    Outputs:
        slopeX: np.vector
        stdX: np.vector
        meanX: np.vector
        '''
    h = int(np.round(w/2))  # Half the window length
    slopeX = np.zeros(x.shape)
    stdX = np.zeros(x.shape)
    meanX = np.zeros(x.shape)
    # For efficiency, we take the mean of the first window, then update it at each new point:
    for i in range(len(x)):
        if i == h + 1: # First point's window
            b = np.mean(x[i:i + h])
            a = np.mean(x[i-h:i])
        if i > h + 1 and i < len(x) - h:  # all ensuing points' windows
            b = b + (x[i + h] - x[i])/h
            a = a + (x[i] - x[i-h])/h
        if i > h and i < len(x) - h:  # all points' windows (ie happens for both above cases)
            slopeX[i] = (b-a)/h
            tilted = x[i-h:i+h] - slopeX[i]*np.array(range(-h, h))
            stdX[i] = np.std(tilted)  # subsampling doesn't speed this up much.
            meanX[i] = 0.5 * (b + a)

    # Fill in start and end values:
    slopeX[0:h + 1] = slopeX[h + 1]
    slopeX[-h:] = slopeX[-h - 1]
    stdX[0:h + 1] = stdX[h + 1]
    stdX[-h:] = stdX[-h - 1]
    meanX[0:h + 1] = meanX[h + 1]
    meanX[-h:] = meanX[-h - 1]

    # account for dt:
    slopeX = slopeX/dt

    return slopeX, stdX, meanX

# End of calculateSlopeAndStd_fn

#------------------------------------------------------------------

def addGaussianNoise_fn(X, noiseFactors, noiseType = 'normal'):
    """ add noise (default gaussian) directly to the trajectory array.
    Inputs:
        X: n x m np.array. n = number of variables, m = number of timepoints, ie each row is
        variable time-course
        noiseFactors: np.array of floats, num real vars x 1
        noiseType: str,  'normal' or 'uniform'
        extraVarsnoiseFactors: float
    Outputs:
        XwithNoise: n x m np.array
        """

    noiseArray = np.tile(noiseFactors, [X.shape[0], 1])
    # Add  noise to xTrain, scaled as fraction of std dev of x, y, z values over the trajectory:
    scalingVals  = np.std(X, axis = 0)  # for scaling noise. Use only x, y, z
    # 'scalingVals' for extra variables currently == 0, so replace these with the mean of others:
    scalingVals[scalingVals < 1e-5] = np.mean(scalingVals[scalingVals > 1e-5])

    # Add rows for the extra variables:
    if noiseType == 'uniform':
        noiseToAdd = 2 * noiseArray * np.multiply(np.tile(scalingVals, (X.shape[0], 1)),
                                                 -0.5 + np.random.uniform(size = X.shape))
    else: #  noiseType == 'normal':
        noiseToAdd = noiseArray * np.multiply(np.tile(scalingVals, (X.shape[0], 1)),
                                              np.random.normal(size = X.shape))
    xNoisy = X + noiseToAdd

    return xNoisy

# End of addNoise_fn
#-----------------------------------------------------------------------------------------

def addWhiteNoise_fn(xArray, noiseFactors):
    ''' Add white noise to trajectory array, using FFT.
    Inputs:
        xArray: np.array, numTimepoints x numVars
        noiseFactors:  np.array of floats, num real vars x 1
    Outputs:
        xNoisy: np.array, real part of ifft of noisy fft signal.
        '''

    xNoisy = np.zeros(xArray.shape)
    for i in range(xArray.shape[1]):
        x = xArray[:, i]
        xF = np.fft.fft(x)
        realSigma = np.std(np.real(xF))
        imSigma = np.std(np.imag(xF))
        noiseToAdd = noiseFactors[i] * (realSigma * np.random.normal(size=xF.shape) + \
                                        1j * imSigma * np.random.normal(size=xF.shape))
        xFNoisy = xF + noiseToAdd
        xNoisy[:, i] = np.fft.ifft(xFNoisy)
    return np.real(xNoisy)

# End of addWhiteNoise_fn
#--------------------------------------------------------------------------------------

def calcWeightArray_fn(cNew, functionsToUseArray, imputed, coeffWeightsCutoffFactor, 
                       percentileOfImputedValuesForWeights):
    """
    Create a weight array to weight the coeffs by size of functional values. Culling will be
    based on the weighted coeffs.
    We give high weights to functionals that tend to have high values, since we want to allow
    them to have lower coeffs without being culled. Functionals that tend to have low values
    get low weights, since their effect will be less. We 'normalize' using the median of
    all functional imputed values.

    Parameters
    ----------
    cNew : np.array of floats. numVars x numFunctionals.
    functionsToUseArray : np.array of booleans. numVars x numFunctionals.
    imputed : np.array of floats. vector 1 x numFunctionals. Estimated magnitudes of each
              functional. 
    coeffWeightsCutoffFactor : (scalar float or int)
    percentileOfImputedValuesForWeights : scalar int

    Returns
    -------
    weightArray : np.array of floats. numVars x numFunctions.

    """
    # Reduce the extremes of this vector in each row (ie for each variable), and normalize:
    weightArray= np.zeros(cNew.shape)
    for i in range(weightArray.shape[0]):
        if np.sum(functionsToUseArray[i, :]) > 0:
            temp = imputed.copy()
            centralVal = np.percentile(temp[functionsToUseArray[i, :]],
                                       percentileOfImputedValuesForWeights, 
                                       interpolation='lower')  
            # Over functionals currently active for this variable.
            # Moderate the extremes of this vector:
            temp[temp >  coeffWeightsCutoffFactor * centralVal] = \
                 coeffWeightsCutoffFactor * centralVal
            temp[temp < centralVal / coeffWeightsCutoffFactor] = \
                centralVal / coeffWeightsCutoffFactor
            temp = temp * functionsToUseArray[i, :]  # Zero the weights of unused functions.
            # Normalize (comment: It would be nice if the normalization here penalized variables
            # with many active functionals, to enforce sparsity.):
            temp = temp / np.median(temp[temp > 0])  # So in the ballpark of 1
            weightArray[i, :] = temp

    return weightArray
#-------- End of calcWeightArray_fn-----------------

def calculateFftForRegression_fn(x, numFftPoints, fftRegressionTarget):
    """
    Create a vector using some form of FFT, to act as a regression target. 

    Parameters
    ----------
    x : np.array (vector) of floats
    numFftPoints : int
    fftRegressionTarget : str

    Returns
    -------
    xT : np.array (vector) of floats

    """
    x = x - np.mean(x)
    xP = np.fft.fft(x)  # Use this if fftRegressionTarget == 'complex'
    if fftRegressionTarget == 'realOnly':
        xP = np.real(xP)
    if fftRegressionTarget == 'magnitude':
        xP = np.abs(xP)
    if fftRegressionTarget == 'power':
        xP = pow(np.real(xP), 2)
        
    xP = xP[0:numFftPoints] / np.sum(np.abs(xP[0:numFftPoints]))  # normalize
    
    return xP
# ------- End of calculateFftRegressionTarget_fn ----------------------------

def cullAndAssessWhetherToRerun_fn(localCoeffArray, variableNames,  functionList, 
                                   imputedSizeOfFunctionals, cullingRulesDict, functionsToUseArray,
                                   cullable, inBalanceVarsArray, coeffWeightsCutoffFactor,
                                   percentileOfImputedValuesForWeights):
    """
    Given results of post-smoothing sindy or linear fit, see if we can cull any variables or
    library functions.
    We make an array of booleans that shows which coeffs of the model were non-zero. This array
    serves as modelActiveLib, restricting which library functions can be used for each of
    the variables. No variables or functions are actually removed. Instead, they are set off-limits
    by modifying the boolean array.

    Parameters
    ----------
    localCoeffArray : np.array of floats, numVars x numFunctionals
    variableNames : vector of strings
    functionList : vector of strings
    imputedSizeOfFunctionals : np.array of floats
    cullingRulesDict : Dict with keys like 'maxAllowedWeightedCoeff'
    functionsToUseArray : np.array booleans, numVars x numFunctionals
    cullable : np.array booleans, numVars x numFunctionals
    inBalanceVarsArray :  np.array booleans, numVars x numFunctionals. True = eligible for culling 
    coeffWeightsCutoffFactor : scalar float 
    percentileOfImputedValuesForWeights : int

    Returns
    -------
    iterateAgainFlag : Boolean
    functionsToUseArray : np.array of booleans, numVars x numFunctions
    outputStr : str
    minNonZeroVal : float, the minimum weighted coeff (to use as a stopping signal)
    """
 
    numExtraToKill = cullingRulesDict['extraNumFnsCulledPerIter'] 
    minAllowedWeightedCoeff = cullingRulesDict['minAllowedWeightedCoeff']
    # Note: these are trivially == 1 and have no effect if 'coeffWeightsCutoffFactor' == 1:
     
    cOld = localCoeffArray  # i'th col corresponds to the i'th entry in functionList. j'th
    #row corresponds to j'th variable.
    cNew = cOld.copy()           # we'll zero out entries in this array
    fOld = functionsToUseArray.copy()  # fOld is for comparison, to determine 'iterateAgainFlag', 
    # since we'll update functionsToUseArray.    
    # We could also get fOld from modelActiveLib

    # Kill the constant term if it is the only non-zero term:
    if cullingRulesDict['setConstantDerivEstimatesToZero']:
        for i in range(cNew.shape[0]):
            if np.sum(np.abs(cNew[i, 1:])) < 1e-5:  # 1e-5 = almostZeroRoundDownThreshold, which
            # deals with weird not-quite-zero zeros
                cNew[i,0] = 0

    # 1. find which vars are non-zero:
    zeroedVars =  np.sum(cNew, axis = 1) == 0 # col vector, True -> var has been zeroed out.

    # 2. Now zero out all functions that involve zeroed-out variables, ie zero out columns that
    # contain the var name:
    # Note: this has not been updated to use 'cullable'.
    for i in range(len(zeroedVars)):
        if zeroedVars[i] == True:
            varName = variableNames[i]
            for j in range(cNew.shape[1]):
                if varName in functionList[j]:
                    cNew[:,j] = 0

    # 3. Create a weight array to weight the coeffs by size of functional values.
    functionsToUseArray = np.abs(cNew) > 1e-5  # almostZeroRoundDownThreshold
    if coeffWeightsCutoffFactor > 1:  # Recall coeffWeightsCutoffFactor = 1 -> no weights.
        weightArray = calcWeightArray_fn(cNew, functionsToUseArray, imputedSizeOfFunctionals, 
                                         coeffWeightsCutoffFactor, 
                                         percentileOfImputedValuesForWeights)
    else:  # Case: we're not weighting the coeffs.
        weightArray = np.ones(cNew.shape) / cNew.shape[1]
    # weightArray is now ready to multiply cNew.

    # 3. Now zero out all functions with coeffs that are too large. Currently never activates (at
    # maxAllowed = 50: we can disable this by picking a very large 'maxAllowedWeightedCoeff' 
    # parameter. inBalanceVarsArray == true prevents culling fnals from vars with low fnal counts.
    tooBigStr = ''
    # cNewWeighted = cNew * weightArray
    # tooBigFnsArray = np.logical_and(np.abs(cNewWeighted) > maxAllowedWeightedCoeff,
    #                                 inBalanceVarsArray)
    # make string of too-big functionals:
    # for i in np.where(np.sum(tooBigFnsArray,axis=1) > 0)[0]:  # ie rows/variables with too-small
    # # functionals.
    #     tooBigStr = tooBigStr + variableNames[i] + ': ' + \
    #         str(np.array(functionList)[tooBigFnsArray[i, :]]) + ' '
    # cNew[np.where(tooBigFnsArray)] = 0  # set too-big coeffs to 0

    # numTooBigCulledFunctionals = np.sum(tooBigFnsArray.flatten())

    # 4. Make an updated boolean array of retained functions. Also cull any functionals with very
    # low weighted coeffs:
    cNewWeighted = cNew * weightArray
    tooSmallFnsArray = np.logical_and.reduce((np.abs(cNewWeighted) < minAllowedWeightedCoeff,
                                      np.abs(cNewWeighted) > 0, inBalanceVarsArray))  
    cNew[np.where(tooSmallFnsArray)] = 0  # set too-small coeffs to 0 
    # make string of too-small functionals:
    tooSmallStr = ''
    for i in np.where(np.sum(tooSmallFnsArray,axis=1) > 0)[0]:  # ie rows/variables with too-small
    # functionals.
        tooSmallStr = tooSmallStr + variableNames[i] + ': ' + \
            str(np.array(functionList)[tooSmallFnsArray[i, :]]) + ' '
    numTooSmallCulledFunctionals = np.sum(tooSmallFnsArray.flatten())

    # 5. see if a variable has been removed:
    oldZeroedVars = np.sum(cOld, axis = 1) == 0
    variableRemovedFlag =  np.sum(zeroedVars) > np.sum(oldZeroedVars)

    functionsToUseArray = np.abs(cNew) > 1e-5  # almostZeroRoundDownThreshold  # Update to remove 
    # too small coeffs
    
    # 6. See if an entire function (a column) has been removed:
    oldZeroedFns = np.sum(fOld, axis = 0)  # Sum the columns
    newZeroedFns = np.sum(functionsToUseArray, axis = 0)
    functionColumnRemovedFlag = np.sum(newZeroedFns == 0) - np.sum(oldZeroedFns == 0) > 0
    # 7. Cull based on weighted coeff size relative to current threshold:
    # If we have not removed a functional column, and if the lowest weighted coeff is small enough,
    # cull more coeffs. 
    # Make a new weight array, since we may have removed some functionals: 
    if coeffWeightsCutoffFactor > 1:  # Recall coeffWeightsCutoffFactor = 1 -> no weights.
        weightArray = calcWeightArray_fn(cNew, functionsToUseArray, imputedSizeOfFunctionals, 
                                         coeffWeightsCutoffFactor, 
                                         percentileOfImputedValuesForWeights)
    else:  # Case: we're not weighting the coeffs.
        weightArray = np.ones(cNew.shape) / cNew.shape[1]  
        
    cNewWeighted = cNew * weightArray
    
    if np.sum(np.abs(cNewWeighted.flatten())) > 1e-5:  # almostZeroRoundDownThreshold:
        minNonZeroVal = np.min(np.abs(cNewWeighted)[np.logical_and(cullable,
                                                                   cNewWeighted != 0)].flatten())
    else:
        minNonZeroVal = 0   
 
    extraFunctionalKilledOffFlag = False
    numExtraFunctionalsKilled = 0
    extraCullBypassedDueToRemovedFunctionColumnFlag = functionColumnRemovedFlag

    # If we removed a function from all vars (ie a column), we're done for this cull. Else see if
    # any functionals have coeffs below the current threshold. We ignore the protected fns (ie we
    # only consider 'cullable' fns)
    culledFnsArray = np.zeros(functionsToUseArray.shape, dtype=bool)  # to record functionals
    # culled because they were the most below the current threshold (but not including 'too-small'
    # fns).
    while numExtraToKill > 0 and (functionColumnRemovedFlag == False) and \
        np.sum(np.abs((cNewWeighted * inBalanceVarsArray).flatten())) > 1e-5:
        # 1e-5 = almostZeroRoundDownThreshold 
        loc = np.where(np.logical_and.reduce((cullable, np.abs(cNewWeighted) == minNonZeroVal,
                                              inBalanceVarsArray)))
        functionsToUseArray[loc[0], loc[1]] = False  # Update the boolean functionals array
        cNewWeighted[loc[0], loc[1]] = 0  # Update the beta weights array
        culledFnsArray[loc[0], loc[1]] = True  # Record this culled functional
        numExtraFunctionalsKilled += len(loc[0])
        # outputStr = outputStr + '; ' + str(functionList[loc[1][0]]) + \
        #     ' (' + str(variableNames[loc[0][0]] + ')')
        extraFunctionalKilledOffFlag = True
        # Update 'functionRemovedFlag' accounting for newly-culled function:
        newZeroedFns = np.sum(functionsToUseArray > 0, axis = 0)
        functionColumnRemovedFlag = np.sum(newZeroedFns == 0) - np.sum(oldZeroedFns == 0) > 0
            
        temp = cNewWeighted * inBalanceVarsArray * cullable
        if np.sum(np.abs(temp.flatten())) > 1e-5:  # almostZeroRoundDownThreshold
            minNonZeroVal = np.min(np.abs(temp[temp != 0].flatten())) 
            if 'loc' in locals():
                numExtraToKill -= len(loc[0])
            else:
                numExtraToKill -= 1
        else:  # escape
            minNonZeroVal = 0
            numExtraToKill = 0
            
        

    culledStr = ''
    # List the culled fns in a str:
    for i in np.where(np.sum(culledFnsArray, axis=1) > 0)[0]:
        culledStr = culledStr + variableNames[i] + ': ' + \
            str(np.array(functionList)[culledFnsArray[i, :]]) + ' ' 

    # 8. If there are any functions still below thisThreshold, we want to rerun without changing 
    # the threshold, in order to pick them off one by one:
    if np.sum(np.abs(cNewWeighted.flatten())) > 1e-5 and \
        np.sum(np.logical_and(cullable, cNewWeighted != 0).flatten()) > 0:
        newMinNonZeroVal = np.min(np.abs(cNewWeighted)[np.logical_and(cullable,
                                                                  cNewWeighted != 0)].flatten())    
        cullableFunctionFlag = True
    else:
        cullableFunctionFlag = False

    # Make a combined too-big, too-small output str:
    # if numTooBigCulledFunctionals > 0:
    #     tooBigStr = '. Culled ' + tooBigStr + ' with weighted coeffs > ' + \
    #         str(maxAllowedWeightedCoeff) + '. '
    if numTooSmallCulledFunctionals > 0:
        tooSmallStr = 'Culled ' + tooSmallStr + \
            ' with weighted coeffs < ' + str(minAllowedWeightedCoeff) + '. '
    if numExtraFunctionalsKilled > 0:
        culledStr = 'Culled ' + culledStr
    else:
        culledStr = ''
        
    if extraCullBypassedDueToRemovedFunctionColumnFlag:
        culledStr = ' Extra culling step bypassed due to removed function column.'

    outputStr = tooBigStr + tooSmallStr + culledStr
    
    # if any new variable, new function, or even one new functional has been removed, we will do
    # another cull.
    iterateAgainFlag = variableRemovedFlag or functionColumnRemovedFlag or \
        extraFunctionalKilledOffFlag or cullableFunctionFlag

    # Note: the returned arg 'culledFnsArray' shows the functionals culled in a way (by being 
    # below the ratcheting threshold) that makes them eligible to be restored. Functionals that 
    # have too-big or too-small weighted coeffs are not restorable.

    return iterateAgainFlag, functionsToUseArray, culledFnsArray, outputStr, \
        minNonZeroVal, cullableFunctionFlag

# End of cullAndAssessWhetherToRerun_fn 
# -----------------------------------------------------------------------------------------
  

def smoothData_fn(x, window, smoothType='hamming'):
    """
    Smooth each column of an input array using a window.
    
    Parameters
    ----------
    x : np.array of floats, likely numTimepoints x numSomething (variables or functionals)
    window : np.array, vector of floats
    smoothType : str, eg 'hamming' 

    Returns
    -------
    xSmoothed : np.array, size x.shape

    """
    xSmoothed = np.zeros(x.shape)
    if True:  # smoothType == 'hamming': Assume it's always hamming
        for i in range(x.shape[1]):
            temp = x[:, i] 
            xSmoothed[:, i] = np.convolve(temp - temp[0], window, mode='same') + temp[0]
            # shift time-series to start at 0 for the convolution, to mitigate edge effects

    return xSmoothed
#  End of smoothLibraryFunctionsForRegression_fn
#-----------------------------------------------------------------------

def parseTimepointsByMagnitudesOfVariables_fn(x, functionsArray, nonConstantIndices, margin, 
                                              maxFnValRatio, minNumStartIndsToUse, 
                                              removeMarginFlag=True):
    """
    Accept or reject timepoints to use in regressions, based on whether the functionals have
    widely disparate values or not.

    Parameters
    ----------
    x : np.array of floats, numTimepoints x numFunctionals. Each column is time-series of a
        functional's values.
    functionsArray : np.array of booleans, numVariables x numFunctionals.
    nonConstantIndices : list of ints (generated after the functional library is created)
    margin : int
    maxFnValRatio : float
    minNumStartIndsToUse : int
    removeMarginFlag : bool
  
    Returns
    -------
    startIndsToUse : np.array of booleans numVariables x numTimepoints (not counting margins at
                     either end).

    """
    lengthOfPadToIgnore = 0
    if removeMarginFlag:
        lengthOfPadToIgnore = margin
    startIndsToUse = np.zeros((functionsArray.shape[0], x.shape[0] - 2*lengthOfPadToIgnore),
                              dtype=bool)

    for var in range(functionsArray.shape[0]):
        if np.sum(functionsArray[var,]) == 0:  # Ignore fully culled variables
            pass
        else:
            thisX = x * np.tile(functionsArray[var,], (x.shape[0], 1))
            if removeMarginFlag:
                thisX = thisX[margin:-margin,]  # Ignore outer margins.
            # At spaced timepoints, find the median of abs() of non-zero values, and use
            # that as an anchor for scaling:
            okToUse = np.ones(thisX.shape, dtype=bool)
            subsampleRate = 3
            for t in range(0, thisX.shape[0], subsampleRate):
                v = np.abs(thisX[t,])
                m = np.median(v[v != 0])
                okToUse[t:t + subsampleRate,] = \
                    np.logical_and.reduce((v < m * maxFnValRatio, v > m / maxFnValRatio,
                                           functionsArray[var,]))
            # 'okToUse' is a boolean array that tells us, for each timepoint, which functionals 
            # are ok to regress on. We now select the timepoints that are valid for the highest 
            # number of functionals: 

            numOkFnsAtEachTimepoint = np.sum(okToUse, axis=1)
            # 'removeMarginFlag' = False is a signal that we are testing linear dependence, so we
            # wish to use all the columns of functionsArray because we have already subselected 
            # the functionals of interest.
            if removeMarginFlag:
                counts = np.zeros((1, int(np.sum(functionsArray[var, nonConstantIndices]))))
            else:  # case: lin dependence context, use all columns of functionsArray
                counts = np.zeros((1, int(np.sum(functionsArray[var, :]))))
            for i in range(counts.shape[1]):
                counts[0, i] = np.sum(numOkFnsAtEachTimepoint == i + 1)  # sum the columns
                # to get number of useable timepoints.
                
            # Keep only those timepoints that use all non-constant available fns
            # (captured in 'countValToUse'), ie ignore any timepoints where any
            # two functions exceed the maxFnValRatio-derived bounds, EXCEPT subject to a 
            # guaranteed minimum number of timepoints 'minNumStartIndsToUse', enforced in the 
            # while loop below.
            if removeMarginFlag:
                countValToUse = np.sum(functionsArray[var, nonConstantIndices])
            else:  # case: lin dependence context, use all columns of functionsArray
                countValToUse = np.sum(functionsArray[var, :])
            # fullCountValToUse = countValToUse.copy()  # used for diagnostic printout below.
            startIndsToUse[var,] = numOkFnsAtEachTimepoint >= countValToUse
            # Add a catch to prevent too few startInds:
            while np.sum(startIndsToUse[var,]) < minNumStartIndsToUse:
                countValToUse = countValToUse - 1
                startIndsToUse[var,] = numOkFnsAtEachTimepoint >= countValToUse
            # Diagnostic printount:
            # print('var ' + str(var) + ': functional count used for selecting timepoints: ' + \
            #       str(countValToUse) + ' out of ' + str(fullCountValToUse) + \
            #       ', numTimepoints = ' + str(np.sum(startIndsToUse[var,])) + ' out of ' + \
            #       str(startIndsToUse.shape[1])) 
                
    # 'startIndsToUse' is an array of booleans, which say whether a certain
    # timepoint (starting at 'margin' and ending at {len(tTrain) - margin}) is eligible
    # to use.

    return startIndsToUse

#  End of parseTimepointsByMagnitudesOfVariables_fn
# ---------------------------------------------------------------------------

def calculateDerivativesFromData_fn(x, t, method='centralDifference'):
    """
    Given time-series of variables x, calculate the derivatives of each variable. 
    This function differs from 'estimateDerivatives_fn' below by (a) accepting arrays (ie multiple
    time-series); (b) handling endpoints; (c) not returning weights. Could be combined.
    NOTE: Currently only allows central difference method

    Parameters
    ----------
    x: np.array of floats, numTimepoints x numVars 
    t : np.vector of floats. The timepoints

    Returns
    -------
    df : np.array of floats, numTimepoints x numFunctionals

    """
    t = t.reshape(-1,1)
    # if True method == 'centralDifference':
        
    dfMiddle = (x[2:, :] - x[0:-2, :]) / np.tile(t[2:] - t[0:-2], (1, x.shape[1]))
    dfFirst = (x[1, :] - x[0, :]) / (t[1] - t[0])
    dfLast = (x[-1, :] - x[-2, :]) / (t[-1] - t[-2])
    
    df = np.vstack((dfFirst, dfMiddle, dfLast))
        
    return df

# End of calculateDerivativesFromData_fn
#------------------------------------------------------------------

def calculateDerivativesFromModel_fn(x, coeffs, fVals):
    """
    Given a model with coefficients for functionals, calculate the derivatives based on this model,
    ie return what the model thinks the derivatives are.
    This is different from 'calculateDerivativesFromData_fn' above, which returns the central difference 
    derivative of the time-series.

    Parameters
    ----------
    x : np.array of floats, numTimepoints x numVars
    coeffs : np.array of floats, numVars x numFunctionals
    fVals : np.array of floats, numTimepoints x numFunctionals

    Returns
    -------
    xDotByModel : np.array of floats, numTimepoints x numVars

    """
    
    xDotByModel = np.zeros(x.shape)
    for i in range(x.shape[1]):
        xDotByModel[:, i] = np.sum(fVals * np.tile(coeffs[i, :], (fVals.shape[0], 1)), axis=1)
        
    return xDotByModel

# End of calculateDerivativesFromModel_fn
#----------------------------------------------------------------------

def estimateDerivatives_fn(xS, startInds, numDtStepsForDeriv, pointWtsForThisVar, dt):
    """
    For a vector, calculate the target derivative (as in dx/dt = rise/run) that we hope to match
    when we regress on the library functionals. The target derivative is calculated using the
    variables' time-series. Also return the weights for each element of the target rise, for use
    in the regression.
    NOTE: We require values in xS before the first startInd and beyond the last start ind

    Parameters
    ----------
    xS : np.array of floats, (column vector same length as timepoints eg tTrain), the
    pre-processed time-series of one variable.
    startInds : np.array (vector of ints). Indices of timepoints.
    numDtStepsForDeriv : scalar int
    pointWtsForThisVar :  np.array (vector of floats, same length as timepoints eg tTrain).
    dt : scalar float. The length of the simulation timestep

    Returns
    -------
    derivEstimate : np.array (vector of floats). size = startInds.shape
    weights : np.array (vector of floats). size = startInds.shape

    """
    timeStep = dt*numDtStepsForDeriv  # float (ie a measure of time, not an index of timepoints)

    # 4th order approx to derivative:
    m2Inds = startInds - 2*numDtStepsForDeriv  # indices
    m1Inds = startInds - 1*numDtStepsForDeriv
    p2Inds = startInds + 2*numDtStepsForDeriv
    p1Inds = startInds + 1*numDtStepsForDeriv
    derivEstimates = \
        (xS[m2Inds] - 8*xS[m1Inds] + 8*xS[p1Inds] - xS[p2Inds]) / (12*timeStep)
    # The weight for each deriv estimate combines the timepoint values used:
    weights = \
        pointWtsForThisVar[m2Inds] + 8 * pointWtsForThisVar[m1Inds] + \
            8 * pointWtsForThisVar[p1Inds] + pointWtsForThisVar[p2Inds]
    weights = weights / sum(weights)

    return derivEstimates, weights

# End of estimateDerivatives_fn
#--------------------------------------------------------

def defineXAndYForLinRegressionOnEvolution_fn(xT, functionalVals, startInds, numEvolveSteps,
                                              pointWtsForThisVar, dt):
    """
    Given start inds, create a target y to be fitted by a linear function of functionals by:
    1. define a set of subtargets subY which are individual estimates of derivatives over short
    hops, using a sequence of equally-spaced points.
    2. define an 'evolution' from the first to the last point by Euler stepping. It is not
    a real evolution because the input values of the points at each step are from the given time-
    series, not the prediction from the previous timepoint. We do this to maintain a linear
    relationship between the functionals and the target value:
    (x[t + n] - x[t])/dt = summation( beta_i * (f_i[t] + f_i[t+1] + ... + f_i[t+n-1]) ).
    So X = summations of the functionals over the n timepoints; and y = the difference between
    the two end timepoints, divided by the timestep.

    Parameters
    ----------
    xT : np.array of floats, numTimepoints x numVariables (the pre-processed time-series)
    functionalVals : np.array of floats, numTimepoints x num active functionals  for this
                     variable (the value of the active library functionals at each timepoint).
    startInds : np.array (vector of ints)
    numEvolveSteps : scalar int. How far to evolve to get regression target.
    pointWtsForThisVar : np.array (vector of floats, length = numTimepoints)
    dt : scalar float

    Returns
    -------
    X : np.array of floats, len(startInds) x num active functionals
    y : np.array (vector of floats, same size as 'startInds')
    weights : np.array (vector of floats, same size as 'startInds')
    """
    
    wtsPerPoint = np.zeros((numEvolveSteps, len(startInds)))
    evolveSteps = np.zeros((len(startInds), functionalVals.shape[1], numEvolveSteps))
    for i in range(numEvolveSteps):
        evolveSteps[:, :, i] = functionalVals[startInds + i, :]
        wtsPerPoint[i, :] = pointWtsForThisVar[startInds + i]

    X = np.sum(evolveSteps, axis=2)  # Sum over the evolve steps. The result is
    # len(startInds) x numFunctionals.
    y = (xT[startInds + 1] - xT[startInds]) / dt
    weights = np.median(wtsPerPoint, axis=0)
    weights = weights / np.sum(weights)  # so weights sum to 1
    return X, y, weights

# End of defineXAndYForLinRegressionOnEvolution_fn
#-----------------------------------------------------------

def combineSegmentCoeffs_fn(coeffsPerSeg, printDiagnosticsFlag, varName, fnalName,
                            snrThreshold, minNumSegmentResultsToUse, outputFilename):
    """
    Given a vector of coefficients (found by regressing on many segments, or subsets of
    timepoints) for some {variable, functional} pair, return (i) a median value, and (ii) a
    measure of reliability (stdDev/median). We do this by rejecting outliers until either a
    minimum number of coeffs are left, or the stdDev/median is within some bound.

    Parameters
    ----------
    coeffsPerSeg : np.array (vector of floats). length = number of segments that were used in
                   regressions.
    printDiagnosticsFlag : scalar boolean
    varName : str
    fnalName : str
    snrThreshold : scalar float
    minNumSegmentResultsToUse : scalar int (could be a float)
    outputFilename : str

    Returns
    -------
    coeff : scalar float
    stdOverMedian : scalar float

    """

    # Decide what coeff value to carry forward:
    # 1. If the SNR is low, just use mean or median.
    segsToUse = coeffsPerSeg != -100  # This should not be empty.
    this = coeffsPerSeg[segsToUse].copy()
    m = np.median(this)
    if m == 0:  # to catch an edge case
        m = np.mean(this)
    s = np.std(this)
    # 2. Start removing outliers until snr is low:
    while s/np.abs(m) > snrThreshold and len(this) > minNumSegmentResultsToUse:
        # Eliminate values and retry:
        dist = np.abs(this - m)
        this = this[dist < max(dist)]
        m = np.median(this)
        if m == 0:
            m = np.mean(this)
        s = np.std(this)

    # Assign the final coeffs for this {variable, function} pair:
    coeff = np.median(this)
    stdOverMedian = np.abs(s / m)  # We'll cull functions based on high variability.

    # (Diagnostic) Print out vector of segment coeffs if not too many:
    if printDiagnosticsFlag:  # ie few enough fns in library that we can print the outcome:
        console = sys.stdout
        with open(outputFilename, 'a') as file:
            print(varName + "', " + fnalName + ' values: ' + \
                  str(np.round(coeffsPerSeg,2)) + ', median ' + \
                  str(np.round(np.median(coeffsPerSeg), 2)) + ', std ' + \
                  str(np.round(np.std(coeffsPerSeg), 2)) + '. Final median = ' + \
                  str(np.round(coeff, 2)) + ', final stdOverMedian = ' +  \
                  str(np.round(stdOverMedian, 2)), file=file)
            sys.stdout = console
            file.close()
        

    return coeff, stdOverMedian

# End of combineSegmentCoeffs_fn
#------------------------------------------------------------

def cullBasedOnStdOverMedian_fn(coeffArray, coeffsBySeg, stdOverMedianArray, functionsToUseArray,
                                startIndsToUse, segStartInds, segEndInds, pointWeights, 
                                functionalVals, p):
    """
    Cull functions whose fitted coeffs had high stdDev/Median, ie high variability over the
    various segments. This has two steps:
      1. Cull functionals based on high variability. Do not cull functionals with relatively high
         median (weighted) coefficients.
      2. Do a new regression, to calculate new coeffs for the remaining functionals.
    NOTE: This method gave bad results (though it is the culling criterion used in REF).
    
    Parameters
    ----------
    coeffArray : np.array of floats, numVariables x numFunctionals. The previous coefficients.
    coeffsBySeg : np.array of floats, numVariables x numFunctionals x numSegments. The coefficients 
                  from the new regressions on each segment, to be combined in this function.
    stdOverMedian : np.array of floats, numVariables x numFunctionals
    functionsToUseArray : np.array of booleans, numVariables x numFunctionals
    startIndsToUse : np.array of booleans, (numTimepoints - 2*margin) x numVars
    pointWeights : np.array of floats, numTimepoints x numVars
    functionalVals : np.array of floats, numTimepoints x numFunctionals
    p : dict of params, including: 
        xTrain : np.array of floats, numTimepoints x numVars
        minStdOverMedianThreshold : scalar float
        numFunctionsToTriggerCullBasedOnStdOverMean : scalar int
        maxNumToRemoveByStdPerCull : scalar int
        numDtStepsForDeriv : scalar int
        dt : scalar float (the timestep)
        regressOnFftAlpha : scalar float (the fraction of regression that is on FFTs)
        fftXDotTrainTarget : np.array (numFftPoints x numVars)
        fftLibFns : np.array (numFftPoints x numFunctionals)
        variableNames : list (np.array?) of str
        functionList : list (np.array?) of str
        margin : int
        weightTimepointsFlag : bool
        fftRegressionTarget : str
        snrThreshold : scalar float
        outputFilename : str

    Returns
    -------
    postRegressionCoeffs : np.array of floats, , numVariables x numFunctionals
    functionsToUseArray : np.array of booleans, numVariables x numFunctionals

    """

    # Loop through the variables, handling each one independently:
    for v in range(functionsToUseArray.shape[0]):
        # Check if any functions have high variance and also lower magnitude for this variable,
        # and also if there are few enough functions left to start this type of cull:
        # Calculate stdOverMedian threshold:
        tempStd = stdOverMedianArray[v, functionsToUseArray[v, :]]
        stdThreshold = np.median(tempStd) + 1 * np.std(tempStd)
        stdThreshold = max(stdThreshold, p['minStdOverMedianThreshold'])
        # Calculate coeff median threshold:
        tempMag = np.abs(coeffArray[v, functionsToUseArray[v, :]])
        magThreshold = np.median(tempMag)
        pointWtsForThisVar = pointWeights[:, v]

        # If there are no active functions, stdThreshold and medianThreshold = np.nan
        # Identify noisy functionals: high variability and low magnitude:
        noisyFnInds = np.where(np.logical_and(tempStd > stdThreshold, tempMag < magThreshold))[0]

        numActiveFns = np.sum(functionsToUseArray[v, :])
        if len(noisyFnInds) > 0 and numActiveFns <= \
            p['numFunctionsToTriggerCullBasedOnStdOverMean'] and numActiveFns > 2:
            # Cull only an allowable number of these. Argsort in descending order, then
            # cull the first several:
            inds = (np.argsort(stdOverMedianArray[v, noisyFnInds]))[-1: :-1]
            noisyFnInds = noisyFnInds[inds[0:p['maxNumToRemoveByStdPerCull']]]

            # (Diagnostic) Output to console:
            console = sys.stdout
            with open(p['outputFilename'], 'a') as file:
                print('Culled ' + str(np.array(p['functionList'])[noisyFnInds]) + \
                      ' from variable ' + p['variableNames'][v] + \
                      ' due to coeff stdOverMedian = ' + \
                      str(np.round(tempStd[noisyFnInds], 2)) + ' > ' + \
                      str(np.round(stdThreshold, 2)) + ' and coeff mag = ' + \
                      str(np.round(tempMag[noisyFnInds], 2)) + ' < ' + \
                      str(np.round(magThreshold, 2)) + '. Re-running linear regression.',file=file)
                sys.stdout = console
                file.close()

            # Update functionsToUseArray to cull noisy functions for this variable:
            functionsToUseArray[v, noisyFnInds] = False

            # Zero out coeffs for this variable. The relevant ones will be replaced using the new
            # regressions:
            coeffsBySeg[v, :, :] = -100
            coeffArray[v, :] = 0

            # Pick new segments and regress on each in turn. Use the same 'startIndsToUse' array.
            startIndsVarI = startIndsToUse[v,].copy()  # booleans
            # Convert to indices of timepoints:
            startIndsVarI  = np.where(startIndsVarI == True)[0] + p['margin']
            for seg in range(p['numRegressionSegments']):
                # Pick a subset of startIndsVarI  for this segment:
                if p['useRandomSegmentsFlag']:  # Case: Randomly choose points to regress on.
                    numPointsPerSegment = \
                        int(len(startIndsVarI) / p['numRegressionSegments'] * \
                            (1 + 2 * p['overlapFraction']))
                    startIndIndices = np.sort(np.random.choice(range(len(startIndsVarI)),
                                              numPointsPerSegment, replace=False))
                    startInds = startIndsVarI[startIndIndices]
                else:  # Case: Use sequential segments
                    startInds = startIndsVarI[np.logical_and(startIndsVarI >= segStartInds[seg],
                                                             startIndsVarI <= segEndInds[seg])]

                # Now do regression, if there are startInds in this segment:
                functionInds = np.where(functionsToUseArray[v, :] == True)[0]
                if len(startInds) > 0 and len(functionInds) > 0:  # Case: There are some startInds.
                    # Define the target rise based on the derivative approximation:
                    if p['regressionTarget'] == 'estimated derivatives':  # Currently always true
                        y, weights = \
                            estimateDerivatives_fn(p['xTrain'][:, v], startInds, 
                                                   p['numDtStepsForDeriv'], 
                                                   pointWtsForThisVar, p['dt'])

                        # Extract the relevant functions (columns):
                        X = functionalVals[startInds, :].copy()
                        X = X[:, functionInds]
                    if p['weightTimepointsFlag']:
                        sampleWeights = weights
                    else:
                        sampleWeights = np.ones(weights.shape) / len(weights)  # uniform, sum to 1

                    # Maybe also regress on fft(xDotTrain) as targets. There are many
                    # fewer regression points in the fft target: 50 vs. up to 5000 for the
                    # usual raw derivs, so use the sample weights to balance this out.
                    if p['regressOnFftAlpha'] > 0:
                        rawYAlpha = 1 - p['regressOnFftAlpha']
                        yFftOfXDot = p['fftXDotTrainTarget'][:, v]
                        y = np.hstack((y, yFftOfXDot ))  # Stack the targets
                        XForFft = p['fftLibFns'][:, functionInds].copy()
                        XForFft[np.where(np.isnan(XForFft))] = 0  # since fft of constant == nan
                        X = np.vstack((X, XForFft))
                        
                        numFftPoints = len(yFftOfXDot)
                        sampleWeights = \
                                np.hstack((rawYAlpha * sampleWeights, p['regressOnFftAlpha'] * \
                                           np.ones(numFftPoints) / numFftPoints))

                    # Finally ready to do the regressions on the segment 'seg':
                    # Special case: If we are regressing on complex FFT, we need to use lstsq:
                    if p['fftRegressionTarget'] == 'complex' and p['regressOnFftAlpha'] > 0:
                        w = sampleWeights.reshape(-1, 1)
                        betas = lstsq(np.sqrt(w) * X, np.sqrt(w) * y.reshape(-1, 1), rcond=-1)[0]
                        betas = np.real(betas)  # This may be very important and harmful.
                        coeffsBySeg[v, functionInds, seg] = betas.flatten()
                    else:
                        linReg = LinearRegression()
                        linReg = linReg.fit(X, y, sample_weight=sampleWeights)
                        coeffsBySeg[v, functionInds, seg] = linReg.coef_

            # Regressions have now been done on each segment for variable v.
            # For this variable, process the collections of coeffs on the different
            # segments to get a single set of coeffs.
            prs = coeffsBySeg.copy()
            # We are still in variable 'i'.
            for j in range(len(p['functionList'])):
                if functionsToUseArray[v, j]:  # Case: this function is in use for this
                # var, so we need to calculate a single coeff.
                    coeff, dummy = \
                        combineSegmentCoeffs_fn(prs[v, j, :], False, p['variableNames'][v],
                                                p['functionList'][j], p['snrThreshold'], 
                                                p['minNumSegmentResultsToUse'], 
                                                p['outputFilename'])  
                                                # Output to file, not console
                    coeffArray[v, j] = coeff
                else:  # Case: this functional has been culled for this variable.
                    pass
        # All functionals have new coeffs for this variable i, after 2nd fitting
        # without the functionals culled due to high stdDev / median.

    return coeffArray, functionsToUseArray

# End of cullBasedOnStdOverMedian_fn
#-------------------------------------------------------------------------------

def printWeightedCoeffModel_fn(coeffArray, functionsToUseArray, weightArray, variableNames,
                               functionList, outputFilename):
    """
    Print out a version of the sindy model that shows the weighted coefficients used when culling
    functionals. These weights tend to inflate the coefficients of functionals with relatively high
    values, and decrease the coefficients of functionals with relatively low values, because a
    functional with high values can tolerate a lower coefficient and still have the same impact as
    a functional with low values but a higher coefficient.

    Parameters
    ----------
    coeffArray : np.array, floats numVars x numFunctionals
    functionsToUseArray : np.array of booleans, numVariables x numFunctionals
    weightArray : np.array of floats, numVariables x numFunctionals
    variableNames : list of str
    functionList : list of str
    outputFilename : str

    Returns
    -------
    None.

    """ 
    weightedCoeffArray = weightArray * coeffArray  
    this = printModel_fn(weightedCoeffArray, variableNames, functionList)
    console = sys.stdout
    with open(outputFilename, 'a') as file:
        print('(Weighted coefficients:) \n' + this + '\n', file=file)
        file.close()
        sys.stdout = console 

# End of printWeightedCoefficientModel_fn
#--------------------------------------------------------------------------
 
def evolveModel_fn(coeffArray, recipes, initConds, timepoints):
    
    """
    Evolve odes using solve_ivp.    
    Because odeint can blow up and treat '0' coefficients as non-zero, we add some fuss to prevent
    this in case it can happen with solve_ivp. Method:
    Make a new vector of starting values, with culled variables values set to 0. Use this to
    simulate the system. Then tack on constant values for the culled variables. This approach 
    assumes that if a variable has been zeroed out, then all functionals involving that variable
    have been zeroed.
    
    Parameters:
    ----------
    coeffArray : np.array of floats, numVars x numFunctionals. Coeffs for the model to be evolved.
    recipes : list of lists, length = numFunctionals. The i'th list gives the indices of the 
             variables to be multiplied to make the i'th fnal.
    initConds : np.array of floats, vector 1 x numVars
    timepoints : np.array of floats, vector
    
    Returns:
    -------
    xEvolved : np.array of floats. numVars x len(timepoints)
    
    """
    
    # The function used by solve_ivp:
    def odeFn(t, s, coef, recipes): 
        """
        Return the values of s at the next step:

        Parameters
        ----------
        t : implicit variable
        s : np.array of floats, vector with len = numVars
        coef : np.array of floats, numVars x numFunctionals
        recipes : list of lists, each list may be an int (or np.int64), or a list of ints

        Returns
        -------
        df : np.array of floats, vector with len = numVars

        """
        coef = np.array(coef)
        coef = coef.reshape(len(s), -1)
        fnals = np.zeros(len(recipes)) 
        for i in range(len(fnals)):
            r = recipes[i]
            if isinstance(r, (int, np.integer)):
                if r == -1:  # -1 means constant term
                    fnals[i] = 1 
                else:
                    fnals[i] = s[r]
            else:
                temp = 1
                for j in range(len(r)):
                    temp = temp * s[r[j]]
                fnals[i] = temp
        
        df = np.zeros(s.shape)
        for i in range(len(df)):
            df[i] = np.dot(fnals, coef[i, ])
         
        return df
     
    method =  'LSODA'  # 'RK45'. For stiff equations: ‘Radau’ or ‘BDF’
    startEnd = [timepoints[0], timepoints[-1]]
    argList = []
    argList.append(coeffArray)
    argList.append(recipes)
    
    # Fussing to prevent possible blow-ups of zeroed-out variables:
    # First remove any variables that have all zero coefficients:
    temp = np.abs(coeffArray) > 1e-5
    zeroedOutVars = np.where(np.sum(temp, axis=1) == 0)[0]  # The culled variables. 

    # Make a set of starting data with zeros in the culled variables: 
    initCondsCulledVarsZeroed = initConds.copy() 
    if len(zeroedOutVars) > 0:
        initCondsCulledVarsZeroed[zeroedOutVars] = 0

    xEvolved = np.zeros((len(timepoints), len(initConds)))  # Initialize
        
    sol = solve_ivp(odeFn, t_span=startEnd, y0=initCondsCulledVarsZeroed, method=method, 
                    t_eval=timepoints, args=argList)

    if sol.success:
        xEvolved = (sol.y).transpose()
    else:
        print('Error notice: solve_ivp failed.')
    
    # Fill in the constant vars:
    if len(zeroedOutVars) > 0:
        xEvolved[:, zeroedOutVars] = np.tile(initConds[zeroedOutVars], (len(timepoints), 1))
        
    return xEvolved  # could also return sol.t

# End of evolveModel_fn
#-------------------------------------------------------------------------

def generateVariableEnvelopes_fn(xData, numInWindow, dt, shrinkFactor=1):
    """
    For each point in a time-series, find the max and min values in a neighborhood for each
    variable.

    Parameters
    ----------
    xData : np.array of floats, numTimepoints x numVariables
    numInWindow : scalar int
    dt : scalar float
    shrinkFactor : scalar float >= 1

    Returns
    -------
    varLocalMax: np.array of floats, size xData.shape
    varLocalMin: np.array of floats, size xData.shape

    """
    # If required, shrink data towards local mean values:
    if shrinkFactor > 1:
        for i in range(xData.shape[1]):
            slopeX, stdX, meanX = calculateSlopeAndStd_fn(xData[:, i], dt, numInWindow)
            xData[:, i] = (xData[:, i] - meanX) / shrinkFactor + meanX

    half = int(np.floor(numInWindow / 2))

    varLocalMax = np.zeros(xData.shape)
    varLocalMin = np.zeros(xData.shape)

    for i in range(xData.shape[0]):  # loop over timepoints in this variable
        startInd = max(0, i - half)
        endInd = min(xData.shape[0], i + half)
        varLocalMax[i, :] = np.max(xData[startInd:endInd, :], axis=0)
        varLocalMin[i, :] = np.min(xData[startInd:endInd, :], axis=0)

    return varLocalMax, varLocalMin

# End of generateVariableEnvelopes_fn
#------------------------------------------------------------------------

def calculateFiguresOfMerit_fn(x, xTrue, fftPowerTrue, localMin, localMax, stdDevVector, 
                               meanVector, medianVector, fomTimepoints, maxPhaseShift):
    """
    Evolve a model and calculate various figures of merit.
    NOTE: We assume that xEvolved includes only timepoints in 'fomTimepointInds'. For 
    'inEnvelopeFoM' we take the max over small phase shifts of the whole time-series.

    Parameters
    ----------
    x : np.array of floats, numTimepoints x numVariables
    xTrue : np.array of floats, numTimepoints x numVariables
    fftPowerTrue :  np.array of floats, numFftPoints x numVariables 
    localMin : np.array of floats, numTimepoints x numVariables
    localMax : np.array of floats, numTimepoints x numVariables
    stdDevVector : np.array of floats, numTimepoints x numVariables
    meanVector : np.array of floats, numTimepoints x numVariables
    medianVector : np.array of floats, numTimepoints x numVariables
    fomTimepoints : np.array (list?) of ints
    maxPhaseShift : scalar int. Must be >= 1

    Returns
    -------
    fomDict: dict

    """
    if len(x.shape) == 1:  # Case: 1-dim with shape (val, ) rather than (val, 1)
        x = np.expand_dims(x, 1)

    phaseShifts= np.array(range(-maxPhaseShift, maxPhaseShift, 3))  # hop by 3s for speed
    first = maxPhaseShift # Used to prevent overshooting.
    final = x.shape[0] - maxPhaseShift # Ditto.

    fomDict = dict()
    inBoundsFoM = np.zeros(x.shape[1])
    inEnvelopeFoM = np.zeros(x.shape[1])
    inEnvPhaseShiftVals = np.zeros(len(phaseShifts))
    globalMin = np.min(localMin, axis=0)
    globalMax = np.max(localMax, axis=0)
    for i in range(x.shape[1]):
        inBoundsFoM[i] = \
            np.sum(np.logical_and(x[:, i] > globalMin[i], 
                                  x[:, i] < globalMax[i])) / len(fomTimepoints)
        for j in range(len(phaseShifts)):
            pS = phaseShifts[j]
            inEnvPhaseShiftVals[j] = \
                np.sum(np.logical_and(x[first + pS:final + pS, i] > \
                                         localMin[first:final, i],
                                         x[first + pS:final + pS, i] < \
                                             localMax[first:final, i])) / len(fomTimepoints)
        inEnvelopeFoM[i] = max(inEnvPhaseShiftVals)

    # Statistics:
    temp = np.std(x, axis=0)
    stdDevFoM = (temp - stdDevVector) / stdDevVector
    stdDevFoM[stdDevFoM > 100] = 100 # To avoid uselessly big numbers
    temp = np.mean(x, axis=0)
    meanFoM = (temp - meanVector) / meanVector
    meanFoM[meanFoM > 100] = 100 # To avoid uselessly big numbers
    meanFoM[meanFoM < -100] = -100 # To avoid uselessly big numbers
    temp = np.median(x, axis=0)
    medianFoM = (temp - medianVector) / medianVector
    medianFoM[medianFoM > 100] = 100 # To avoid uselessly big numbers
    medianFoM[medianFoM < -100] = -100 # To avoid uselessly big numbers

    # Correlation of FFT:
    numFftPoints = fftPowerTrue.shape[0]
    fftPower = np.zeros([numFftPoints, x.shape[1]])
    fftCorrelationFoM = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        x1 = x[:, i].copy()
        x1 = x1 - np.mean(x1)
        xP = pow(np.real(np.fft.fft(x1)), 2) # power spectrum of fft
        temp = xP[0:numFftPoints] / np.sum(xP[0:numFftPoints])
        temp[np.isnan(temp)] = 0
        fftPower[:, i] = temp
        # Handle case that a var is identically 0:
        if not np.isnan(np.sum(fftPowerTrue[:, i])):
            fftCorrelationFoM[i] = np.dot(fftPower[:, i], fftPowerTrue[:, i]) / \
                np.dot(fftPowerTrue[:, i],  fftPowerTrue[:, i])  # Normalize
        else:
            fftCorrelationFoM[i] = 0

    # Correlation of histograms:
    numHistogramBins = 100
    histogramRange = np.array((np.min(xTrue, axis=0), np.max(xTrue, axis=0)))  # 2 x numVars
    theseHistograms = np.zeros([numHistogramBins, x.shape[1]])
    theseHistogramBins = theseHistograms.copy()
    histogramCorrelationFoM = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        histTrue = np.histogram(xTrue[:, i], bins=numHistogramBins)[0]
        temp = np.histogram(x[:, i], bins=numHistogramBins,
                            range=(histogramRange[0, i], histogramRange[1, i]))
        theseHistograms[:, i] = temp[0]
        theseHistogramBins[:, i] = temp[1][0:-1]
        histogramCorrelationFoM[i] = np.dot(histTrue, theseHistograms[:, i]) / \
            np.dot(histTrue, histTrue)

    # # Print to console:
    # print('inBoundsFoM = ' + str(np.round(inBoundsFoM,2)) + \
    #       ', inEnvelopeFoM = ' + str(np.round(inEnvelopeFoM, 2)) + \
    #       ', stdDevFoM = ' + str(np.round(stdDevFoM, 2)) + \
    #       ', meanFoM = ' + str(np.round(meanFoM, 2)) + \
    #       ', medianFoM = ' + str(np.round(medianFoM, 2)) + \
    #       ', fftCorrelationFoM = ' + str(np.round(fftCorrelationFoM, 2)) + \
    #       ', histogramCorrelationFoM = ' + str(np.round(histogramCorrelationFoM, 2)) + '\n')

    # Bundle FoMs into dict for return:
    fomDict['inBoundsFoM'] = inBoundsFoM
    fomDict['inEnvelopeFoM'] = inEnvelopeFoM
    fomDict['stdDevFoM'] = stdDevFoM
    fomDict['meanFoM'] = meanFoM
    fomDict['medianFoM'] = medianFoM
    fomDict['fftCorrelationFoM'] = fftCorrelationFoM
    fomDict['fftPower'] = fftPower
    fomDict['histogramCorrelationFoM'] = histogramCorrelationFoM
    fomDict['histograms'] = theseHistograms
    fomDict['histogramBins'] = theseHistogramBins

    return fomDict

# End of calculateFiguresOfMerit_fn
#--------------------------------------------------------------

def findSpansOfFunctionals_fn(fnArray, fnVals, fnValsNoisy, wts, includeConstantFlag, p):
    """
    Find which functionals are in the linear span of the final selected functionals (per variable).
    Return R-squared values for each {functional, variable} pair (each variable has a few 
    selected functionals).

    Parameters
    ----------
    fnArray : np.array of booleans, numVars x numFunctionals in library. Shows which functionals
              were used for each variable.
    fnVals : np.array, numTimepoints (used) x numFunctionals. Gives values of functionals at 
             timepoints, using the smoothed time-series
    fnValsNoisy : np.array, numTimepoints (used) x numFunctionals. Gives values of functionals at 
             timepoints, but using the post-added noise, pre-smoothing time-series. 
    wts : np.array, numTimepoints (used) x numVars. Gives weights of timepoints 
    includeConstantFlag : bool 
    p : dict of miscellaneous parameters, including:
        functionList : list of str
        variableNames : list of str
        outputFilename : str
        plottingThreshold : float
        windowLengthForFomEnvelopes : int
        dt : float
        margin : int
        maxFnValRatio : float
        minNumStartIndsToUse : int

    Returns
    -------
    rSq : np.array of floats, size fnArray.shape. Gives the R-squared from linear fits of
                each functional using just the selected functionals (per variable).
    betaZeros : np.array of floats, size fnArray.shape. Gives .intercept_ of each fit
    betas : np.array of lists, array has size fnArray.shape. Gives the .coef_ of each fit

    """

    betaZeros = np.zeros(fnArray.shape)  # to save the intercepts
    betas = np.empty(fnArray.shape, dtype=object)  # to save the coefficients
    rSq = np.zeros(fnArray.shape)  # to save R-squared values
    lrSp = LinearRegression(fit_intercept=includeConstantFlag)
    for v in range(fnArray.shape[0]):
        if np.sum(fnArray[v, :]) == 0:
            console = sys.stdout
            with open(p['outputFilename'], 'a') as file:
                print(p['variableNames'][v] + ' has no functionals left in sparse library.', 
                      file=file)
                sys.stdout = console
                file.close()
        else:  # case: there are functionals in this variable's sparse library
            basis = np.where(fnArray[v, :])[0]
            basisStr = str(np.array(p['functionList'])[basis]).replace('[','').replace(']','')
            X = fnVals[:, fnArray[v, :]]  # The values of the selected functionals
            for i in range(fnArray.shape[1]):  # num functionals
                if fnArray[v, i]:
                    rSq[v, i] = 1
                else:
                    y = fnVals[:, i]
                    # Exclude any timepoints from the regression with very large differences in
                    # functional values used as features: 
                    miniFnArray = np.ones((1, X.shape[1]), dtype=bool)
                    removeMarginFlag = False # since for FoM timepoint assessment, we don't ignore 
                    # a margin at each end.
                    nonConstantIndices = np.where(np.array(p['functionList']) != '1')[0]
                    goodInds = \
                        parseTimepointsByMagnitudesOfVariables_fn(X.copy(), miniFnArray, 
                                                                  nonConstantIndices, p['margin'], 
                                                                  p['maxFnValRatio'], 
                                                                  p['minNumStartIndsToUse'],
                                                                  removeMarginFlag)
                    # goodInds is 1 x numTimepoints vector
                    theseWts = wts[:, v].copy() 
                    theseWts[np.logical_not(goodInds.flatten())] = 0 
                    # theseWts = theseWts / np.sum(goodInds)  # normalize the sum
                    lrSp = lrSp.fit(X, y, sample_weight=theseWts)
                    rSq[v, i] = lrSp.score(X, y, sample_weight=theseWts)
                    betaZeros[v, i] = lrSp.intercept_
                    betas[v, i] = lrSp.coef_
                    # plot functionals that are somewhat well-fit:
                    showNoiseEnvelopesFlag = True
                    if rSq[v, i] > p['plottingThreshold'] and rSq[v, i] < 1:  
                        # < 1 to ignore basis functionals
                        yHat = lrSp.predict(X)
                        plt.figure()
                        plt.title('var = ' + p['variableNames'][v] + ', basis = ' + \
                                  basisStr + '\n' + 'functional = ' + \
                                  p['functionList'][i] + ', R_sq = ' + str(np.round(rSq[v, i], 2)),
                                  fontsize=14, fontweight='bold')
                        plt.scatter(np.array(range(len(y))), y, s=3, c='k', 
                                    label='functional values')
                        plt.scatter(np.array(range(len(y))), yHat, s=3, c='r', 
                                    label='fit values')
                        if showNoiseEnvelopesFlag:
                            slopeY, stdY, meanY = \
                                calculateSlopeAndStd_fn(fnValsNoisy[:, i], p['dt'],
                                                        p['windowLengthForFomEnvelopes'])
                            plt.plot(meanY + 2 * stdY, 'g', label='two std dev of noisy data')
                            plt.plot(meanY - 2 * stdY, 'g')
                        plt.xlabel('FoM timepoints', fontsize=14, fontweight='bold')
                        plt.ylabel('functional value', fontsize=14, fontweight='bold')
                        plt.legend(fontsize=14)

    return rSq, betaZeros, betas

# End of findSpansOfFunctionals_fn
#---------------------------------------------------------------------

def findSpansOfFunctionalsLeaveOneOut_fn(fnArray, fnVals, fnValsNoisy, wts, includeConstantFlag, 
                                         p):
    """
    Considering the retained functionals (per variable), find which functionals are in the linear 
    span of the other functionals (leave-one-out).
    Return rR-squared values for each {functional, variable} pair (each variable has a few 
    selected functionals).

    Parameters
    ----------
    fnArray : np.array of booleans, numVars x numFunctionals in library. Shows which functionals
              are active for each variable.
    fnVals : np.array, numTimepoints (used) x numFunctionals. Gives values of functionals at 
             timepoints, using the post-smoothed time-series.
    fnValsNoisy : np.array, numTimepoints (used) x numFunctionals. Gives values of functionals at 
             timepoints, using the post-added noise and pre-smoothing time-series. 
    wts : np.array, numTimepoints (used) x numVars. Gives weights of timepoints 
    includeConstantFlag : bool 
    p : dict of miscellaneous parameters, including:
        functionList : list of str
        variableNames : list of str
        outputFilename : str
        windowLengthForFomEnvelopes : int
        dt : float
        margin : int
        maxFnValRatio : float
        minNumStartIndsToUse : int
        plottingThreshold : float

    Returns
    -------
    rSq : np.array of floats, size fnArray.shape. Gives the R-squared from linear fits of
                each functional using just the selected functionals (per variable).
    betaZeros : np.array of floats, size fnArray.shape. The {i,j}th entry is the intercept for the 
                fit of the j'th functional using the active functionals, for the i'th variable.
    betas : np.array of objects, size fnArray.shape. The {i,j}th object is the vector of beta 
            coefficients found by fitting the j'th functional using the active functionals, for 
            the i'th variable. If the j'th functional is not active, the object = []. Else its
            length = # active functionals for the variable.

    """  

    betaZeros = np.zeros(fnArray.shape)  # to save .intercept_
    betas = np.empty(fnArray.shape, dtype=object)  # to save .coef_
    rSq = np.zeros(fnArray.shape)
    lrSp = LinearRegression(fit_intercept=includeConstantFlag)
    for v in range(fnArray.shape[0]):  # loop over numVars
        if np.sum(fnArray[v, :]) <= 1:
            console = sys.stdout
            with open(p['outputFilename'], 'a') as file:
                print(p['variableNames'][v] + ' has <= 1 functional left in library.', file=file)
                sys.stdout = console
                file.close()
        else:  # case: there are functionals in this variable's sparse library 
            inds = np.where(fnArray[v, :]==True)[0]
            for i in inds:  # num functionals 
                y = fnVals[:, i]
                others =  inds[inds != i].copy() 
                basisStr = str(np.array(p['functionList'])[others]).replace('[','').replace(']','')
                X = fnVals[:, others].copy()  # the other retained functionals
                # Exclude any timepoints from the regression with very large differences in
                # functional values used as features: 
                miniFnArray = np.ones((1, X.shape[1]))
                removeMarginFlag = False # since for FoM timepoint assessment, we don't ignore a
                # margin at each end.
                nonConstantIndices = np.where(np.array(p['functionList'] != '1'))[0]
                goodInds = \
                    parseTimepointsByMagnitudesOfVariables_fn(X.copy(), miniFnArray, 
                                                              nonConstantIndices, 
                                                              p['margin'], p['maxFnValRatio'], 
                                                              p['minNumStartIndsToUse'],
                                                              removeMarginFlag)
                # goodInds is 1 x numTimepoints vector
                theseWts = wts[:, v].copy()  
                theseWts[np.logical_not(goodInds.flatten())] = 0 
                theseWts = theseWts / np.sum(theseWts) # normalize back to sum = 1
                lrSp = lrSp.fit(X, y, sample_weight=theseWts)
                rSq[v, i] = lrSp.score(X, y, sample_weight=theseWts)
                betaZeros[v, i] = lrSp.intercept_
                betas[v, i] = lrSp.coef_
                # plot functionals that are somewhat well-fit:
                showNoiseEnvelopesFlag = True
                if rSq[v, i] > p['plottingThreshold'] and rSq[v, i] < 1:  # < 1 to ignore basis 
                # functionals, which have rSq = 1. 
                    yHat = lrSp.predict(X)
                    plt.figure()
                    plt.title('var = ' + p['variableNames'][v] + ', sub-basis = ' + \
                              basisStr + '\n' + 'left-out functional = ' + \
                              p['functionList'][i] + ', R_sq = ' + str(np.round(rSq[v, i], 2)),
                              fontsize=14, fontweight='bold')
                    plt.scatter(np.array(range(len(y))), y, s=3, c='k', 
                                label='functional values')
                    plt.scatter(np.array(range(len(y))), yHat, s=3, c='r', 
                                label='fit values')
                    if showNoiseEnvelopesFlag:
                        slopeY, stdY, meanY = \
                            calculateSlopeAndStd_fn(fnValsNoisy[:, i], p['dt'],
                                                    p['windowLengthForFomEnvelopes'])
                        plt.plot(meanY + 2 * stdY, 'g', label='two std dev of noisy data')
                        plt.plot(meanY - 2 * stdY, 'g')
                    plt.xlabel('FoM timepoints', fontsize=14, fontweight='bold')
                    plt.ylabel('functional value', fontsize=14, fontweight='bold')
                    plt.legend(fontsize=14)

    return rSq, betaZeros, betas

# End of findSpansOfFunctionalsLeaveOneOut_fn
#----------------------------------------------------------

def findClosestLinearlyEquivalentVersion_fn(nC, tC, rSqLooV, betasLooV, params):
    """
    Given a set of coefficients (for one variable), use linear combinations of true 
    functionals, whose leave-one-out linear fits have sufficiently high Rsquared values, to 
    iteratively substitute to minimize the maximum absolute error in coefficients of true 
    functionals relative to the true system coefficients. 
    We only substitute using linear combinations among true functionals. 
    Note that any functional in the discovered model that is not part of the true functionals has
    no error measure.
    In this function, the inputs are for one system variable (e.g. 'x', 'y').

    Parameters
    ----------
    nCR : np.array (vector) of floats. The model coefficients 
    tCR : np.array (vector) of floats. The true system coefficients 
    rSqLooV : np.array (vector) of floats. The Rsquared values for leave-one-out linear fits
    betasLooV : np.array of objects. numVariables x numLibraryFunctionals. The {i,j}th object is 
                the vector of beta coefficients found by fitting the j'th functional using the 
                active functionals, for the i'th variable. If the j'th functional is not active, 
                the object = []. Else its length = # active functionals for the variable.
    params : dict of miscellaneous parameters

    Returns
    -------
    nC : np.array (vector) of floats. These are coefficients for a model that is linearly 
         equivalent to the argin model 'nC' and with a minimal maximum error relative to true 
         functional coefficients.
    coeffErrR : np.array (vector) of floats. The final percentage absolute coefficient errors of
                each true library functional. length = number of true functionals, ordered 
                according to how the functionals are listed in params['functionList'].

    """
    # Method: Iteratively, do a transform that reduces the biggest error. 
    tF = tC.astype(bool)
    nCR = nC[tF]  # the 'R' means Restricted, to just true functionals. We modify these only.
    tCR = tC[tF]  # ditto 
    numActiveFnals = np.sum(nCR != 0)  # will be used to detect resurrected fnals.
    resurrectedTrueFnalFlag = False
    # We also need the full index of each fnal:
    tC2 = tC.copy()
    tC2[tC2 == 0] = 1  # since this will be in a denominator in loop below
    # initialize for while loop:
    coeffErrR = np.abs((nCR - tCR) / tCR)
    newErr = max(coeffErrR[np.abs(nCR) > 0]) # don't consider error in missing fnals, 
    # which have error = -100% 
    oldErr = 2 * newErr 
    while newErr < oldErr * params['tolRatio'] or \
        (newErr > oldErr and resurrectedTrueFnalFlag):  # ie most recent change was 
    # non-trivial. The first condition is expected after a well-behaved iteration. The
    # second condition happens if an absent true fnal gets resurrected.
        previousNCR = nCR.copy()  # In case we overshoot, and want to revert
        if params['verbose']:
            print('newErr = ' + str(np.round(newErr, 2)))
        oldErr = newErr
        # Find an in-span fit equation and update the coefficients nC with it. Note that although
        # argin0 is the full list of coefficients, we only substitute among true functionals 
        # because argin2 and argin3 only contain linear fits among true functionals.
        nCR = updateLibCoeffsViaInSpanLoo_fn(nC, tC, rSqLooV, betasLooV, params['functionList'], 
                                             params['rSqThresh'], params['stepSize'], 
                                             params['verbose'])
          
        # Update nC etc:
        newErr = max(np.abs((nCR - tCR) / tCR))
        resurrectedTrueFnalFlag = np.sum(nCR != 0) > numActiveFnals
        numActiveFnals = np.sum(nCR != 0) 
        if newErr > oldErr and not resurrectedTrueFnalFlag: # case: undo this iteration
            if params['verbose']:
                print('newErr = ' + str(np.round(newErr, 2)) + ', undoing this iteration.')
            nCR = previousNCR
            newErr = oldErr 
        nC[tF] = nCR
    # End of while loop
    
    # Coefficient errors on true library functionals
    coeffErrR = np.abs((nCR - tCR) / tCR) 
            
    return nC, coeffErrR  

# End of findClosestLinearlyEquivalentVersion_fn 
#---------------------------------------------------------------------
 
def evolveModelAndCalcFoms_fn(coeffArray, recipes, initCond, timepoints, x, fftPow, 
                                        varLocMin, varLocMax, phase):
    """ For parallel processing."""
    
    print('starting evolution...\n')
    xTrainEvolved = evolveModel_fn(coeffArray, recipes, initCond, timepoints)
    print('done with evolution.\n')
    # xTrain is used for xDot predictions.
    # Calculate and print the FoMs. We only use the histograms, to compare the evolutions:
    fomDict = calculateFiguresOfMerit_fn(xTrainEvolved, x, fftPow, varLocMin, varLocMax, 
                                         timepoints, phase)
     
    return fomDict['histograms']
    #  np.array([[99,99,99],[99,99,99],[99,99,99]]) # 

# End of evolveModelAndCalcFoms_fn
#-------------------------------------------------------------------

def appendToLocalHistograms_fn(localHistograms, result):
    """ Helper to give a usable parallel processing result"""
    localHistograms.append(result)
 
# End of appendToLocalHistograms_fn
#-----------------------------------------------------------------------

def printModel_fn(coeffArray, variableNames, functionList, precision=2):
    """ Return a printable string version of a linear model.    

    Parameters
    ----------
    coeffArray : np.array of floats, numVars x numFunctionals 
    variableNames : list of str
    functionList : list of str
    precision : int, optional, default = 5.

    Returns
    -------
    mS : str (multi-line string version of model)

    """
    numFns = len(functionList)
    mS = ''
    for v in range(len(variableNames)):
        mS = mS + variableNames[v] + "' = "
        for j in range(numFns):
            if np.abs(coeffArray[v, j]) > 1e-5:
                signTag = ' + '
                if coeffArray[v, j] < 0:
                    signTag = ' - '
                mS = mS + signTag + str(np.round(np.abs(coeffArray[v, j]), precision)) + \
                    ' ' +  functionList[j] + ' '
        mS = mS + '\n'
    
    return mS
# End of printModel_fn
#-------------------------------------------------------------------
           
def updateLibCoeffsViaInSpanLoo_fn(nC, tC, rSqLooRow, betasLooRow, functionList, rSqThresh=0.95, 
                             stepSize=0.25, verbose=False):
    """
    An inner while loop, for one variable: given (i) a set of current coeffs of true library
    functionals; (ii) info about the true leave-one-out betas; and (iii) misc parameters, find a 
    functional with high Rsquared and also high error, and use its betas to make an incremental 
    substition into the coefficients equation. Go through the functionals via: first choose the 
    one with highest error. If its Rsquared from leave-one-out fit is high, use its fit equation
    eg x1 = b2*x2 + b3*x3 -> 0 = -x1 + b2*x2 + b3*x3, to modify the coefficients. If the Rsquared
    is too low, go to the functional with the next-highest error.
    
    Parameters
    ----------
    nC : np.array of floats, vector with len = numFnals. The current coefficients
    tC : ditto. The true coefficients 
    rSqLooRow : np.array of floats, vector with len = numFnals.
    betasLooRow : list (len = numFnals) of lists. Each inner list contains floats (len = number of 
               true library functionals)
    functionList : list of str, len = numFnals
    rSqThresh : float, default 0.95
    stepSize : float, default 0.25
    verbose : bool
    
    Returns
    -------
    nCR : np.array of floats, vector with length = number of true library functionals
    
    """
    indsRTried = []
    indsFTried = []
    tF = tC.astype(bool)
    nCR = nC[tF]  # the 'R' means Restricted, to just true fnals
    tCR = tC[tF]  # ditto 
    tC2 = tC.copy()
    tC2[tC2 == 0] = 1
    while len(indsRTried) < len(nCR):
        coeffErrR = np.abs((nCR - tCR) / tCR)  # Restricted to just true fnals
        coeffErrF = np.abs((nC - tC2 * tF) / tC2)  # 'F' means Full, ie over all fnals.
        # Remove already-tried inds from consideration:
        coeffErrR[indsRTried] = 0
        coeffErrF[indsFTried] = 0
        newErr = max(np.abs(coeffErrR))
        indR = np.where(coeffErrR == newErr)[0][0]  # ind with biggest error
        indF = np.where(coeffErrF == newErr)[0][0]  # we need this to get betasLoo
        # See if the loo fit has high enough rSq:
        if rSqLooRow[indF] > rSqThresh:
            if verbose:
                print('Using loo fit for ' + functionList[indF])
            b = betasLooRow[indF]  # the betas for fitting the left-out fnal using other 
            # fnals.
            # Let b equal "x1 = b2*x2 + b3*x3". Then we'll add some factor times the eqn
            # 0 = -1*x1 + b2*x2 + b3*x3 to the coeffs nCR
            factor = stepSize * (nCR[indR] - tCR[indR])  
            remainingFnalsCounter = 0
            for i in range(len(nCR)):
                if i == indR:
                    nCR[indR] = nCR[indR] + factor * (-1) 
                else:
                    nCR[i] = nCR[i] + factor * b[remainingFnalsCounter]
                    remainingFnalsCounter += 1 
            indsRTried = np.arange(len(nCR)) # to exit the inner while loop
        else:
            if verbose:
                print('True fnal ' + functionList[indF] + \
                      ' had highest error, but its Loo Rsq is too low  ' + \
                          '(' + str(np.round(rSqLooRow[indF], 2)) + ')')
            indsRTried.append(indR)
            indsFTried.append(indF) 
        
    return nCR 
 
# End of updateLibCoeffsViaInSpanLoo_fn
#---------------------------------------------------

def replaceNonLibraryFunctionals_fn(dC, tF, rSqRow, betasRow, functionList, rSqThresh, 
                                    verbose=False):
    """
    Given a set of coefficients and functionals, replace functionals that are not in the 'true'
    library using the in-span fit equation. Only do this if the Rsquared of the fit was high
    enough.

    Parameters
    ----------
    dC : np.array of floats, vector len = numFnals. Discovered coeffs for the variable at hand 
    tF : np.array of bools, vector len = numFnals. The true library functionals for the variable  
    rSqRow : np.array of floats, vector with len = numFnals.
    betasRow : list (len = numFnals) of lists. Each inner list contains floats (len = number of 
               true library functionals)
    functionList : list of str, len = numFnals
    rSqThresh : scalar float
    verbose : bool, default = False

    Returns
    -------
    nCLocal : np.array of floats, vector len = numFnals.

    """
    dF = dC.astype(bool)
    nCLocal = dC.copy()  # To store new (transformed coefficients)
    counter = 0
    for i in range(len(dF)):
        if dF[i] and not tF[i]:
            counter += 1
            if rSqRow[i] > rSqThresh:
                if verbose:
                    print('Replacing untrue fnal ' + functionList[i] + ' (Rsq = ' + \
                           str(np.round(rSqRow[i], 2)) + ')')
                b = betasRow[i]  # betasRow[i] are the coeffs in {fnal i = lin combo of true 
                # fnals}
                nCLocal[tF] = nCLocal[tF] + nCLocal[i] * b  # the [tF] ensures the correct cols get
                # updated.
                nCLocal[i] = 0  # since we subtracted out this term
            else:
                if verbose:
                    print('Untrue fnal ' + functionList[i] + ' was in the discovered model ' + \
                          'but is not in-span of true fnals (Rsq = ' + \
                          str(np.round(rSqRow[i], 2)) + ')')
    if counter == 0:
        if verbose:
            print('No untrue fnals for this variable')
    
    return nCLocal

# End of replaceNonLibraryFunctionals_fn
#-------------------------------------------------------------- 

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
       
