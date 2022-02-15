#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper function to plot various time-series associated with a particular model(s).
Original data, noisy data, smoothed data, model predictions, xDot, xDot predictions, training
trajectories, validation trajectories, test trajectories. Time-series and 3-D.

Called by 'plotSelectedIterations.py'.

For the full procedure, see "README.md".
 
For method details, please see "A toolkit for data-driven discovery of governing equations in 
high-noise regimes" (2022) by C.B. Delahunt and J.N. Kutz.

For these plots to appear in separate windows, run in the python console:
%matplotlib qt
         
Copyright (c) 2021 Charles B. Delahunt.  delahunt@uw.edu
MIT License
"""
import numpy as np
from matplotlib import pyplot as plt


def plotVariousTimeseries_fn(showPlotsFlag, d, iterationsToUse=[]):
    """
    Plots of trajectories: raw, evolved, xDot, etc. Not all of these plots are always useful.
    Comment: during plotting, the range of timepoints plotted for a given time-series depends on
    whether the time-series starts at timepoint 0 and ends at -1, or if it starts at 'margin' and
    ends at -'margin'.

    Parameters
    ----------
    showPlotsFlag : list with entries 0 or 1. Controls which plots are drawn (1 -> yes, 0 -> no)
    d : large dict with all the relevant time-series, parameters, and other data.
    iterationsToUse : list of ints. Only used as argin by

    Returns
    -------
    None. Generates plots

    """
    # Unpack d, the dataDict:
    systemInfoStr = d['systemInfoStr'] 
    iterationNumbersToPlot = d['iterationNumbersToPlot']
    
    dt = d['dt']
    marginInSecs = d['marginInSecs']
    numSecsInTrain = d['numSecsInTrain']
    numSecsInTest = d['numSecsInTest']
    numTrajTrain = d['numTrajTrain']
    variableNames = d['variableNames']
    numVars = len(variableNames)
    numTrajTest = d['numTrajTest']

    tTrain = d['tTrain']
    xTrainNoisyOriginalAll = d['xTrainNoisyOriginalAll']
    xTrainCleanAll = d['xTrainCleanAll']
    xTrainAll = d['xTrainAll']
    xTrainEvolvedAll = d['xTrainEvolvedAll']
    xDotTrainAll = d['xDotTrainAll']
    xDotTrainTrueAll = d['xDotTrainTrueAll']
    xDotTrainPredictedAll = d['xDotTrainPredictedAll']
    #xDotTrainSmoothedAll = d['xDotTrainSmoothedAll']
    #xDotTrainUnsmoothedAll = d['xDotTrainUnsmoothedAll']

    xValEvolvedAll = d['xValEvolvedAll']
    indsValAll = d['indsValAll']
    # Note that xValCleanAll etc are contained in xTrainCleanAll, since val trajectories for a
    # given model are just the training trajectories that it was not trained on.

    tTest = d['tTest']
    xTestAll = d['xTestAll']
    xTestEvolvedAll = d['xTestEvolvedAll']
    xDotTestTrueAll = d['xDotTestTrueAll']
    xDotTestPredictedAll = d['xDotTestPredictedAll']

    # For histograms of FFT and time-series values:
    historyFftPowerAll = d['historyFftPowerAll']
    xTrainFftPowerAll = d['xTrainFftPowerAll']
    historyHistogramsAll = d['historyHistogramsAll']
    historyHistogramBinsAll = d['historyHistogramBinsAll']
    xTrainHistogramAll = d['xTrainHistogramAll']
    xTrainHistogramBinsAll = d['xTrainHistogramBinsAll']
        
    historyXDotHistogramsAll = d['historyXDotHistogramsAll']
    historyXDotHistogramBinsAll = d['historyXDotHistogramBinsAll']
    xDotTrainHistogramAll = d['xDotTrainHistogramAll']
    xDotTrainHistogramBinsAll = d['xDotTrainHistogramBinsAll']
    
    # Odds and ends:
    colorsForTraj = ['orchid', 'orchid', 'orchid', 'orchid', 'orchid', 'orchid']  # This can be
    # reset to give different colors for each indexed trajectory, if wished.
    tag = ''  # For legend labels
    # if d['smoothInitialDerivsFlag']:
    #     tag = ' (smoothed)'

    #%% Remind ourselves of the target system to console:
    print(systemInfoStr)

    #%% 0. Plot original full Training set:

    if showPlotsFlag[0]:
        startTime = marginInSecs
        stopTime = numSecsInTrain - marginInSecs
        startInd = int(startTime/dt)
        stopInd = int(stopTime/dt)
        inds = np.array(range(startInd, stopInd, 2))

        fig, axs = plt.subplots(numVars, numTrajTrain, sharex='col', figsize=(7,9))
        for j in range(numTrajTrain):
            xTrainNoisyOriginal = xTrainNoisyOriginalAll[j]
            yLimits = (1.1 * np.percentile(xTrainNoisyOriginal.flatten(), 1),  # hope it's negative
                       1.1 * np.percentile(xTrainNoisyOriginal.flatten(), 99.9))            
            xTrainClean = xTrainCleanAll[j]
            for i in range(xTrainNoisyOriginal.shape[1]): 
                axs[i, j].plot(tTrain[inds], xTrainClean[inds,i], 'b', linewidth = 1,
                                    label='clean')
                axs[i, j].plot(tTrain[inds], xTrainNoisyOriginal[inds,i], '.', color='gray',
                               label='noisy')
                # Repeat a line to get legend and layering in the best order:
                axs[i, j].plot(tTrain[inds], xTrainClean[inds,i], 'b', linewidth = 1)
                axs[i, j].set_xlabel('Time', fontweight='bold', fontsize=12) 
                axs[i, j].set_ylim(yLimits)
                if i == 0:
                    axs[i, j].legend()
                if i == numVars - 1:
                    axs[i, j].set_ylabel(variableNames[i], fontweight='bold', fontsize=12)
                
            axs[0, j].set_title('Train traj ' + str(j) + ', clean = blue',
                                fontweight='bold', fontsize=12)
        plt.tight_layout()
        fig.show()

    #%% 1. Plot learned model's derivatives on Train:

    if showPlotsFlag[1]:
        startTime = marginInSecs
        stopTime = numSecsInTrain - marginInSecs
        startInd = int(startTime/dt)
        stopInd = int(stopTime/dt)
        inds = np.array(range(startInd, stopInd, 2))  # note the skip = 2. Must include this skip 
        # for other time-series also, below.

        fig, axs = plt.subplots(numVars, numTrajTrain, sharex='col', sharey='col', figsize=(7,9))
        
        for j in range(numTrajTrain):
            xDotTrain = xDotTrainAll[j]
            xDotTrainClean = xDotTrainTrueAll[j]
            xDotTrainPredicted = xDotTrainPredictedAll[j]
            # xDotTrainSmoothed = xDotTrainSmoothedAll[j]
            # xDotTrainUnsmoothed = xDotTrainUnsmoothedAll[j] 
            yLimits = (1.1 * np.percentile(xDotTrain.flatten(), 1),
                       1.1 * np.percentile(xDotTrain.flatten(), 99.9))
            c = colorsForTraj[j]
            
            for i in range(numVars):
                # axs[i, j].plot(tTrain[inds], xDotTrainUnsmoothed[inds,i], 'k.', markersize = 1,
                #                label='unsmoothed deriv of smoothed data' + tag)
                axs[i, j].plot(tTrain[inds], xDotTrainClean[inds,i], 'g', linewidth = 3,
                               label='true clean')
                axs[i, j].plot(tTrain[inds], xDotTrain[inds,i], 'b', linewidth=2, 
                               label='smoothed ' + tag)
                axs[i, j].plot(tTrain[inds], xDotTrainPredicted[::2, i], color=c, linewidth=2,
                               label = 'predicted')
                # axs[i, j].plot(tTrain[inds], xDotTrainSmoothed[::2, i], color=c, linewidth = 2,
                #                label = 'predicted (smoothed)')                
                axs[i, j].set_ylabel(variableNames[i], fontweight='bold', fontsize=12)
                axs[i, j].set_ylim(yLimits)
                axs[i, j].set_title('Derivs for train traj ' + str(j) + '\n' + \
                                    'predicted by model ' + str(j) + ' (iter ' + \
                                    str(iterationNumbersToPlot[j]) + ')', 
                                    fontweight='bold', fontsize=12)                    
                if i == 0 and j == 0:
                    axs[i, j].legend()
                if i == numVars - 1:
                    axs[i, j].set_xlabel('Time', fontweight='bold', fontsize=12) 
        #plt.tight_layout()
        fig.show()
 
    #%% 2. Plot simulations of train trajectories, both as training and as validation

    # Training set:
    if showPlotsFlag[2]:

        startTime = marginInSecs
        stopTime = numSecsInTrain - marginInSecs
        startInd = int(startTime/dt)
        stopInd = int(stopTime/dt)
        inds = np.array(range(startInd, stopInd, 2))  # note the skip = 2. Must include this skip 
        # for other time-series also, below.

        # Each plot will show the evolutions of the j'th model (ie trained on the j'th trajectory) 
        # for each true training trajectory (one per column). Each row is a variable.
        for j2 in range(numTrajTrain):
            fig, axs = plt.subplots(nrows=numVars, ncols=numTrajTrain, sharex='col', 
                                    figsize=(12, 9))

            # First, plot j'th trajectory in the j'th column:
            for j in range(numTrajTrain):
                xTrainNoisyOriginal = xTrainNoisyOriginalAll[j]
                xTrain = xTrainAll[j]
                xTrainClean = xTrainCleanAll[j]
                xTrainEvolved = xTrainEvolvedAll[j]  # The 'home' model's evolution. It includes
                # fomTimepoints only.
                yLimits = (1.1 * np.percentile(xTrainNoisyOriginal.flatten(), 1),
                           1.1 * np.percentile(xTrainNoisyOriginal.flatten(), 99.9))
                for i in range(numVars):
                    axs[i, j].plot(tTrain[inds], xTrainNoisyOriginal[inds,i], '.', 
                                   color='gray', linewidth = 2, label='noise-added')
                    axs[i, j].plot(tTrain[inds], xTrain[inds,i], 'b', linewidth = 2,
                                   label='smoothed')
                    axs[i, j].plot(tTrain[inds], xTrainClean[inds,i], 'g', linewidth = 2,
                                   label='clean true')     
                    axs[i, j].set_ylabel(variableNames[i], fontweight='bold', fontsize=12)
                    
                    if i == numVars - 1:  # Put xlabel on bottom plots only:
                        axs[i, j].set_xlabel('Time', fontweight='bold', fontsize=12)  
                axs[0, j].set_ylim(yLimits)
                axs[0, j].set_title('Training traj # ' + str(j) + '\n' + \
                                    'predicted by traj # ' + str(j2) + "'s model", 
                                    fontweight='bold', fontsize=12)

            # Then plot the evolution of the j2'th model in the j'th column:
                if j2 == j:
                    for i in range(numVars):
                        axs[i, j].plot(tTrain[inds], xTrainEvolved[::2, i], 
                                       color=colorsForTraj[j2], linewidth = 2)
                else:  # case: j is one of the val evolutions of the j2'th model
                    valEvolved = xValEvolvedAll[j2]  # valEvolved = list of trajectories, length =
                    # (numTrainTraj - 1)
                    indsVal = indsValAll[j2]
                    for k in range(numTrajTrain - 1):
                        thisEvolved = valEvolved[k]
                        col = indsVal[k]  # which column of plots = which training trajectory
                        if col == j:  # ie the trajectory evolved is the j'th, while the model is 
                        # the j2'th).
                            for i in range(numVars):
                                axs[i, col].plot(tTrain[inds], thisEvolved[::2, i],
                                                 color=colorsForTraj[j2], linewidth=2, 
                                                 label='predicted')  
                if j == 0 and i == 0: # Only put a legend of true data in top left
                    axs[i, j].legend() 
            # plt.tight_layout()
            fig.show() 
            
    #%% 3. Plot derivatives of test set:
    if showPlotsFlag[3]: 
        # One mosaic per Test trajectory.
        # First plot the true derivatives, each column = a trained model, each row a variable.
        # Indexing key: k = train traj; i = variables (row); j = model (cols)
        for k in range(numTrajTest):
            fig, axs = plt.subplots(numVars, numTrajTrain, sharex=True, figsize=(7,9))
            xDot = xDotTestTrueAll[k]
            
            for i in range(numVars):
                for j in range(numTrajTrain):
                    color = colorsForTraj[j]
                    xDotPred = xDotTestPredictedAll[j][k] 
                    axs[i, j].plot(tTest, xDot[:, i], 'dimgray', linewidth = 4,label='true')
                    axs[i, j].plot(tTest, xDotPred[:, i], color, linewidth = 1, 
                                   label = 'predicted' + str(j))
                    axs[i, j].set_ylabel('$\dot{}$'.format(variableNames[i]), fontweight='bold', 
                                         fontsize=12)
                    axs[0, j].set_title('Derivatives of test # ' + str(k) + '\n' + \
                                        'predicted by model # ' + str(j), fontweight='bold', 
                                        fontsize=12)
                    if i == 0 and k == 0:
                        axs[i, j].legend()
                    if i == numVars - 1:
                        axs[i, j].set_xlabel('Time', fontweight='bold', fontsize=12)

        plt.tight_layout()
        fig.show()

    #%% 4. Plot evolutions of models, on Test set:
        # one mosaic for each test trajectory, with each row a variable and each column a model:
    if showPlotsFlag[4]:

        startInd = 0
        stopTime = numSecsInTest
        stopInd = int(stopTime/dt)
        inds = np.array(range(startInd, stopInd, 2))  # note the skip = 2. Must include this skip 
        # for other time-series also, below.

        for k in range(numTrajTest):
            fig, axs = plt.subplots(numVars, numTrajTrain, sharex='col', figsize=(12, 9))
            for j in range(numTrajTrain):
                # Print the true trajectories first:
                    xTrue = xTestAll[k]
                    for i in range(numVars):
                        ax = axs[i, j]
                        ax.plot(tTest[inds], xTrue[inds, i], 'darkgray', linewidth = 4, 
                                label='true')
                        # Print the evolved trajectories:
                        color = colorsForTraj[j]
                        xTestEvolved = xTestEvolvedAll[j]  # a list of numTrajTest evolutions
                        x = xTestEvolved[k]
                        yLimits = (1.1 * min(min(x.flatten()), min(xTrue.flatten())),
                                   1.1 * max(max(x.flatten()), max(xTrue.flatten())))

                        ax = axs[i, j]
                        ax.plot(tTest[inds], x[inds, i], color, linewidth = 2,
                                label = 'predicted')
                        ax.set_ylabel(variableNames[i], fontweight='bold', fontsize=12)
                        ax.set_ylim(yLimits)
                        ax.set_title('test traj # ' + str(k) + '\n' +  \
                                     'evolved using traj # ' + str(j) + "'s model", 
                                      fontweight='bold', fontsize=12) 
                        if i == 0:
                            ax.legend()
                        if i == numTrajTrain - 1:
                            ax.set_xlabel('Time', fontweight='bold', fontsize=12)

        plt.tight_layout()
        fig.show()

    #%% 5. 3D train-val trajectory plots:
    # One set with original data only, one set with smoothed estimate also, one set with evolved
    # trajectories. One subplot per training trajectory.
    # Train-Val:
    if showPlotsFlag[5]:
        # define plotting times
        startTime = marginInSecs
        stopTime = numSecsInTrain - marginInSecs
        startInd = int(startTime/dt)
        stopInd = int(stopTime/dt)
        inds = np.array(range(startInd, stopInd, 1)) # note the skip. Also do for evolved 
        # time-series.

        # 5a. Plot original data only:
        fig, ax = plt.subplots(1, numTrajTrain, subplot_kw={'projection':'3d'})

        # First plot the true trajectory (and the smoothed one) for best layering:
        for j in range(numTrajTrain):
            c = colorsForTraj[j]
            xTrainNoisyOriginal = xTrainNoisyOriginalAll[j]
            xTrainClean = xTrainCleanAll[j]

            # If 2-D, make dummy z variable (actually, two dummy variables for convenience):
            if numVars == 2:
                xTrainNoisyOriginal = np.hstack((xTrainNoisyOriginal,
                                                 np.zeros(xTrainNoisyOriginal.shape)))
                xTrainClean = np.hstack((xTrainClean, np.zeros(xTrainClean.shape)))

            ax[j].plot(xTrainNoisyOriginal[inds, 0], xTrainNoisyOriginal[inds, 1],
                       xTrainNoisyOriginal[inds, 2], '.', color='gray') 
            ax[j].plot(xTrainClean[inds, 0], xTrainClean[inds, 1], xTrainClean[inds, 2], 'g',
                       markersize=1)

            if numVars == 2:
                ax[j].view_init(elev=90, azim=0)

            # Other j-dependent details:
            ax[j].set_xlabel(variableNames[0], fontweight='bold', fontsize=12)
            ax[j].set_ylabel(variableNames[1], fontweight='bold', fontsize=12)
            if numVars > 2:
                ax[j].set_zlabel(variableNames[2], fontweight='bold', fontsize=12)
            
            ax[j].set_xlim((min(xTrainNoisyOriginal[:, 0]), max(xTrainNoisyOriginal[:, 0])))
            ax[j].set_ylim((min(xTrainNoisyOriginal[:, 1]), max(xTrainNoisyOriginal[:, 1])))
            ax[j].set_zlim((min(xTrainNoisyOriginal[:, 2]), max(xTrainNoisyOriginal[:, 2])))
            ax[j].set_title('Training trajectory #' + str(j) + ', clean data = green', 
                            fontweight='bold', fontsize=12)

        # 5b. Plot original data plus smoothed estimated trajectory:
        fig, ax = plt.subplots(1, numTrajTrain, subplot_kw={'projection':'3d'})

        # First plot the true trajectory (and the smoothed one) for best layering:
        for j in range(numTrajTrain):
            c = colorsForTraj[j]
            xTrain = xTrainAll[j]
            xTrainNoisyOriginal = xTrainNoisyOriginalAll[j]
            xTrainClean = xTrainCleanAll[j]

            # If 2-D, make dummy z variable (actually, two dummy variables for convenience):
            if numVars == 2:
                xTrain = np.hstack((xTrain, np.zeros(xTrain.shape)))
                xTrainNoisyOriginal = np.hstack((xTrainNoisyOriginal,
                                                 np.zeros(xTrainNoisyOriginal.shape)))
                xTrainClean = np.hstack((xTrainClean, np.zeros(xTrainClean.shape)))

            ax[j].plot(xTrainNoisyOriginal[inds, 0], xTrainNoisyOriginal[inds, 1],
                       xTrainNoisyOriginal[inds, 2], '.', color='gray') # the noisy data points
            ax[j].plot(xTrainClean[inds, 0], xTrainClean[inds, 1], xTrainClean[inds, 2], 'g')
            ax[j].plot(xTrain[inds, 0], xTrain[inds, 1], xTrain[inds, 2], 'k.', markersize=1)

            if numVars == 2:
                ax[j].view_init(elev=90, azim=0)

            # Other j-dependent details:
            ax[j].set_xlim((min(xTrainNoisyOriginal[:, 0]), max(xTrainNoisyOriginal[:, 0])))
            ax[j].set_ylim((min(xTrainNoisyOriginal[:, 1]), max(xTrainNoisyOriginal[:, 1])))
            ax[j].set_zlim((min(xTrainNoisyOriginal[:, 2]), max(xTrainNoisyOriginal[:, 2])))
            ax[j].set_title('Training trajectory #' + str(j) +  '\n' + \
                            'clean = green, smoothed = black', fontweight='bold', fontsize=12)
        fig.show()
        
        # 5c. Plot original data AND evolved trajectories by home models only. 
        # This is similar to (6d) below except that we plot only each home model on its home
        # trajectory, ie we do not plot validation trajectories. So one row only.
        fig, ax = plt.subplots(1, numTrajTrain, subplot_kw={'projection':'3d'}) # 1 row, unlike 6d

        # First plot the true trajectory (and the smoothed one) for best layering:
        for j in range(numTrajTrain):
            c = colorsForTraj[j]
            xTrain = xTrainAll[j]
            xTrainNoisyOriginal = xTrainNoisyOriginalAll[j]
            xTrainClean = xTrainCleanAll[j]
            xTrainEvolved = xTrainEvolvedAll[j]

            # If 2-D, make dummy z variable (actually, two dummy variables for convenience):
            if numVars == 2:
                xTrain = np.hstack((xTrain, np.zeros(xTrain.shape)))
                xTrainNoisyOriginal = np.hstack((xTrainNoisyOriginal,
                                                 np.zeros(xTrainNoisyOriginal.shape)))
                xTrainClean = np.hstack((xTrainClean, np.zeros(xTrainClean.shape)))
                xTrainEvolved = np.hstack((xTrainEvolved, np.zeros(xTrainEvolved.shape)))

            # The next two are optional:
            ax[j].plot(xTrainNoisyOriginal[inds, 0], xTrainNoisyOriginal[inds, 1],
                       xTrainNoisyOriginal[inds, 2], '.', color='gray') # the noisy time-series.
            ax[j].plot(xTrain[inds, 0], xTrain[inds, 1], xTrain[inds, 2], 'k.', markersize=1)  
                                                                                    # smoothed
            
            ax[j].plot(xTrainClean[inds, 0], xTrainClean[inds, 1], xTrainClean[inds, 2], 'g')
            
            # Then plot the 'home' evolved trajectory:
            ax[j].plot(xTrainEvolved[:, 0], xTrainEvolved[:, 1], xTrainEvolved[:, 2],
                       color=c, linewidth=2)
            # make initial point bigger:
            ax[j].plot(xTrainEvolved[0:1, 0], xTrainEvolved[0:1, 1], xTrainEvolved[0:1, 2],
                       color=c, markersize=16)
            if numVars == 2:
                ax[j].view_init(elev=90, azim=0)

            # Other j-dependent details:
            # ax[j].set(xlabel=variableNames[0], ylabel=variableNames[1], zlabel=variableNames[2], 
            #           fontweight='bold', fontsize=12)
            
            ax[j].set_xlabel(variableNames[0], fontweight='bold', fontsize=12)
            ax[j].set_ylabel(variableNames[1], fontweight='bold', fontsize=12)
            if numVars > 2:
                ax[j].set_zlabel(variableNames[2], fontweight='bold', fontsize=12)
            
            ax[j].set_xlim((min(xTrainNoisyOriginal[:, 0]), max(xTrainNoisyOriginal[:, 0])))
            ax[j].set_ylim((min(xTrainNoisyOriginal[:, 1]), max(xTrainNoisyOriginal[:, 1]))) 
            ax[j].set_zlim((min(xTrainNoisyOriginal[:, 2]), max(xTrainNoisyOriginal[:, 2])))
            ax[j].set_title('Train traj ' + str(j) + '\n' + 'clean = green, smoothed = black' + \
                            ', predicted = purple', fontweight='bold', fontsize=12) 
        fig.show()
        #---------------------
        # 5d. Plot original data AND evolved trajectories by each model (both home model and 
        # models trained on other trajectories). Multiple rows.
        # Each row is a trajectory, each column are predicted trajectories of a particular model.

        fig, ax = plt.subplots(numTrajTrain, numTrajTrain, subplot_kw={'projection':'3d'})

        # First plot the true trajectory (and the smoothed one) for best layering:
        for j in range(numTrajTrain): # the row = trajectory
            for m in range(numTrajTrain): # the column = model generating predicted trajectories
                c = colorsForTraj[j]  # Each training trajectory has a particular purple 
                # associated with its model.
                xTrain = xTrainAll[j]
                xTrainNoisyOriginal = xTrainNoisyOriginalAll[j]
                xTrainClean = xTrainCleanAll[j]
                xTrainEvolved = xTrainEvolvedAll[j]

                # If 2-D, make dummy z variable (actually, two dummy variables for convenience):
                if numVars == 2:
                    xTrain = np.hstack((xTrain, np.zeros(xTrain.shape)))
                    xTrainNoisyOriginal = np.hstack((xTrainNoisyOriginal,
                                                     np.zeros(xTrainNoisyOriginal.shape)))
                    xTrainClean = np.hstack((xTrainClean, np.zeros(xTrainClean.shape)))
                    xTrainEvolved = np.hstack((xTrainEvolved, np.zeros(xTrainEvolved.shape)))

                ax[j, m].plot(xTrainNoisyOriginal[inds, 0], xTrainNoisyOriginal[inds, 1],
                              xTrainNoisyOriginal[inds, 2], '.', color='gray') # the noisy data 
                ax[j, m].plot(xTrainClean[inds, 0], xTrainClean[inds, 1], xTrainClean[inds, 2], 
                              'g')

                # Then plot the 'home' evolved trajectory:
                if m == j:
                    ax[j, m].plot(xTrainEvolved[:, 0], xTrainEvolved[:, 1], xTrainEvolved[:, 2],
                                  color=c, linewidth=2)
                    # make initial point bigger:
                    ax[j, m].plot(xTrainEvolved[0:1, 0], xTrainEvolved[0:1, 1], 
                                  xTrainEvolved[0:1, 2], color=c, markersize=16)
                if numVars == 2:
                    ax[j, m].view_init(elev=90, azim=0)

                # Other j-dependent details:
                # ax[j, m].set(xlabel=variableNames[0], ylabel=variableNames[1], 
                #              zlabel=variableNames[2], fontweight='bold', fontsize=12)
                ax[j, m].set_xlabel(variableNames[0], fontweight='bold', fontsize=12)
                ax[j, m].set_ylabel(variableNames[1], fontweight='bold', fontsize=12)
                if numVars > 2:
                    ax[j, m].set_zlabel(variableNames[2], fontweight='bold', fontsize=12)
                
                ax[j, m].set_xlim((min(xTrainNoisyOriginal[:, 0]), max(xTrainNoisyOriginal[:, 0])))
                ax[j, m].set_ylim((min(xTrainNoisyOriginal[:, 1]), max(xTrainNoisyOriginal[:, 1])))
                ax[j, m].set_zlim((min(xTrainNoisyOriginal[:, 2]), max(xTrainNoisyOriginal[:, 2])))
                ax[j, m].set_title('Train traj ' + str(j) + '\n' + 'predicted by model ' + \
                                   str(m) + ' = purple', fontweight='bold', fontsize=12)

        # Now plot the models' evolutions on non-home trajectories:
        for j in range(numTrajTrain):
            xValEvolved = xValEvolvedAll[j]
            indsVal = indsValAll[j]
            for k in range(numTrajTrain - 1):
                x = xValEvolved[k]
                if numVars == 2:
                    x = np.hstack((x, np.zeros(x.shape)))
                c = colorsForTraj[j]
                ax[indsVal[k], j].plot(x[::2, 0], x[::2, 1], x[::2, 2], color=c) 
        fig.show()
    #%% 6. 3D test trajectory plots:
    if showPlotsFlag[6]:
        # Test trajectories:
        # true and evolved have the same length
        startInd = 0
        stopTime = numSecsInTest
        stopInd = int(stopTime/dt)
        inds = np.array(range(startInd, stopInd, 2))

        # Each row corresponds to a test trajectory. Each column shows the evolution of a model 
        # (from a certain training trajectory)
        fig, axs = plt.subplots(numTrajTest, numTrajTrain, subplot_kw={'projection':'3d'})

        # First plot the true trajectory (and the smoothed one) for best layering:
        for j in range(numTrajTrain):
            c = colorsForTraj[j]
            xTestEvolved = xTestEvolvedAll[j]
            for k in range(numTrajTest):
                xTest = xTestAll[k]
                x = xTestEvolved[k]
                if numVars == 2:
                    xTest = np.hstack((xTest, np.zeros(xTest.shape)))
                    x = np.hstack((x, np.zeros(x.shape)))

                # To handle indexing weirdness:
                if numTrajTest > 1 and numTrajTrain > 1:
                    ax = axs[k, j]
                if numTrajTest > 1 and numTrajTrain == 1:
                    ax = axs[k]
                if numTrajTest == 1 and numTrajTrain > 1:
                    ax = axs[j]
                if numTrajTest == 1 and numTrajTrain == 1:
                    ax = axs
                # The true trajectory, assumed not noisy:
                ax.plot(xTest[inds, 0], xTest[inds, 1], xTest[inds, 2], '.', color='dimgray')
                # The evolved trajectory:
                ax.plot(x[inds, 0], x[inds, 1], x[inds, 2], color=c)
                # Formatting etc:
                # ax.set(xlabel=variableNames[0], ylabel=variableNames[1], 
                #        zlabel=variableNames[2], fontweight='bold', fontsize=12)
                ax.set_xlabel(variableNames[0], fontweight='bold', fontsize=12)
                ax.set_ylabel(variableNames[1], fontweight='bold', fontsize=12)
                if numVars > 2:
                    ax.set_zlabel(variableNames[2], fontweight='bold', fontsize=12) 
                
                ax.set_xlim((min(xTest[:, 0]), max(xTest[:, 0])))
                ax.set_ylim((min(xTest[:, 1]), max(xTest[:, 1])))
                if numVars > 2:
                    ax.set_zlim((min(xTest[:, 2]), max(xTest[:, 2])))
                ax.set_title('Test traj # ' + str(k) + '\n' + 'evolved by model # ' + str(j) + \
                             ' = purple', fontweight='bold', fontsize=12)
                if numVars == 2:
                    ax.view_init(elev=90, azim=0) 
        fig.show()
    
    #%% 7. FFT power histograms and time-series histograms: 
    # One mosaic per training trajectory.
    if showPlotsFlag[7]: 
        for traj in range(numTrajTrain):
            i2 = iterationNumbersToPlot[traj] 
            
            fP = historyFftPowerAll[traj][i2] 
            xP = xTrainFftPowerAll[traj]
            fH = historyHistogramsAll[traj][i2]
            fHBins = historyHistogramBinsAll[traj][i2]
            xH = xTrainHistogramAll[traj]
            xHBins = xTrainHistogramBinsAll[traj]
            
            fDotH = historyXDotHistogramsAll[traj][i2]
            fDotHBins = historyXDotHistogramBinsAll[traj][i2]
            xDotH = xDotTrainHistogramAll[traj]
            xDotHBins = xDotTrainHistogramBinsAll[traj]
            
            fig, axs = plt.subplots(3,3, figsize=(12,12))
            for j2 in range(len(variableNames)):
                # fft:
                axs[j2, 0].plot(xP[:, j2], 'b', label='true') 
                axs[j2, 0].plot(fP[:, j2], 'darkred', label='model')
                axs[j2, 0].set_title('FFT power for ' + variableNames[j2] + \
                                     '\n' + 'traj = ' + str(traj) + ', iter = ' + str(i2), 
                                     fontweight='bold', fontsize=12)
                if j2 == len(variableNames) - 1:
                    axs[j2, 0].set_xlabel('FFT power coeff index', fontweight='bold', fontsize=12)
                if j2 == 0:
                    axs[j2, 0].legend()
                # x histograms:
                axs[j2, 1].plot(xHBins[:, j2], xH[:, j2], 'b', label='true') 
                axs[j2, 1].plot(fHBins[:, j2], fH[:, j2], 'darkred', label='model')
                axs[j2, 1].set_title('Histogram for ' + variableNames[j2] + \
                                     '\n' + 'traj = ' + str(traj) + ', iter = ' + str(i2), 
                                     fontweight='bold', fontsize=12)
                if j2 == len(variableNames) - 1:
                    axs[j2, 1].set_xlabel('values of trajectory', fontweight='bold', fontsize=12)
                if j2 == 0:
                    axs[j2, 1].legend()
                # xDot histograms:
                axs[j2, 2].plot(xDotHBins[:, j2], xDotH[:, j2], 'b', label='true') 
                axs[j2, 2].plot(fDotHBins[:, j2], fDotH[:, j2], 'darkred', label='model')
                axs[j2, 2].set_title('Histogram for deriv of' + variableNames[j2] + \
                                     '\n' + 'traj = ' + str(traj) + ', iter = ' + str(i2), 
                                     fontweight='bold', fontsize=12)
                if j2 == len(variableNames) - 1:
                    axs[j2, 2].set_xlabel('values of derivative of trajectory', 
                                          fontweight='bold', fontsize=12)
                if j2 == 0:
                    axs[j2, 1].legend()
            fig.show()

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


