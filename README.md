# sindyToolkitForHighNoise

This is the Python codebase for "A toolkit for data-driven discovery of governing equations in high-noise regimes" (2022) by C.B. Delahunt and J.N. Kutz.
Any questions or comments, please email delahunt@uw.edu.


Procedure briefly:

0. Run "%matplotlib qt" in console
1. Run 'runToolkit.py'
2. Examine the outputted FoM mosaics, and choose the best model for each trajectory, eg (42, 47, 49).
3. Run 'plotSelectedIterations.py'
4. Re-run 'runToolkit.py' with 'useFullLibraryFlag' = False
5. Repeat steps 2 and 3 with the new mosaics.

-----------------------------
Procedure in detail:

0. Run "%matplotlib qt" in console (to get separate image windows).

1. Run 'runToolkit.py' or 'runToolkitExtended.py' (a version with extra experimental but maybe useless options).
	User entries can be divided as follows:
	a. System details (system type, noise level, extra variables)
	b. Important algorithm parameters (these are marked in the script with "#!!!!!#"): 
	   polynomialLibrarydegree, useFullLibraryFlag (True), initialFunctionsToUseArray, hammingWindowLengthForData, balancedCullNumber, percentileOfImputedValuesForWeights,	numEvolutionsForTrainFom, restoreBasedOnFoMsFlag, fomChangesDict. See script for usage details.
	c. Unimportant algorithm parameters: Many others, which in general do not need tuning. 
	
This script outputs, for each training trajectory, a sequence of progressively sparser models and a mosaic of Figures of Merit (FoMs) showing FoMs for each iteration in the culling sequence. It also outputs a text file detailing the sequences (culled functionals, updated weights, etc), and a .pickle file for later use.

2. Examine the mosaics, and choose a best model for each training trajectory, based on the FoMs. All the subplots zoom (in x-axis) together. Particularly useful FoMs are:
	a. In-envelope fraction: this measures whether the predicted evolution by the model stays within some nearby envelope of the true trajectory.
	b. Relative error of std dev: This compares the std devs of the predicted and true trajectories. If this value is close to 0, the behavior of the trajectories is similar, even if the exact trajectories are different.
	Other FoMs are useful to identify bad models:
	c. xDot 80th percentile of errors: Since models can typically match the true derivatives well, values diverging from 0 signal bad models. 
	d. In-bounds fraction: This measures whether the predictions stay within the overall bounds of the true trajectory. Values < 1 usually signal exploding predictions.
	e. Histogram and FFT correlations: Values far different from ideal can signal worse models.
	f. Minimum histogram correlation between evolutions: If this is not equal to 1, the model is not stable (relevant only if numEvolutionsForTrainFom > 1).
	g. Number of functionals per variable: If you know the desired level of sparsity, this can be a guide.
	
3. Given the best iteration for each trajectory, eg (43, 47, 49), run 'plotSelectedIterations.py'.
	Key user entries are:
	a. IterationNumbersToPlot: eg (43, 47, 49)
	b. savedHistoriesFile: the .pickle outputted by 'runToolkit'.
This script plots trajectories for the chosen models (ideally, suppress or ignore test set trajectories). It also prints to console a boolean array 'unionFunctionsToUseArray'. This is the union of the libraries of the chosen models (since the models may have each retained or culled different vital functionals).

4. Re-run 'runToolkit.py', with the following two changes to User Entries:
	a. useFullLibraryFlag = False
	b. initialFunctionsToUseArray = np.array( [printout of 'unionFunctionsToUseArray'] )
This run starts with a much smaller library consisting of the most likely candidate functionals.
	
Repeat steps 2 and 3 with the new mosaics, which hopefully reflect the best of the initial run models (test set trajectories can be examined for these 2nd stage model choices).

