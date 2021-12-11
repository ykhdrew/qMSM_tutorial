#!/usr/bin/env python
#######################################################
#Written by Daniel Silva 
#Based in the original SHC code from Yuan YAO and Xuhui Huang:
#  Proceedings of the Pacific Symposium on Biocomputing, 15, 228-239, (2010)
#
#Intended to be used in the SimTK project
#Ver. 1.5b 21/Apr/2011
#######################################################
import optparse
import sys
import linecache
import scipy.io
import numpy as np
#import colorsys
from pylab import *
from numpy import *
from scipy import *
from scipy.sparse import *
from scipy.sparse.linalg import *
from scipy.linalg import eig
from scipy.interpolate import interp1d
from scipy.sparse.linalg.eigen.arpack import *


def version():
 print "(OUTPUT) Python SHC ver 2.01b"


#def licence():

def main():
    version()
#	licence()
    p = optparse.OptionParser()
    p.add_option('--outMicroCountMatrixName', '-c', default="microstateCountMatrix.mtx")
    p.add_option('--lagstep', '-l', default="1")
    p.add_option('--headDir', '-d', default="./")
    p.add_option('--trajlist', '-t', default="-1")
	p.add_option('--plevelsFile', '-p', default="plevels.shc")
	p.add_option('--outMacrostateAssignementsMap', '-s', default="macrostateMap.map")
	p.add_option('--writeMacroAssignments', '-w', default="0")
	p.add_option('--optimumMacrostateSize', '-o', default="0.01")
	p.add_option('--maximumAssignIterations', '-i', default="10")
	p.add_option('--removeBarelyConnectedMicros', '-r', default="0")
    p.add_option('--writeTCMtxt', '-x', default="0")
    p.add_option('--scanModeTopDensity', '-a', default="0.0")
    p.add_option('--inputMatrix', '-m', default="")
	p.add_option('--outFlowGraphName', '-f', default="macroFlowGraph.dot")
	p.add_option('--bJumpWindow', '-j', default="0")
	p.add_option('--whichGap', '-g', default="1")

	
	options, arguments = p.parse_args()
	outMicroCountMatrixName = (options.outMicroCountMatrixName)
	tLag = int(options.lagstep)
	headDir = (options.headDir)
	trajlistfiles = (options.trajlist)
	pLevelsFilename = (options.plevelsFile)
	outMacrostateAssignementsMap = (options.outMacrostateAssignementsMap)
	optimumMacrostateSize = float(options.optimumMacrostateSize)
	writeMAssignments = int(options.writeMacroAssignments)
	maximumAssignIterations = int(options.maximumAssignIterations)
	numRemoveBarelyConnectedMicros = int(options.removeBarelyConnectedMicros)
	writeTCMtxt = int(options.writeTCMtxt)
	scanModeTopDensity = float(options.scanModeTopDensity)
	inputMatrix = (options.inputMatrix)
	outFlowGraphName = (options.outFlowGraphName)
	bJumpWindow = int(options.bJumpWindow)
	chooseGap = int(options.whichGap)

	#if (len(inputMatrix) == 0 ):
	#        originalMicrosCountM= getMicroTransitionsFromAssignements(tLag, headDir, trajlistfiles, bJumpWindow)
	#else:
	#	print "(OUTPUT) ", ("Reading data from TCM file: \"%s\" ", inputMatrix)
	#	if ( linecache.getline(inputMatrix, 1).strip() == "%%MatrixMarket matrix coordinate integer general"):
	#		print "(OUTPUT) ", ("Detected sparce matrix in the Matrix Market format")
	#		originalMicrosCountM = scipy.io.mmread(inputMatrix)
	#	else:
    #       		print "(OUTPUT) ", ("Detected matrix in raw txt format")
    #        		originalMicrosCountM = genfromtxt(inputMatrix)
    #       		originalMicrosCountM = lil_matrix(originalMicrosCountM)

	originalMicrosCountM = scipy.io.mmread(inputMatrix)
	#The code is made to use a float matrix, even if the input (transitions) are integers. This way is just convinient to avoid errors due to loosing floats
	originalMicrosCountM = originalMicrosCountM.tocsc()/1.0

    writeCountMatrix(originalMicrosCountM, outMicroCountMatrixName, "Writing microstates transition count matrix", writeTCMtxt)

    if (numRemoveBarelyConnectedMicros > 0):
	originalMicrosCountM = removeBarelyConnectedMicros(originalMicrosCountM, numRemoveBarelyConnectedMicros)
		
    connectedMicrosCountM, connectedMicrosIndex = getConnectedMicrostates(originalMicrosCountM)
        writeCountMatrix(connectedMicrosCountM, ("%s_connected" % outMicroCountMatrixName), "Writing connected microstates transition count matrix", 0)
        connectedMicrosCountM_X = csc_matrix(connectedMicrosCountM + connectedMicrosCountM.conj().transpose())/2 ;
        microstate_size = connectedMicrosCountM.sum(axis=1)
        cumulativeSumOfRows = cumulativeDesityFunctionOfHeightFilter(microstate_size)
    pLevels=[]
	if ( scanModeTopDensity > 0.0 ):
		pLevels = scanPlevels(cumulativeSumOfRows, connectedMicrosCountM_X, microstate_size, 0.01, 0.01, scanModeTopDensity, chooseGap)

	else:
		pLevels = readPlevels(pLevelsFilename, cumulativeSumOfRows)
    clusters = zeros(len(pLevels), int)
        levels = []
    levelsLine=""

    for i in range (0, len(pLevels)):
		if ((sum(cumulativeSumOfRows<=pLevels[i])) > 2): #Detect and remove density levels with <10 microstate
			levels.append(sum(cumulativeSumOfRows<=pLevels[i]))
			levelsLine += ("%1.3f  " % pLevels[i])
		else:
			print "(OUTPUT) ", ("Density level at \"%1.3f\" is empty or have to few microstates (<2), it was removed it from the analysis"% (pLevels[i]))
	print "(OUTPUT) ", ("**SHC analysis will use %d density levels: %s" % (len(levels), levelsLine))
	(aBettis, specGaps) = apBetti(connectedMicrosCountM_X, microstate_size, levels)
	for i in range (0, len(levels)):
                if (chooseGap < 1):
                        print "(OUTPUT) ", ("WARNING: The spectral gap choosen (1st, 2nd, etc) cannot have a value less than 1, automaticaly changing the gap (-g) to 1")
			chooseGap=1
		if (chooseGap > 1):
			print "(OUTPUT) ", ("WARNING:You are using an spectral gap ( ) different to the 1st. Is this really what you want to do?")
		clusters[i] = aBettis[i][chooseGap-1]
	superLevels=superLevelSet(microstate_size, levels, clusters) #0 is IDs and 1 is IGs
	(adja, nodeInfoLevel, nodeInfoLevelSize, nodeInfoSet, nodeInfoFilter, levelIdx, ci, csize) = superMapper(connectedMicrosCountM_X, superLevels)
	(cptLocalMax, cptGradFlow, cptEquilibriumEQ) = flowGrad(adja, levelIdx, nodeInfoLevel, nodeInfoLevelSize, nodeInfoSet, nodeInfoFilter)
	writeFlowGraph(cptGradFlow, nodeInfoLevel, nodeInfoLevelSize, nodeInfoSet, nodeInfoFilter, levelIdx, superLevels, outFlowGraphName, pLevels)
	(ci, csize, fassign, T, Qmax, id_fuzzy) = optimumAssignment(connectedMicrosCountM_X, cptEquilibriumEQ, nodeInfoLevelSize, nodeInfoSet, nodeInfoFilter, maximumAssignIterations)
	writeMacrostateMap(outMacrostateAssignementsMap, originalMicrosCountM.shape[0], ci, connectedMicrosIndex)
	if (writeMAssignments ==1):
		writeMacroAssignments(tLag, headDir, trajlistfiles, ci, connectedMicrosIndex, originalMicrosCountM.shape[0])
	print "(OUTPUT) ", ("Done with SHC!")



def scanPlevels(cumulativeSumOfRows, connectedMicrosCountM_X, microstate_size, start, incr, end, chooseGap):
                print "(OUTPUT) ", "Will perform a scan to discover optimum density levels for SHC (EXPERIMENTAL)"
                clustersScan = zeros(1, int)
		pLevels=[]
                pLevelsScan=[]
                pLevelsScan.append(0)
                pLevelSGQuality=[]
                pLevelNumMacro=[]
                tmpMaxNumMacro=0
                tmpMaxGapQuality=0
		testLevels = np.arange(start,end,incr)
                for i in testLevels:
                        levelsScan = []
                        pLevelsScan[0] = i
                        specGapQuality=0
                        print "(OUTPUT) ", ("Testing Density level: \"%1.3f\" " % pLevelsScan[0])
                        if ((sum(cumulativeSumOfRows<=pLevelsScan[0])) > 1+chooseGap):
                                levelsScan.append(sum(cumulativeSumOfRows<=pLevelsScan[0]))
                                (aBettis, specGaps) = apBetti(connectedMicrosCountM_X, microstate_size, levelsScan)
                                clustersScan[0] = aBettis[0][0]
                                superLevels=superLevelSet(microstate_size, levelsScan, clustersScan) #0 is IDs and 1 is IGs
                                (adja, nodeInfoLevel, nodeInfoLevelSize, nodeInfoSet, nodeInfoFilter, levelIdx, ci, csize) = superMapper(connectedMicrosCountM_X, superLevels)
				print specGaps
				specGapQuality = specGaps[0][chooseGap-1] - specGaps[0][chooseGap]
                                if ( (len(csize[0])) > tmpMaxNumMacro):
                                        tmpMaxNumMacro = len(csize[0])
                                        tmpMaxGapQuality = specGapQuality
                                        pLevels.append(np.copy(pLevelsScan[0]))
                                        pLevelSGQuality.append(np.copy(specGapQuality))
                                        pLevelNumMacro.append(np.copy(tmpMaxNumMacro))
                                elif ( ((len(csize[0])) <= tmpMaxNumMacro) and (specGapQuality > tmpMaxGapQuality) ):
                                        tmpMaxNumMacro = len(csize[0])
                                        tmpMaxGapQuality = specGapQuality
                                        pLevels[(len(pLevels)-1)] = pLevelsScan[0]
                                        pLevelSGQuality[(len(pLevels)-1)] = specGapQuality
                                        pLevelNumMacro[(len(pLevels)-1)] = len(csize[0])
                        else:
                                print "(OUTPUT) ", ("Skipping density level at \"%1.3f\" because it contains to few microstates ( <2 )" % pLevelsScan[0])
                print "(OUTPUT) ", "Optimum density levels identified  &  num of macrostates per level:"
		counter=0
		for i in pLevels:
			print "(OUTPUT) \t", i, "\t", pLevelNumMacro[counter]
			counter+=1
                print "(OUTPUT) ", "Sum of the differences of the spectral gaps:"
		for i in pLevelSGQuality:
			print "(OUTPUT) ", i
                print "(OUTPUT) ", "Density levels scan DONE. Proceding to the SHC clustering!"
		return pLevels


def removeBarelyConnectedMicros(originalMicrosCountM, numRemoveBarelyConnectedMicros):
	print "(OUTPUT) ", ("Removing barely connected microstates with a cut off <%d transitions (in or out) (EXPERIMENTAL)" % numRemoveBarelyConnectedMicros)
	counter=0
	originalMicrosCountM = originalMicrosCountM.todense()
	for i in range (0, originalMicrosCountM.shape[0]):
		if (((originalMicrosCountM[i,:].sum() - originalMicrosCountM[i,i] - numRemoveBarelyConnectedMicros) < 0 ) or ((originalMicrosCountM[:,i].sum() - originalMicrosCountM[i,i] - numRemoveBarelyConnectedMicros) < 0 )):
			counter+=1
			originalMicrosCountM[i,:] = 0
			originalMicrosCountM[:,i] = 0
	print "(OUTPUT) ", ("Removed %d barely connected microstates (turn to pop 0)..." % counter)
	originalMicrosCountM = csc_matrix(originalMicrosCountM)
	return(originalMicrosCountM)

def writeMacrostateMap(outName, nMicro, ci, connectedMicrosIndex):
	print "(OUTPUT) ", ("Writting macrostate maping file: %s" % outName)
	f = open(outName,'w')
	micro2macro = zeros((nMicro), int)
	micro2macro[connectedMicrosIndex] = ci
	for i in range(0, nMicro):
		line = (micro2macro[i]-1)
		print >>f, line
	f.close()
	print "(OUTPUT) ", ("Done writting macrostate maping file!")
	
def writeMacroAssignments(tLag, headDir, trajlistfiles, ci, connectedMicrosIndex, nMicro):
	print "(OUTPUT) ", ("Writting macrostate assignments to:")
	micro2macro = zeros((nMicro), int)
	micro2macro[connectedMicrosIndex] = ci
	for filenameInp in file(trajlistfiles):
		filenameInp = filenameInp.strip()
		filenameInp = "%s/assignments/%s" % (headDir,filenameInp)
		tmpLineLen=len(filenameInp)+10
		sys.stdout.write('(OUTPUT)  %s' % filenameInp)
		for i in range (0,  tmpLineLen):
			sys.stdout.write('\b')
		output = []
		for line in file(filenameInp):
			line=line.strip().split()
			if (int(line[0]) > -1):
				lineout="%d %d" %( int(line[0]), (micro2macro[int(line[0])] -1))
			else:
				lineout="%d -1" %( int(line[0]))
			output.append(lineout)
			
		f = open(filenameInp,'w')
		for line in output:
			print >>f, line
		f.close()
	print "\n", "(OUTPUT) ", ("Done writting macrostate assignments!")
		

 
def getMicroTransitionsFromAssignements(tLag, headDir, trajlistfiles,bJumpWindow):
 originalNumOfMicros=0
 totalCounts=0
 numberOfTrajs=0
 print "(OUTPUT) ", ("Assesing the number of microstates...")
 for filenameInp in file(trajlistfiles):
   filenameInp = filenameInp.strip()
   filenameInp = "%s/assignments/%s" % (headDir,filenameInp)
   tmpLineLen=len(filenameInp)+10
   numberOfTrajs+=1
   sys.stdout.write('(OUTPUT)  %s' % filenameInp)
   for i in range (0,  tmpLineLen):
	   sys.stdout.write('\b')
   for line in file(filenameInp):
     line = line.strip().split()
     line = int(line[0])
     if (line > originalNumOfMicros):
        originalNumOfMicros = line
 if (originalNumOfMicros>0):
   originalNumOfMicros+=1
   print "(OUTPUT) ", ("Found %d microstates in %d trajectories" % (originalNumOfMicros, numberOfTrajs))
 elif (originalNumOfMicros==0):
   print "(OUTPUT) ", ("Found 0 microstates in %d trajectories, cannot continue!", numberOfTrajs)
   exit(0)
 
 print "(OUTPUT) ", ("Reading microstates assignments from files and counting transitions:")
 originalMicrosCount= lil_matrix((originalNumOfMicros, originalNumOfMicros))
 tmpLineLen=0
 for filenameInp in file(trajlistfiles):
   filenameInp = filenameInp.strip()
   filenameInp = "%s/assignments/%s" % (headDir,filenameInp)
   tmpLineLen=len(filenameInp)+10
   
   for i in range (0,  tmpLineLen):
           sys.stdout.write('\b')
   previousm=-1
   trajLength = 0
   for line in file(filenameInp):
     trajLength += 1

###NEXT IS SLIDING WINDOW###
   if ( bJumpWindow == 0 ):
	   trajectory=zeros ((trajLength), int)
	   for i in range (1, trajLength):                      
	       line = linecache.getline(filenameInp, i).strip().split()
	       trajectory[i] = line[0]
	   for i in range (0, trajLength-tLag):
	       if ((trajectory[i] >= 0) & (trajectory[i+tLag]>= 0)) :
	           originalMicrosCount[trajectory[i], trajectory[i+tLag]]+=1         
###END SLIDING WINDOW###

###NEXT IS JUMP WINDOW###
   if ( bJumpWindow == 1 ):
	   trajectory=zeros ((trajLength/tLag)+1, int)
	   for i in range (0, trajLength):			 #Qin's Fix (THX)
	       line = linecache.getline(filenameInp, i+1).strip().split()
	       if(i%tLag==0): 						 #Qin's Fix (THX)
	         trajectory[i/tLag]=(int(line[0]))
	   for i in range(0, (trajLength/tLag)-1):
			if ((trajectory[i] >= 0) & (trajectory[i+1]>= 0)) :
				originalMicrosCount[trajectory[i], trajectory[i+1]]+=1            
###END JUMP WINDOW##

 print "\n", "(OUTPUT) ", ("Finished with microstates count!")
 print "(OUTPUT) ", ("Total number of microstate transitions: %d" % originalMicrosCount.sum() )
 originalMicrosCount = originalMicrosCount.tocsc()
 emptyNumber=0
 for i in range (0, originalNumOfMicros):
   if ((originalMicrosCount[i,:].sum() + originalMicrosCount[:,i].sum()) == 0):
	   emptyNumber+=1   
	   print("Warning microstate %d is empty!" % i)  
 if(emptyNumber > 0):
	 print "(OUTPUT) ", ("Warning, there are %d empty microstates" % emptyNumber)
 print "(OUTPUT) ", ("There are %d non-empty microstates" % (originalMicrosCount.shape[0]-emptyNumber))
 return (originalMicrosCount)


def writeCountMatrix ( originalMicrosCount, outMicroCountMatrixName, message, doWriteTXT):
 print "(OUTPUT) ", (message)
 scipy.io.mmwrite(outMicroCountMatrixName, originalMicrosCount, field="integer")
 if (doWriteTXT == 1):
  print "(OUTPUT)  Writing (also) a count matrix in TXT format! (May be very slow, be patient)"
  outMicroCountMatrixName="%s.txt"%(outMicroCountMatrixName)
  f = open(outMicroCountMatrixName,'w')
  advanceCounter=0.0
  numMicros=originalMicrosCount.shape[0]
  originalMicrosCount=originalMicrosCount.tolil()
  outline="0.0% Complete" 
  sys.stdout.write('(OUTPUT)  %s' %outline)
  for i in range(0, numMicros):
   advanceCounter+=1.0
   print advanceCounter, numMicros
   line=" "
   for j in range(0, numMicros):
    line+= str(int(originalMicrosCount[i,j])) + " "
   print >>f, line
   if (advanceCounter >= (numMicros/100.0)):
    for k in range (0, len(outline)+10):
     sys.stdout.write('\b')
    sys.stdout.write('(OUTPUT)  %s' % outline)
    outline="%.1f%% Complete " % ((i+1)*100/numMicros)
    advanceCounter=0
  print "\n", "(OUTPUT) ", ("Finished TXT write!")
  f.close()


def getConnectedMicrostates (originalMicrosCount):
 print "(OUTPUT) ", ("Searching connected microstates using graph theory")
 microConnectedComponents=cs_graph_components((originalMicrosCount + originalMicrosCount.conj().transpose()))
 componentsSize=zeros((microConnectedComponents[0]+1), int)
 emptySize=0
 for i in microConnectedComponents[1]:
  if (i >= 0):
   componentsSize[i+1]+=1
  else:
   emptySize +=1

 
 indexMaxConnected, sizeMaxConnected = componentsSize.argmax(0), componentsSize.max(0)
 lineout = ("Found %d connected microstates, %d disconnected microstates and %d empty microstates" % (sizeMaxConnected, (componentsSize.sum()-sizeMaxConnected), emptySize))
 print "(OUTPUT) ", lineout
 if ((emptySize > 0) | ((componentsSize.sum()-sizeMaxConnected) > 0)):
  print "(OUTPUT) ", "Removing disconnected microstates"
  connectedMicrosIndex = where(microConnectedComponents[1] == (indexMaxConnected-1))
  connectedMicrosIndex = getIndexFromArray(connectedMicrosIndex[0])
  connectedMicros = originalMicrosCount[ix_(connectedMicrosIndex,connectedMicrosIndex)]
 else:
	connectedMicros = originalMicrosCount
	connectedMicrosIndex = range(0,componentsSize.sum())
 return connectedMicros, connectedMicrosIndex

def readPlevels(fileName, cumulativeSumOfRows):
 print "(OUTPUT) ", ("Reading density levels from file: %s" % fileName)
 pLevels=[] 
 for line in file(fileName):
   line = line.strip()
   pLevels.append(float(line))
 return (pLevels)
 
 
def cumulativeDesityFunctionOfHeightFilter(x):
 total = sum(x)
 x = -x
 x.ravel().sort()
 x = -x
 y = x.cumsum(axis=0)/total
 return y


def getIndexFromMatrix(indexA):
   xx = indexA
   xxx=[]
   for i in range (0, len(xx)):
    xxx.append(xx[i,0])
   return(xxx)
   
def getIndexBFromMatrix(indexB):
   xx = indexB
   xxx=[]
   for i in range (0, len(xx)):
    xxx.append(xx[0,i])
   return(xxx)
   
def getIndexFromArray(indexA):
   xx = indexA
   xxx=[]
   for i in range (0, len(xx)):
    xxx.append(xx[i])
   return(xxx)
   
def IntList2array(listA):
   xx = listA
   xxx= zeros((len(listA)),int)
   for i in range (0, len(xx)):
    xxx[i] = xx[i]
   return(xxx)

def apBetti(X, filterX, levels):
 print "(OUTPUT) ", ("Computing persistent aproximate betti numbers via spectral gaps")
 #X = X.tocsc()
 ig = filterX/(max(filterX))
 #print "PPC",filterX, (max(filterX))
 ig = -ig
 rk = ig.argsort(axis=0)
 ig.sort(axis=0)
 ig = -ig

 MAXNUMBEREIG = 20;
 k = MAXNUMBEREIG
 eps = 1e-4
 randSurf = 1e-1
 N = len(filterX)
 revecs = []
 revals = []
 Components = []
 specGaps = []
 aBettis = []

 for i in range (0, len(levels)):
  revecs.append(0)
  revals.append(0)
  Components.append(0)
  specGaps.append(0)
  aBettis.append(0)

 print "(OUTPUT) ", ("Level\tSize\t#Comp\tB0_1\tGap_1\t\tB0_2\tGap_2\t\tB0_3\tGap_3")
 for i in range (0, len(levels)):
  if (levels[i] > 1):
   n = int(levels[i])
  else:
   n = int(sum(ig>=levels[i]))
  outline= ("%d\t %d\t"%(i,n));
  if (n == 1):
   Components[i] = 1
   specGaps[i] = ones(MAXNUMBEREIG);
   aBettis[i] = [1, zeros(MAXNUMBEREIG-1)]
  else: 
    tmpindx = getIndexFromMatrix(rk[0:n])
    Y = csc_matrix(((X[ix_(tmpindx,tmpindx)])) + (eps*identity(n)) +(randSurf * ones((n,n), float)/n))

    Y2 = zeros((n,n))
    tmparray=[]
    for j in Y.sum(axis=1):
		tmparray.append(j[0,0])
    Y2[diag_indices(n)]= tmparray
    Y2 = csc_matrix(Y2) 
    
    sigma = 1+eps+randSurf  
    B = Y - sigma*Y2
    sigma_solve = dsolve.splu(B)
    Y2L = aslinearoperator(Y2) 

    if ((n-4) > MAXNUMBEREIG):
#		revals[i],revecs[i] = ARPACK_gen_eigs( Y2L.matvec, sigma_solve.solve, Y2L.shape[0], sigma, MAXNUMBEREIG, 'LM' )
		revals[i],revecs[i] = eigs( Y, MAXNUMBEREIG, Y2, sigma, which='LM', maxiter=10000 )
    else:
		revals[i],revecs[i] = scipy.linalg.eig( Y.todense(),Y2.todense() )
		revals[i]=real(revals[i])
    #SORT EIGENVALUES AND EIGENVECTORS
    tmpindsort = argsort(-revals[i])
    revals[i] = revals[i][tmpindsort]
    revecs[i] = revecs[i][:, tmpindsort]   # second axis !!
    if (n > MAXNUMBEREIG):
	revals[i] = revals[i][:MAXNUMBEREIG]
	revecs[i] = revecs[i][:, :MAXNUMBEREIG]

    #Remove later DASM
#    tmplineout=""
#    for ii in revals[i]:
#    	tmplineout+=" "+ str(ii)
#    print "(DEBUG) Using a matrix of %ix%i, eigenvalues are:\n(DEBUG) \t" %((n-4),(n-4)), tmplineout
    #END REMOVE#

    Components[i] = sum(revals[i]>(1-1e-5))
    tmpSpecGaps = -(abs(diff(revals[i])))
    aBettis[i] = tmpSpecGaps.argsort(axis=0)

    for xx in range (1, len(revals[i])):  #FIX for eigenvalues = 1.0 on lowlevels
        if ((revals[i][xx]+1e-5) >= 1) and (aBettis[i][0] < xx):
                aBettis[i][0]+=1
        else:
                break
    tmpSpecGaps.sort(axis=0)   
    specGaps[i] = -tmpSpecGaps
   
    outline += ('%d\t'% Components[i])
    for gaplist in range (0, min(3,len(aBettis[i]))):
     outline += ('%d\t %f\t'%(aBettis[i][gaplist], specGaps[i][gaplist]));
    
    print "(OUTPUT) ", outline
 print "(OUTPUT) ",("Done with betti numbers!")
 return (aBettis, specGaps)

def superLevelSet(filterX, levels, clusters):
 ig = -filterX
 idd = ig.argsort(axis=0)
 ig.sort(axis=0)
 ig = -ig
 superLevelSetId = []
 superLevelSetIg = []
 for i in range (0, len (levels)):
	 superLevelSetId.append(np.copy(idd[0:levels[i]]))
	 superLevelSetIg.append(np.copy(clusters[i]))
 return (superLevelSetId, superLevelSetIg)


def superMapper (X,superLevelSet):
	print "(OUTPUT) ", ('Executing the SMapper')
	numPoints = X.shape[0]
	dim = X.shape[1]
	if (dim!=numPoints):
		print "(OUTPUT) ", ('ERROR: the input for the mapper must be a symmetric transition count matrix!')
		sys.exit()
	numLevels = len(superLevelSet[0])
	lengthX = []
	idxSort = []

	for i in range (0, numLevels):
		lengthX=concatenate((lengthX,len(superLevelSet[0][i])), axis=None)
		tmpReshape = superLevelSet[0][i].reshape(1,lengthX[i])
		tmpReshape2 = []
		for j in range (0, size(tmpReshape, axis=1)):
			tmpReshape2.append(np.copy(tmpReshape[0,j]))
		idxSort=concatenate((idxSort,tmpReshape2), axis=None)
	Y = X[ix_(idxSort,idxSort)];

	print "(OUTPUT) ", ("SMapper:\tnumber of points %d" % numPoints);
	print "(OUTPUT) ", ("\t\tnumber of levels %d" % len(superLevelSet[0]));
	numGraphNodes = 0
	nodeInfoLevel = []
	nodeInfoLevelSize = []
	nodeInfoSet = []
	nodeInfoFilter = []
	levelIdx = []
	adja = []
	ci = []
	csize = []
	numCluster = []
	for level in range (0, len(superLevelSet[0])):
		index1= getIndexFromMatrix(superLevelSet[0][level])
		data = (X[ix_(index1,index1)])
		citmp, csizetmp, specVals, specVecs, specGaps, conduct, cluster_treeData, cluster_treeConduct, cluster_treeLeft, cluster_treeRight = spectralClustering(data,superLevelSet[1][level])
		
		ci.append(np.copy(citmp))
		csize.append(np.copy(csizetmp))
		numCluster.append(len(csize[level]))
		print "(OUTPUT) ", ("Level %d has %d macrostates out of %d microstates" % (level ,numCluster[level], data.shape[0]))
		numGraphNodes = len(nodeInfoLevel)
		for i in range (0,numCluster[level]):
			new_node = i + numGraphNodes
			if (i==0):
				levelIdx.append(np.copy([new_node]))
			else:
				levelIdx[level] = concatenate((levelIdx[level],new_node), axis=None)
			nodeInfoLevel.append(np.copy(level));
			nodeInfoLevelSize.append(data.shape[0])
			thisNodeIndex = where(ci[level]==i)
			nodeInfoSet.append(np.copy(superLevelSet[0][level][thisNodeIndex]))
			nodeInfoFilter.append(np.copy(level))

		if(level > 0):
			prevLvlIdx = levelIdx[level-1]
			thisLvlIdx = levelIdx[level]
			for i in range (0,len(prevLvlIdx)):
				for j in range (0,len(thisLvlIdx)):
					a = prevLvlIdx[i]
					b = thisLvlIdx[j]
					N_ab = len(intersect1d(getIndexFromMatrix(nodeInfoSet[a]),getIndexFromMatrix(nodeInfoSet[b])));
					if (N_ab > 0):
						adja.append(np.copy([a,b,N_ab]))
			adjaArray = array2matrix(adja, len(nodeInfoLevel))

	if (numLevels == 1):
		adjaArray = zeros((len(nodeInfoLevel),len(nodeInfoLevel)),int)
		
	print "(OUTPUT) ", ('SMapper done...')
	return(adjaArray, nodeInfoLevel, nodeInfoLevelSize, nodeInfoSet, nodeInfoFilter, levelIdx, ci, csize)

def array2matrix(arr, lenArra):
	result = zeros((lenArra,lenArra),int)
	for i in arr:
		result[i[0]][i[1]]= i[2]
		result[i[1]][i[0]]= i[2]
	return (result)
			
def spectralClustering(X,k):
	clusterSizeThreshold=0
	n = X.shape[0];
	eps = 1e-4
	randSurf = 1e-1
	MAXCLUSTER = min(50,n)
	Y = csc_matrix(X + (eps*eye(n,n)) +(randSurf * ones((n,n), float)/n))
	Y2 = zeros((n,n))
	tmparray=[]
	for j in Y.sum(axis=1):
		tmparray.append(np.copy(j[0,0]))
	Y2[diag_indices(n)]= tmparray
	Y2 = csc_matrix(Y2)


	sigma = 1+eps+randSurf  
	B = Y - sigma*Y2
	sigma_solve = dsolve.splu(B)
	Y2L = aslinearoperator(Y2)
#	printDebug(MAXCLUSTER, "MAXCLUSTER")

        printDebug(n, "N")
#SPARCE matrix solver HAVES SOME PROBLEM since can return eigenvalues = 0.0, maybe increment the number of cycles DASM#
	if ((n-4) > MAXCLUSTER):
#		specVals,specVecs   = ARPACK_gen_eigs(Y2L.matvec, sigma_solve.solve, Y2L.shape[0], sigma, MAXCLUSTER-1, 'LM')
		specVals,specVecs   = eigs( Y, MAXCLUSTER-1, Y2, sigma, which='LM',maxiter=10000 )
		printDebug(specVals,"Specvals1a")
	else:
		specVals,specVecs  = scipy.linalg.eig(Y.todense(),Y2.todense())
		specVals=real(specVals)
		printDebug(specVals,"Specvals1b")
#END#
#NEXT temporary fix
#        specVals,specVecs  = scipy.linalg.eig(Y.todense(),Y2.todense())
#	specVals=real(specVals)
#END fix

#                #SORT EIGENVALUES AND EIGENVECTORS
        tmpindsort = argsort(-specVals)
        specVals = specVals[tmpindsort]
        specVecs = specVecs[:, tmpindsort]   # second axis !!
	
#        if (n > MAXCLUSTER):
#        	specVals = specVals[:MAXCLUSTER]
#        	specVecs = specVecs[:, :MAXCLUSTER]
	printDebug(specVals, "SpecvalsSortShort")

	specGaps = -(abs(diff(specVals)))
	numComponents = sum(specVals>1-(1e-10))

#TODO: add this DASM#
#if numComponents>1,
#	cluster_tree{1}.left = 2;
#	cluster_tree{1}.right = 3;
#	for i=1:numComponents,
#		mn = mean(abs(spectrum.vecs(:,i)));
#		cluster_tree{i+1}.data = find(abs(spectrum.vecs(:,i))>=mn);
#		cluster_tree{i+1}.left = 0;
#		cluster_tree{i+1}.right = 0;
#		id_complement = find(abs(spectrum.vecs(:,i))<mn);
#		cluster_tree{i+1}.conduct = sum(sum(X(cluster_tree{i+1}.data,id_complement)))/sum(sum(X(cluster_tree{i+1}.data,cluster_tree{i+1}.data)));
#	end		
#end
#END TODO#

	cluster_treeData=[]
	cluster_treeData.append(range(0,n))
	cluster_treeLeft = [0]
	cluster_treeRight = [0]
	cluster_treeConduct = [0]
	printDebug(numComponents,"numComponents")
	printDebug(k+1,"k+1")
	for i in range (numComponents, k+1):  #k is the number of components identified by Betty numbers
		tree_size = len(cluster_treeData)
		variation = zeros((tree_size))             
		for j in range (0, tree_size):
			if ((cluster_treeLeft[j] == 0) and (cluster_treeRight[j] == 0)):
				tmp = specVecs[cluster_treeData[j], i]
				if (len(tmp) > 1):
					variation[j] = (var(tmp)*len(tmp)/(len(tmp)-1));
				else:
					variation[j] = 0;
		mx = variation.max(0)
		ind = variation.argmax(0)
		indices = cluster_treeData[ind]
		printDebug(indices,"indices")
		printDebug(len(indices),"lenindices")
		nn = len(indices)
		if (i==1):
			Xsplit = csc_matrix(X[ix_(indices,indices)]+eps*eye(nn,nn)+randSurf*ones((nn,nn))/nn)
			vecFiedler = specVecs[:,i]		
		else:
			Xsplit = csc_matrix(X[ix_(indices,indices)]+eps*eye(nn,nn)+randSurf*ones((nn,nn))/nn)
			
			Y2 = zeros((nn,nn))
			tmparray=[]
			for j in Xsplit.sum(axis=1):
				tmparray.append(np.copy(j[0,0]))
			Y2[diag_indices(nn)]= tmparray
			Y2 = csc_matrix(Y2)

			B = Xsplit - sigma*Y2
			sigma_solve = dsolve.splu(B)
			Y2L = aslinearoperator(Y2)

			##TODO: maybe somethingWrongHere DASM##
			if ((nn-4) > 20):
#				splitVals,splitVecs  = ARPACK_gen_eigs(Y2L.matvec, sigma_solve.solve, Y2L.shape[0], sigma, 3, 'LM')
				splitVals,splitVecs  = eigs( Xsplit, 3, Y2, sigma, which='LM',maxiter=10000 )
			else:
				splitVals,splitVecs = scipy.linalg.eig(Xsplit.todense(),Y2.todense())
				splitVals=real(splitVals)
			##END ToDo##
                	##SORT EIGENVALUES AND EIGENVECTORS##
        		tmpindsort = argsort(-splitVals)
        		splitVals = splitVals[tmpindsort]
        		splitVecs = splitVecs[:, tmpindsort]   # second axis !!

       		        if (nn > 3):
                	        splitVals = splitVals[:3]
                       		splitVecs = splitVecs[:, :3]

			if (len(splitVecs[0]) > 1):
				vecFiedler = splitVecs[:,1]
			else:
				vecFiedler = splitVecs
		left_indices = (vecFiedler < vecFiedler.mean()).nonzero()[0]
		right_indices = (vecFiedler >= vecFiedler.mean()).nonzero()[0]
		
		if ((min(len(left_indices),len(right_indices))) > 0): #ARPACK needs  matrix >=5 to get speigs
			lind = tree_size + 1
			rind = tree_size + 2
			cluster_treeLeft[ind] = lind
			cluster_treeRight[ind] = rind
			indices = IntList2array(indices)
			cluster_treeData.append(indices[left_indices])
			cluster_treeData.append(indices[right_indices])
			cluster_treeLeft.append(0)
			cluster_treeRight.append(0)
			cluster_treeLeft.append(0)
			cluster_treeRight.append(0)
			if (len(left_indices)==1):
				left_indices = concatenate((left_indices[0], left_indices[0]), axis=None)	
			if (len(right_indices)==1):
				right_indices = concatenate((right_indices[0], right_indices[0]), axis=None)
			cut = Xsplit[ix_(left_indices,right_indices)].sum()
			volume_left = Xsplit[ix_(left_indices,left_indices)].sum()
			volume_right = Xsplit[ix_(right_indices,right_indices)].sum()
			cluster_treeConduct.append(cut/min(volume_left,volume_right))
			cluster_treeConduct.append(cut/min(volume_left,volume_right))
	leaves = []
	leaveSize = []
	ci = zeros((n), int)
	if ((clusterSizeThreshold > 0) and (clusterSizeThreshold < 1)):
		clusterSizeThreshold  = around(clusterSizeThreshold*n);
	else:
		clusterSizeThreshold = around(clusterSizeThreshold);
	for i in range (0, len(cluster_treeData)):
		if ((cluster_treeLeft[i] == 0) and (cluster_treeRight[i] == 0)):
			if (len(leaves) == 0): 
				leaves = [i]
				ci[cluster_treeData[i]] = 1
			else:
				leaves = concatenate((leaves,i), axis=None)			
				ci[cluster_treeData[i]] = len(leaves)
#	print leaves  #Funny that makes an extra cicle?
	leaveSize = zeros((len(leaves)))
	for i in range (0,len(leaves)):
		leaveSize[i] = sum(ci == (i+1))
	
	idd = (leaveSize >= clusterSizeThreshold).nonzero()[0]
	csize = np.copy(leaveSize[idd])
	
	ci = zeros((n),int)
	conduct = zeros((len(idd)));

	for i in range (0, len(idd)):
		ci[cluster_treeData[leaves[idd[i]]]]=i
		conduct[i] = cluster_treeConduct[leaves[idd[i]]]
	return(ci, csize, specVals, specVecs, specGaps, conduct, cluster_treeData, cluster_treeConduct, cluster_treeLeft, cluster_treeRight)
	
def flowGrad(G, levelIdx, nodeInfoLevel, nodeInfoLevelSize, nodeInfoSet, nodeInfoFilter):
	numLevel = len(levelIdx)
	dG = triu(G);
	y=[]
	localMax = (where(dG.sum(axis=0)[0] == 0))[1].transpose()
	dd = zeros((len(G)), int);
	for i in range (0,len(localMax)):
		dd[localMax[i]] = len(nodeInfoSet[localMax[i]]);
	dG = dG + diag(dd)
	dG_inf=dG^numLevel
	for i in range (0,len(G)):
		y.append(where(dG_inf[:,i] > 0))
	dGdivsum = getIndexFromMatrix((1.0/dG.sum(axis=0)).transpose())
	MarkovT = dG * diag(dGdivsum)
	yLocalMax = localMax
	yGradFlow = dG
	yEquilibriumEQ = MarkovT**numLevel
	print "(OUTPUT) ", ("Number of local maxima: %d" % len(localMax))
	return(yLocalMax, yGradFlow, yEquilibriumEQ)
	
def optimumAssignment(X, cptEquilibriumEQ, nodeInfoLevelSize, nodeInfoSet, nodeInfoFilter, maximumAssignIterations):
	print "(OUTPUT) ", ("Finding Optimum Assignments")
	numMicro = max(X.shape[0], X.shape[1])
	numNode = len(nodeInfoSet)
	MacroStates = (where(diag(cptEquilibriumEQ)==1)[0]).transpose()
	numMacro = len(MacroStates)
	if numMacro == 1:
		print "(OUTPUT) ", ("SHC has found only one Macrostate! Noting to optimize...")
		ci = ones((numMicro), int)
		csize = numMicro
		fassign = []
		T = []
		Qmax = 1
		id_fuzzy = []
		return(ci, csize, fassign, T, Qmax, id_fuzzy)
		print "(OUTPUT) ", ("Optimum assignments Done...")
	distEQ = cptEquilibriumEQ[MacroStates,:]
	
	ci = zeros((numMicro), int);
	id_macro = []
	# deterministic assignment on basins of local maxima
	for i in range (0, numMacro):
		macroBasin = (where(distEQ[i,:] == 1)[1]).transpose()
		for j in range (0, len(macroBasin)): 
			if (j==0):
				id_macro.append(np.copy(nodeInfoSet[macroBasin[j]]))
				id_macro[i] = union1d(getIndexFromMatrix(id_macro[i]),getIndexFromMatrix(id_macro[i]))
			else:		
				id_macro[i] = union1d(id_macro[i],getIndexFromMatrix(nodeInfoSet[macroBasin[j]]))

		ci[id_macro[i]] = i+1   #Take care that ci is +1 since it maps microstates numbers, 0 is for fussy
	
	# fuzzy microstates on barrier
	id_fuzzy = (where(ci==0))[0]
	print "(OUTPUT) ", ("Number of barrier microstates: %d" % len(id_fuzzy))
	# Construct new transition count matrix from X
	T = eye((numMacro+len(id_fuzzy)), (numMacro+len(id_fuzzy)))
	T = T.tolil()
	
	Xdense = X.todense()
#	for i in range (0, numMacro):
#		row_id = where(ci==(i+1))
#		for j in range (i, numMacro):
#	#		print i, j
#			col_id = where(ci==(j+1))
#			T[i,j] = X[row_id,col_id].sum()
#		#print len(id_fuzzy)
#		for j in range (1, len(id_fuzzy)):
#	#		print i, j
#			tmpindx=array([id_fuzzy[j]])
#			T[i,j+numMacro] = X[row_id,tmpindx].sum()
			
	
	for i in range (0, numMacro):
		row_id = where(ci==(i+1))[0]
		for j in range (i, numMacro):
			col_id = where(ci==(j+1))[0]
			T[i,j] = Xdense[ix_(row_id,col_id)].sum()
		#print len(id_fuzzy)
		for j in range (1, len(id_fuzzy)):
			tmpindx=array([id_fuzzy[j],id_fuzzy[j]])
			T[i,j+numMacro] = Xdense[ix_(row_id,tmpindx)].sum()

	T = T + (triu(T,1)).transpose()
	T = T.todense()
#	print "(OUTPUT) SLOW 1"
	T[numMacro:(numMacro+len(id_fuzzy)),numMacro:(numMacro+len(id_fuzzy))] = Xdense[ix_(id_fuzzy,id_fuzzy)]
#	print "(OUTPUT) SLOW 2" 
	d = T.sum(axis=1)
	n = d.shape[0]
	dd = zeros((n,n))
	tmparray=[]
#	print "(OUTPUT) SLOW 3"
	for j in d:
		tmparray.append(1.0/j[0,0])
	dd[diag_indices(n)]= tmparray
#	print "(OUTPUT) SLOW 4"
#	dd = lil_matrix((n,n))
#	jj=0
#	for j in (d):
#		if (j[0,0] > 0):
#			dd[jj,jj]=(1.0/j[0,0])
#		else:
#			dd[jj,jj]= 0          #Is this correct? Why this could happen?
#		jj+=1
	dd = csc_matrix(dd)
	T = csc_matrix(T)
#	print "(OUTPUT) SLOW 5"
	M = T*dd
#	print "(OUTPUT) SLOW 6"
	Mp = M.todense()
#	print "(OUTPUT) SLOW 7"
#	print Mp.sum()
	eps = 1e-4 # small for tie-breaker
	fass = zeros((numMacro, (numMacro+len(id_fuzzy))))
#	print "(OUTPUT) SLOW 8"
	for i in range(0, numMacro):
#		print "(OUTPUT) SLOW 8a", i
		fass[i][i]=1
#	print "(OUTPUT) SLOW 9"
	fass[:,numMacro:] = Mp[:numMacro,numMacro:]
	
	iterN = 0
	fassign=[]
	id_domore=[]
	CI=[]
	CSIZ=[]
	Q=[]
	fassign.append(copy(fass))
	fass_sort = -fass
	id_fass_sort = fass_sort.argsort(axis=0)
	fass_sort.sort(axis=0)
	fass_sort= -fass_sort
	
	id_domore.append((where ((fass_sort[0,:] < eps) | ((fass_sort[0,:]-fass_sort[1,:])<eps)))[0])
	print "(OUTPUT) ", ("Number of empty assignments: %d" % len(id_domore[iterN]));
	CI.append(copy(ci))
	CI[iterN][id_fuzzy] = 1+(id_fass_sort[0,numMacro:])
	CSIZ.append(hiscMacro(CI[iterN]))
	Q.append(metastability(X,CI[iterN]))
        numMacro = ci.max(0)
        print "(OUTPUT) ", ("Num of macrostates: %d" % ci.max(0))
	print "(OUTPUT) ", ("Metastability (Q) = %.3f (%2.2f%%)" % (Q[iterN], (Q[iterN]/numMacro*100)))
	
	Qmax = Q[iterN]
	iter_max = iterN
	ci = np.copy(CI[iterN])
	csize = np.copy(CSIZ[iterN])

	fassigne =[]
	while ((id_domore[iterN].size>0) and (iterN < maximumAssignIterations)):
		iterN = iterN + 1
		print "(OUTPUT) ", ("*Iteration %d" % iterN)
                numMacro = ci.max(0)
                print "(OUTPUT) ", ("Number of macrostates: %d" % ci.max(0))
		Mp = Mp*M
		fass[:,id_domore[iterN-1]] = Mp[:numMacro,id_domore[iterN-1]]
		fass_sort = -fass
		id_fass_sort = fass_sort.argsort(axis=0)
		fass_sort.sort(axis=0)
		fass_sort= -fass_sort
		id_domore.append((where ((fass_sort[0,:] < eps) | ((fass_sort[0,:]-fass_sort[1,:])<eps)))[0])
		print "(OUTPUT) ", ("Number of empty assignment: %d" % len(id_domore[iterN]));
		# Method I (first-reach diffusion): find the optimal assignment 
		CI.append(copy(ci))
		CI[iterN][id_fuzzy] = 1+(id_fass_sort[0,numMacro:])
		CSIZ.append(hiscMacro(CI[iterN]))
		Q.append(metastability(X,CI[iterN]))
		print "(OUTPUT) ", ("(Q) I (first-reach) = \t%.3f (%2.2f%%)" % (Q[iterN], (Q[iterN]/numMacro*100)));
#		print 	Qmax,  Q[iterN]
		if (Qmax < Q[iterN]):
			Qmax = Q[iterN]
			iter_max = iterN
			ci = np.copy(CI[iterN])
			csize = np.copy(CSIZ[iterN])
#		print ci
		# Method II (all-iteration diffusion): rearrange the fuzzy assignment by the last iteration of Mp
		numMacro = ci.max(0)
		print "(OUTPUT) ", ("Number of macrostates: %d" % ci.max(0))
		fassign.append(copy(fass))      #Copy the array to avoid creating a pointer (THX Raymond)
		fassign[iterN][:,numMacro:] = Mp[:numMacro,numMacro:]
		fassign[iterN][:,id_domore[iterN]] = (ones((numMacro,len(id_domore[iterN])))/numMacro);
		F_rowsort = -fassign[iterN]
		id_rowsort = F_rowsort.argsort(axis=0)
		F_rowsort.sort(axis=0)
		F_rowsort = -F_rowsort
  		CI[iterN][id_fuzzy] = id_rowsort[0,numMacro:]
		CSIZ[iterN]=hiscMacro(CI[iterN])
		Q[iterN] = metastability(X,CI[iterN])
		print "(OUTPUT) ", ("(Q) II (all-iteration) = \t%.3f (%2.2f%%)" % (Q[iterN], (Q[iterN]/numMacro*100)));
		if (Qmax < Q[iterN]):
			Qmax = Q[iterN]
			iter_max = iterN
			ci = np.copy(CI[iterN])
			csize = np.copy(CSIZ[iterN])
#		print ci

	print "(OUTPUT) ", ("---- Maximal metastability reached at iteration %d: %f (%2.2f%%) ----\n" % (iter_max,Qmax,(Qmax/numMacro*100)))
        print "(OUTPUT) ", ("---- Final number of macrostates: %d ----\n" % ci.max(0))
	print "(OUTPUT) ", ("Optimum assignments Done...")
	return(ci, csize, fassign, T, Qmax, id_fuzzy)

	
	
def metastability(X,ci):
#Compute the metastability according to macro-clustering ci
	numMacro=max(ci);
	idX=[]
	for i in range(0,numMacro):
		idX.append(where(ci==(i+1))[0])
		if (len (idX[i]) == 1):
			idX[i] = [idX[i][0],idX[i][0]]
			
	QQ = zeros((numMacro,numMacro))
	for i in range(0,numMacro):
		for j in range(0,numMacro):
			QQ[i,j]=(X[ix_(idX[i],idX[j])].sum())
			QQ[j,i]=QQ[i,j]
	
	D = QQ.sum(axis=1)
	Q = (diag(diag(1./D)*QQ)).sum()	
	return(Q)
	
def hiscMacro(arr):				#Wrapper to Emulate matlab's --hisc-- function that counts the number of elements per class in a histogram
	hisc=zeros((max(arr)), int)  
	for i in (arr):
		hisc[i-1]+=1
	return (hisc)


def writeFlowGraph(cptGradFlow, nodeInfoLevel, nodeInfoLevelSize, nodeInfoSet, nodeInfoFilter, levelIdx, superLevels, outFlowGraphName, pLevels):
	print "(OUTPUT) ", ("---- Generating Macrostate flowgraph ---")
#	print "(DEBUG) ", scipy.linalg.norm(cptGradFlow - (cptGradFlow.conj().T))
	if ( scipy.linalg.norm(cptGradFlow - (cptGradFlow.conj().T))==0 ):
		print "(OUTPUT) ", ("error: Input graph is UNDIRECTED! I CANNOT GENERATE THE FLOW GRAPHIC!")
		return
	numNodes = max(shape(cptGradFlow))
	colorParam=[]
	sizeParam=[]
	for i in range(len(nodeInfoLevel)):
		colorParam.append(len(superLevels[1]) - nodeInfoFilter[i] - 1 )
		sizeParam.append(100.0*len(nodeInfoSet[i])/nodeInfoLevelSize[i])
#	printDebug(sizeParam, "sizeParam")
	
	maxColorParam=max(colorParam)
	colorScaleORI = arange(0.0,1.1,0.1)
	colorScaleNEW = arange(0.3,.91,0.06)
	print colorScaleORI, colorScaleNEW
	colorInterpolator = interp1d(colorScaleORI,colorScaleNEW)
	for i in range(numNodes):
		colorParam[i]= colorInterpolator(float(colorParam[i])/maxColorParam)
#	printDebug(colorParam, "colorParam")
	sParam = np.copy(sizeParam)

	levelColor = []
	cm = get_cmap('jet')
	for i in range(numNodes):
		tmpColor=cm(colorParam[i]) # color will now be an RGBA tuple, THX internet
		levelColor.append([int(tmpColor[0]*255),int(tmpColor[1]*255),int(tmpColor[2]*255),int(tmpColor[3]*255)])

	for i in range(len(sizeParam)):
	        sizeParam[i] = 0.1 + sizeParam[i]/max(sizeParam)

	outline = 'digraph "G" {\n'
	for i in range(numNodes):
               outline += ' node%d [label="%d:%2.0f%%", color="#%02x%02x%02x%02x",style=filled, shape=circle, width=%0.2f];\n' % (i, i, sParam[i], levelColor[i][0],levelColor[i][1],levelColor[i][2],levelColor[i][3], sizeParam[i])

#	printDebug(cptGradFlow, "cptGradFlow")
	for i in range(numNodes):
		connNodes = where(cptGradFlow[:,i] > 0)[0]
		for j in range(size(connNodes)):
			outline += ' node%d -> node%d [label="%d"];\n' % (i, connNodes[0, j],cptGradFlow[connNodes[0,j],i])


	levelSizes=[]
	for i in range(len(superLevels[1])):
		levelSizes.append(len(superLevels[0][i]))
	levelSizeInfo = ""
	for i in levelSizes:
		levelSizeInfo += '%d; ' % i;	

	l=zeros((len(levelIdx)), int)
	for i in range (len(levelIdx)):
		l[i] = len(levelIdx[i])
	l_end = l.cumsum(axis=0)

	tmpNextLevIdxInfo=0
	levelIdxInfo=""
	for i in range(0,len(l_end)-1):
		levelIdxInfo += "%d-%d; " % (tmpNextLevIdxInfo, l_end[i]-1)
		tmpNextLevIdxInfo=l_end[i]
	levelIdxInfo += "%d-%d; " % (tmpNextLevIdxInfo, l_end[len(l_end)-1])

	levelDesity=""
	for i in pLevels:
		levelDesity += "%2.0f%%; " % (i*100.0)

	
	outline += ' label = " Levels: %d \\l Density Levels: %s \\l Level Sizes: %s \\l Node Index: %s \\l\n' % (len(superLevels[1]), levelDesity, levelSizeInfo, levelIdxInfo)
	outline += ' labelloc="b";\nlabeljust="l";\n'
	outline += ' center = 1;\n overlap=scale;\n'
	outline +='}'
        print "(OUTPUT) ", ("Writting Macrostate flowgraph to: %s" % outFlowGraphName)
        f = open(outFlowGraphName,'w')
        print >>f, outline
        f.close()
        print "(OUTPUT) ", ("Macrostate flowgraph generated")
	return




def printDebug(obj, message):
	outline= ("(DEBUG) %s: " % message)
	try:
		for i in obj:
			outline += ", " + str(i)
	except TypeError:
		outline +=  " " + str(obj)
	print outline.replace("\n", " ")

if __name__ == '__main__':
 main()

