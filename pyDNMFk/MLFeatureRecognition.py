import numpy
import scipy
import os
from scipy.io import loadmat
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import h5py
from .utils import serialize_deserialize_mlp

class MLFeaturetools():
    def __init__(self,targetDir,clf,misVal=1,hitVal=6,appData=None,PropertyList=['minSilhouetteCoefficients', 'AIC', 'avgSilhouetteCoefficients']):
        """
        A tool that takes various silhouette parameters from the pyDNMFk and estimates the latent features using a MLP.
        Implementation based on "A neural network for determination of latent dimensionality in non-negative matrix factorization".
        Args:
            targetDir (str) : directory where the pyDNMFk results are saved into hdf5 files
            clf (object) : MLP classifier to be applied
            misVal (int) :  value added when the model doesn't think the correct no. of features is inside the range
            hitVal (int) :  value added when the model does think the correct no. of features is inside the range
            appData (dict) : dictionary of parameters
            PropertyList (list) : list of features to be trained on NN so that the prediction is made upon these
        """

        self.targetDir = targetDir
        self.PropertyList=PropertyList
        self.appData = {} if appData is None else appData
        self.clf = clf
        self.misVal = misVal
        self.hitVal = hitVal




    def buildStatistics(self):
        """Build the statistics for NN prediction"""
        intList = []
        statListN = []
        statList = ['AIC', 'L_errDist', 'avgErr', 'avgSilhouetteCoefficients', 'clusterSilhouetteCoefficients']
        for curVal in os.listdir(self.targetDir):
            try:
                curInt = int(curVal)
                intList.append(curInt)
            except:
                pass
        intList.sort()
        self.appData['k'] = numpy.array(intList)
        nPts = self.appData['k'].shape[0]
        maxK = numpy.max(self.appData['k'])
        self.appData['minSilhouetteCoefficients'] = numpy.zeros(nPts)

        for curStat in statList:
            if curStat == 'clusterSilhouetteCoefficients':
                self.appData[curStat] = numpy.zeros([nPts, maxK])
            else:
                self.appData[curStat] = numpy.zeros(nPts)

        for ind, curK in enumerate(self.appData['k']):
            curF = h5py.File(self.targetDir + '/' + str(curK) + '/' + 'results.h5','r')
            for curStat in statList:
                if curStat == 'clusterSilhouetteCoefficients':
                    self.appData[curStat][ind, :curK] = numpy.array(curF[curStat])
                    self.appData['minSilhouetteCoefficients'][ind] = numpy.min(numpy.array(curF[curStat]))
                else:
                    self.appData[curStat][ind] = float(numpy.array(curF[curStat]))
            curF.close()

        self.appData['AIC'] = (self.appData['AIC'] - numpy.min(self.appData['AIC'])) / numpy.max(
            self.appData['AIC'] - numpy.min(self.appData['AIC']))


    def predictStatistics(self):
        """Predict the latent feature count based on pyDNMFk statistics"""
        if not any(self.appData):
           print('loading the contents from the hdf5 files.')
           self.buildStatistics()

        MLwin = 7
        initFeat = self.appData['k'][0]
        maxK = self.appData['k'].shape[0]
        npreds = maxK - MLwin
        fetArray = []
        for curPred in range(npreds):
            fetArray.append(numpy.concatenate([self.appData[curP][curPred:curPred + MLwin] for curP in self.PropertyList]))
        fetArray = numpy.array(fetArray)
        preds = self.clf.predict(fetArray)
        preds = numpy.array(preds, dtype='int64')
        counts = numpy.zeros(npreds, dtype='int64')
        nHits = numpy.zeros_like(counts)
        for curI in range(npreds):
            if preds[curI] == 6.0:
                counts[curI + 6:] += self.misVal
            elif preds[curI] == 0.0:
                counts[:(curI + 1)] += self.misVal
            elif curI + preds[curI] < npreds:
                counts[curI + preds[curI]] += self.hitVal
                nHits[curI + preds[curI]] += 1

        predF = numpy.nonzero(numpy.array(counts) == numpy.array(counts).max())[0][-1] + initFeat
        return predF


