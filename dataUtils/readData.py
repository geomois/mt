import pandas as pd
import numpy as np
import re
import dataUtils.data_utils as du
import dataUtils.dbDataPreprocess as dbc
import models.instruments as inst
import os.path
import pdb


class dataHandler(object):
    def __init__(self, dataFileName='../../data/vol_train.npy'):
        self.endOfSeriesCount = 0
        self.prevIrStartPosition = -1
        self.prevVolStartPosition = -1
        self.prevVolDepthPosition = -1
        self.prevIrDepthPosition = -1
        self.prevWidthStopPosition = -1
        self.prevWidthStartPosition = -1
        self.trainData = None
        self.testData = None
        self.volatilities = None
        self.ir = None
        self.params = None
        self.currentBatch = None
        self.dataFileName = dataFileName
        self.modes = ['vol', 'ir', 'params']
        self.filePrefix = None
        self.volDepth = 3
        self.irDepth = 1
        self.sliding = True

    def readH5(self, fileName):
        raise NotImplemented()

    def getTestData(self):
        raise NotImplemented()

    def getTrainingData(self, batchSize, depth, clean=False):
        '''
        :param batchSize: size of training batch
        :param depth: number of
        :return:
        '''
        raise NotImplemented()

    def readData(self, fileName, batchSize, volDepth=3, irDepth=1, sliding=True, twinFile=True, clean=False):
        if (fileName is None):
            fileName = self.dataFileName

        name, prefix, fileType, mode, rest = dbc.breakPath(fileName)  # rest=[] without '.'
        fileList = [(fileName, mode)]
        self.filePrefix = prefix + ''.join(rest)
        if twinFile:
            currentMode = mode
            for mode in self.modes:
                if (mode != currentMode):
                    path = str(self.filePrefix + mode + '.' + fileType)
                    if (os.path.isfile(path=path)):
                        fileList.append((path, mode))

        if fileType.lower() == 'csv':
            for path, mode in fileList:
                if (clean):
                    # pdb.set_trace()
                    df, npy = dbc.cleanCsv(path, mode=mode, toNNData=True, exportPath=True, dbFormat=True)
                else:
                    df, npy = dbc.cleanCsv(path, mode=mode, toNNData=True, exportPath=True, dbFormat=False)
                self._setDataFiles(npy, mode)
        elif fileType.lower() == 'npy':
            for path, mode in fileList:
                self._setDataFiles(np.load(path), mode)
        elif fileType.lower() == 'h5':
            raise NotImplemented()

        if (self.params is None):
            return 1

        self._setChunkedProps(batchSize, volDepth, irDepth, sliding)

        return 0

    def createCalibrateParams(self, swoFile, irFile, modelMap, currency, irType):
        swo = inst.get_swaptiongen(modelMap=modelMap, currency=currency, irType=irType, volFileName=swoFile,
                                   irFileName=irFile)
        params = swo.calibrate_history(csvFilePath=self.filePrefix + 'params.csv', fileName=None)
        # pdb.set_trace()
        if ("b'Date'" in params.columns):
            params = params.rename(index=str, columns={"b'Date'": 'Date'})
        if ("OrigParam0" in params.columns):
            params = params.rename(index=str, columns={"OrigParam0'": 'Alpha'})
        if ("OrigParam1" in params.columns):
            params = params.rename(index=str, columns={"OrigParam1'": 'Sigma'})

        return params

    def _setDataFiles(self, data, mode):
        if (mode.lower() == 'vol'):
            self.volatilities = data
        elif (mode.lower() == 'ir'):
            self.ir = data
        elif (mode.lower() == 'params'):
            self.params = data

    def _setChunkedProps(self, batchSize, volDepth=3, irDepth=1, sliding=True):
        if (volDepth == -1):
            self.volDepth = self.volatilities.shape[0]
        if (irDepth == -1):
            self.irDepth = self.ir.shape[0]
        self.sliding = sliding
        if (self.volatilities.shape[1] > batchSize):
            self.batchSize = batchSize
            # self.getNextBatch()
        else:
            raise Exception()

    def preprocess(self, x_train, y_train, x_test, y_test):
        raise NotImplemented()

    def getNextBatch(self, batchSize=None, volDepth=None, irDepth=None):
        irDepthEnd = False
        volDepthEnd = False
        seriesEnd = False
        volD = self.volDepth
        if (volDepth is not None):
            volD = volDepth
        irD = self.irDepth
        if (irDepth is not None):
            irD = irDepth
        batchS = self.batchSize
        if (batchSize is not None):
            batchS = batchSize

        if (self.endOfSeriesCount == 1):
            self.prevWidthStopPosition = -1
            self.prevWidthStartPosition = -1
            seriesEnd = True
            self.endOfSeriesCount = 0

        step = 1
        if (not self.sliding or self.prevWidthStopPosition == -1):
            step = batchS

        startWidthPosition, endWidthPosition, widthEndFlag = \
            self._checkLimits(self.prevWidthStartPosition, self.prevWidthStopPosition, step, self.volatilities.shape[1])

        if (widthEndFlag):
            self.endOfSeriesCount += 1

        if (not seriesEnd and self.prevVolDepthPosition != -1):
            volStartPosition = self.prevVolStartPosition
            volStopPosition = self.prevVolDepthPosition
        else:
            volStartPosition, volStopPosition, volDepthEnd = \
                self._checkLimits(self.prevVolStartPosition, self.prevVolDepthPosition, volD,
                                  self.volatilities.shape[0])

        if (not seriesEnd and self.prevIrDepthPosition != -1):
            irStartPosition = self.prevIrStartPosition
            irStopPosition = self.prevIrDepthPosition
        else:
            irStartPosition, irStopPosition, irDepthEnd = \
                self._checkLimits(self.prevIrStartPosition, self.prevIrDepthPosition, irD, self.ir.shape[0])

        volData = self.volatilities[volStartPosition:volStopPosition, startWidthPosition:endWidthPosition]
        irData = self.ir[irStartPosition:irStopPosition, startWidthPosition:endWidthPosition]
        params = self.params[:, endWidthPosition - 1]
        print(volStartPosition, volStopPosition, startWidthPosition, endWidthPosition, irStartPosition, irStopPosition,
              widthEndFlag)
        self.prevIrDepthPosition = irStopPosition
        self.prevVolDepthPosition = volStopPosition
        self.prevVolStartPosition = volStartPosition
        self.prevIrStartPosition = irStartPosition
        self.prevWidthStopPosition = endWidthPosition
        self.prevWidthStartPosition = startWidthPosition

        #Add test data
        return volData, irData, params

    def _checkLimits(self, prevStartPosition, prevStopPosition, step, limit):
        endFlag = False
        if (step > limit):
            raise IndexError()
        # pdb.set_trace()
        if (prevStopPosition == -1):  # init value -1
            prevStopPosition = 0
            prevStartPosition = 0
            newPosition = step
            additive = 0  # to include batchSize increment
        else:
            newPosition = prevStopPosition + step
            additive = step
        if (newPosition >= limit):
            # overLimit = limit - newPosition
            pdb.set_trace()
            startPosition = limit - (prevStopPosition - prevStartPosition)
            # else:
            #     startPosition = prevStartPosition
            endFlag = True  # to keep ir and vol windows until the whole series is traversed
            endPosition = limit
        else:
            startPosition = prevStartPosition + additive
            endPosition = newPosition

        return startPosition, endPosition, endFlag
