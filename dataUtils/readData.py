import pandas as pd
import numpy as np
import re
import dataUtils.data_utils as du
import dataUtils.dbDataPreprocess as dbc
import models.instruments as inst
import os.path
import pdb


class DataHandler(object):
    def __init__(self, dataFileName='../../data/AH_vol.npy', testDataPercentage=0.2):
        self.endOfSeriesCount = 0
        self.prevIrStartPosition = -1
        self.prevVolStartPosition = -1
        self.prevVolDepthPosition = -1
        self.prevIrDepthPosition = -1
        self.prevWidthStopPosition = -1
        self.prevWidthStartPosition = -1
        self.trainData = None
        self.volatilities = None
        self.ir = None
        self.params = None
        self.segmentWidth = None
        self.batchSize = 10
        self.dataFileName = dataFileName
        self.modes = ['vol', 'ir', 'params']
        self.filePrefix = None
        self.volDepth = 3
        self.irDepth = 1
        self.sliding = True
        self.testDataPercentage = testDataPercentage
        self.splitBooleanIndex = []
        self.testData = {}
        self.inputSegments = []
        self.outputSegments = []
        self.lastBatchPointer = -1

    def readH5(self, fileName):
        raise NotImplemented()

    def getTestData(self):
        assert(len(self.testData)>0),"Test data not present"
        return self.testData["input"], self.testData['output']

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
        self.splitTestData()
        self._setChunkedProps(batchSize, volDepth, irDepth, sliding)

        return 0

    def splitTestData(self):
        width = self.volatilities.shape[1]
        testDataPointer = int(np.floor(self.testDataPercentage * width))
        testStart = width - testDataPointer
        self.testData['vol'] = self.volatilities[:, testStart: width]
        self.testData['ir'] = self.params[:, testStart: width]
        self.volatilities = np.delete(self.volatilities, np.arange(testStart, width), axis=1)
        self.ir = np.delete(self.ir, np.arange(testStart, width), axis=1)
        self.params = np.delete(self.params, np.arange(testStart, width), axis=1)

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

    def _setChunkedProps(self, batchSize, segmentWidth=50, volDepth=3, irDepth=1, sliding=True):
        if (volDepth == -1):
            self.volDepth = self.volatilities.shape[0]
        if (irDepth == -1):
            self.irDepth = self.ir.shape[0]
        if(segmentWidth <= self.volatilities.shape[1]):
            self.segmentWidth = segmentWidth
        self.sliding = sliding
        self.batchSize = batchSize

    def preprocess(self, x_train, y_train, x_test, y_test):
        raise NotImplemented()

    def getNextBatch(self, batchSize=None, width=None, volDepth=None, irDepth=None):
        if (self.lastBatchPointer == -1):
            volD = self.volDepth
            if (volDepth is not None):
                volD = volDepth
            irD = self.irDepth
            if (irDepth is not None):
                irD = irDepth
            if(width is None):
                width = self.segmentWidth
            if(len(self.inputSegments)== 0 or self.segmentWidth != width):
                self._segmentDataset(width, volDepth, irDepth)
                self.segmentWidth = width
            if(batchSize is None):
                batchSize = self.batchSize
            pdb.set_trace()
            self.inputSegments = self.inputSegments.reshape((len(self.inputSegments), 1, width, self.inputSegments.shape[2]))
            self.splitBooleanIndex = np.random.rand(len(self.inputSegments)) < (1 - self.testDataPercentage)
            trainX = self.reshapedSegments[self.splitBooleanIndex]
            trainY = self.outputSegments[self.splitBooleanIndex]
            testX = self.reshapedSegments[~self.splitBooleanIndex]
            testY = self.outputSegments[~self.splitBooleanIndex]
        else:
            self.lastBatchPointer = (self.lastBatchPointer + 1) % self.inputSegments.shape[0]
            trainX = self.reshapedSegments[self.splitBooleanIndex]
            trainY = self.outputSegments[self.splitBooleanIndex]
            testX = self.reshapedSegments[~self.splitBooleanIndex]
            testY = self.outputSegments[~self.splitBooleanIndex]

        return trainX, trainY, testX, testY

    def _segmentDataset(self, width, volDepth, irDepth):
        inSegments = np.empty((0, width, volDepth + irDepth))
        targetSegments = np.empty((0, len(self.params[0])))
        #bad memory handling, we could only keep coordinates of each chunk
        while(True):
            vol, ir, params, traversedDataset = self._buildBatch(width, volDepth, irDepth)
            temp = np.column_stack((vol.T,ir.T))
            #reshape to (1,width,#channels)
            temp = temp.reshape((1,temp.shape[0],temp.shape[1]))
            inSegments=np.vstack((inSegments,temp))
            targetSegments =np.vstack((targetSegments, params))
            if(traversedDataset):
                break

        self.inputSegments = inSegments
        self.outputSegments = targetSegments


    def _buildBatch(self, width, volDepth, irDepth):
        irDepthEnd = False
        volDepthEnd = False
        seriesEnd = False
        traversedFullDataset = False

        if (self.endOfSeriesCount == 1):
            self.prevWidthStopPosition = -1
            self.prevWidthStartPosition = -1
            seriesEnd = True
            self.endOfSeriesCount = 0

        step = 1
        if (not self.sliding or self.prevWidthStopPosition == -1):
            step = width # this is a bit confusing and adds complexity -> simplify

        startWidthPosition, endWidthPosition, widthEndFlag = \
            self._checkLimits(self.prevWidthStartPosition, self.prevWidthStopPosition, step, self.volatilities.shape[1])

        if (widthEndFlag):
            self.endOfSeriesCount += 1

        if (not seriesEnd and self.prevVolDepthPosition != -1):
            volStartPosition = self.prevVolStartPosition
            volStopPosition = self.prevVolDepthPosition
        else:
            volStartPosition, volStopPosition, volDepthEnd = \
                self._checkLimits(self.prevVolStartPosition, self.prevVolDepthPosition, volDepth,
                                  self.volatilities.shape[0])

        if (not seriesEnd and self.prevIrDepthPosition != -1):
            irStartPosition = self.prevIrStartPosition
            irStopPosition = self.prevIrDepthPosition
        else:
            irStartPosition, irStopPosition, irDepthEnd = \
                self._checkLimits(self.prevIrStartPosition, self.prevIrDepthPosition, irDepth, self.ir.shape[0])

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

        if(seriesEnd and irDepthEnd and volDepthEnd):
            self.endOfSeriesCount = 0
            self.prevIrStartPosition = -1
            self.prevVolStartPosition = -1
            self.prevVolDepthPosition = -1
            self.prevIrDepthPosition = -1
            self.prevWidthStopPosition = -1
            self.prevWidthStartPosition = -1
            traversedFullDataset = True

        # Add test data
        return volData, irData, params, traversedFullDataset

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
            startPosition = limit - (prevStopPosition - prevStartPosition)
            # else:
            #     startPosition = prevStartPosition
            endFlag = True  # to keep ir and vol windows until the whole series is traversed
            endPosition = limit
        else:
            startPosition = prevStartPosition + additive
            endPosition = newPosition

        return startPosition, endPosition, endFlag
