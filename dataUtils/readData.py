import pandas as pd
import numpy as np
import re
import dataUtils.data_utils as du
import dataUtils.dbDataPreprocess as dbc
import models.instruments as inst
import os.path
import pdb


class DataHandler(object):
    def __init__(self, dataFileName='../../data/AH_vol.npy', testDataPercentage=0.2, batchSize=50, width=50, volDepth=3,
                 irDepth=2, sliding=True):
        self.modes = ['vol', 'ir', 'params']
        self.dataFileName = dataFileName
        self.batchSize = batchSize
        self.segmentWidth = width
        self.volDepth = volDepth
        self.irDepth = irDepth
        self.sliding = sliding
        self.testDataPercentage = testDataPercentage
        self.splitBooleanIndex = []
        self.testData = {}

        self.endOfSeriesCount = 0
        self.prevIrStartPosition = -1
        self.prevVolStartPosition = -1
        self.prevVolDepthPosition = -1
        self.prevIrDepthPosition = -1
        self.prevWidthStopPosition = -1
        self.prevWidthStartPosition = -1
        self.volatilities = None
        self.ir = None
        self.params = None
        self.filePrefix = None
        self.inputSegments = []
        self.outputSegments = []
        self.reshapedSegments = []
        self.lastBatchPointer = -1

    def readH5(self, fileName):
        raise NotImplemented()

    def getTestData(self, batchSize=None, width=None, volDepth=None, irDepth=None):
        if (len(self.testData) == 0):
            pdb.set_trace()
            batchSize, width, volDepth, irDepth = self._checkFuncInput(batchSize, width, volDepth, irDepth)
            self.splitTestData(batchSize=batchSize, width=width, volDepth=volDepth, irDepth=irDepth)
        assert (len(self.testData) > 0), "Test data not present"
        return self.testData["input"], self.testData['output']

    def getTrainingData(self, batchSize, depth, clean=False):
        '''
        :param batchSize: size of training batch
        :param depth: number of
        :return:
        '''
        raise NotImplemented()

    def readData(self, fileName, batchSize=None, volDepth=3, irDepth=1, sliding=True, twinFile=True, clean=False):
        if (fileName is None):
            fileName = self.dataFileName
        if (batchSize is None):
            batchSize = self.batchSize

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
                pdb.set_trace()
                self._setDataFiles(np.load(path), mode)
        elif fileType.lower() == 'h5':
            raise NotImplemented()

        if (self.params is None):
            return 1
        # self._setChunkedProps(batchSize, volDepth, irDepth, sliding)

        return 0

    def splitTestData(self, batchSize=None, width=None, volDepth=None, irDepth=None):
        batchSize, width, volDepth, irDepth = self._checkFuncInput(batchSize, width, volDepth, irDepth)
        if (self.volatilities is None):
            self.readData(self.dataFileName)
        if (len(self.inputSegments) == 0 or self.segmentWidth != width):
            self._segmentDataset(width, volDepth, irDepth)
            self.segmentWidth = width
            self.reshapedSegments = []
        pdb.set_trace()
        self.splitBooleanIndex = np.random.rand(len(self.inputSegments)) < (1 - self.testDataPercentage)
        if (len(self.reshapedSegments) == 0):
            self.reshapedSegments = self.inputSegments.reshape(
                (len(self.inputSegments), 1, width, self.inputSegments.shape[2]))
        self.testData['input'] = self.reshapedSegments[~self.splitBooleanIndex]
        self.testData['output'] = self.outputSegments[~self.splitBooleanIndex]

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

    # def _setChunkedProps(self, batchSize, segmentWidth=50, volDepth=3, irDepth=1, sliding=True):
    #     if (volDepth == -1):
    #         self.volDepth = self.volatilities.shape[0]
    #     if (irDepth == -1):
    #         self.irDepth = self.ir.shape[0]
    #     if (segmentWidth <= self.volatilities.shape[1]):
    #         self.segmentWidth = segmentWidth
    #     self.sliding = sliding
    #     self.batchSize = batchSize

    def preprocess(self, x_train, y_train, x_test, y_test):
        raise NotImplemented()

    def getNextBatch(self, batchSize=None, width=None, volDepth=None, irDepth=None):
        if (self.lastBatchPointer == -1):
            batchSize, width, volDepth, irDepth = self._checkFuncInput(batchSize, width, volDepth, irDepth)
            if (len(self.inputSegments) == 0 or self.segmentWidth != width):
                self._segmentDataset(width, volDepth, irDepth)
                self.segmentWidth = width
                self.reshapedSegments = []
            pdb.set_trace()
            if (len(self.reshapedSegments) == 0):
                self.reshapedSegments = self.inputSegments.reshape(
                    (len(self.inputSegments), 1, width, self.inputSegments.shape[2]))
            trainX = self.reshapedSegments[self.splitBooleanIndex]
            trainY = self.outputSegments[self.splitBooleanIndex]
        else:
            self.lastBatchPointer = (self.lastBatchPointer + 1) % self.inputSegments.shape[0]
            trainX = self.reshapedSegments[self.splitBooleanIndex]
            trainY = self.outputSegments[self.splitBooleanIndex]

        return trainX, trainY

    def _segmentDataset(self, width, volDepth, irDepth):
        inSegments = np.empty((0, width, volDepth + irDepth))
        targetSegments = np.empty((0, self.params.shape[0]))
        # bad memory handling, we could only keep coordinates of each chunk
        while (True):
            vol, ir, params, traversedDataset = self._buildBatch(width, volDepth, irDepth)
            temp = np.column_stack((vol.T, ir.T))
            # reshape to (1,width,#channels)
            temp = temp.reshape((1, temp.shape[0], temp.shape[1]))
            inSegments = np.vstack((inSegments, temp))
            targetSegments = np.vstack((targetSegments, params))
            if (traversedDataset):
                break

        self.inputSegments = inSegments
        self.outputSegments = targetSegments

    def _buildBatch(self, width, volDepth, irDepth, pointers=True):
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
            step = width  # this is a bit confusing and adds complexity -> simplify

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

        self.prevIrDepthPosition = irStopPosition
        self.prevVolDepthPosition = volStopPosition
        self.prevVolStartPosition = volStartPosition
        self.prevIrStartPosition = irStartPosition
        self.prevWidthStopPosition = endWidthPosition
        self.prevWidthStartPosition = startWidthPosition

        if (seriesEnd and irDepthEnd and volDepthEnd):
            self.endOfSeriesCount = 0
            self.prevIrStartPosition = -1
            self.prevVolStartPosition = -1
            self.prevVolDepthPosition = -1
            self.prevIrDepthPosition = -1
            self.prevWidthStopPosition = -1
            self.prevWidthStartPosition = -1
            traversedFullDataset = True
            print(volStartPosition, volStopPosition, startWidthPosition, endWidthPosition, irStartPosition,
                  irStopPosition,
                  widthEndFlag)

        if (pointers):
            volData = (volStartPosition, volStopPosition, startWidthPosition, endWidthPosition)
            irData = (irStartPosition, irStopPosition, startWidthPosition, endWidthPosition)
            params = endWidthPosition - 1
        else:
            volData, irData, params = self._getActualData(volStartPosition, volStopPosition, irStartPosition, irStopPosition,
                                          startWidthPosition, endWidthPosition)

        return volData, irData, params, traversedFullDataset

    def _getActualData(self, volStart, volStop, irStart, irStop, widthStart, widthStop):
        volData = self.volatilities[volStart:volStop, widthStart:widthStop]
        irData = self.ir[irStart:irStop, widthStart:widthStop]
        params = self.params[:, widthStop - 1]

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
            startPosition = limit - (prevStopPosition - prevStartPosition)
            # else:
            #     startPosition = prevStartPosition
            endFlag = True  # to keep ir and vol windows until the whole series is traversed
            endPosition = limit
        else:
            startPosition = prevStartPosition + additive
            endPosition = newPosition

        return startPosition, endPosition, endFlag

    def _checkFuncInput(self, batchSize, width, volDepth, irDepth):
        if (batchSize is None):
            batchSize = self.batchSize
        if (width is None):
            width = self.segmentWidth
        if (volDepth is None):
            volDepth = self.volDepth
        if (irDepth is None):
            irDepth = self.irDepth

        assert (batchSize is not None), "Batch size not set"
        assert (width is not None), "Width not set"
        assert (volDepth is not None), "Volatility depth not set"
        assert (irDepth is not None), "Interest rate depth not set"
        return batchSize, width, volDepth, irDepth
