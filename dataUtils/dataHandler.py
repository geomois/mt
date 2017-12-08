import numpy as np
import re
import dataUtils.dbDataPreprocess as dbc
import models.instruments as inst
import os.path
import pdb
import datetime as dt


class DataHandler(object):
    def __init__(self, dataFileName='data/toyData/AH_vol.npy'
                 , testDataPercentage=0.2, batchSize=50, width=50, volDepth=3,
                 irDepth=2, sliding=True, useDataPointers=True, randomSplit=False, datePointer=False, save=False):
        self.modes = ['vol', 'ir', 'params']
        self.dataFileName = dataFileName
        self.batchSize = batchSize
        self.segmentWidth = width
        self.volDepth = volDepth
        self.irDepth = irDepth
        self.sliding = sliding
        self.testDataPercentage = testDataPercentage
        self.splitBooleanIndex = []
        self.testData = {"input": [], "output": []}
        self.trainData = {"input": [], "output": []}
        self.useDataPointers = useDataPointers
        self.trainIndices = []
        self.datePointer = datePointer

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
        self.dataPointers = {"vol": [], "ir": [], "params": []}
        self.lastBatchPointer = -1
        self.saveProcessedData = save
        self.delegatedFromFile = False
        self.runId = None
        self.randomSpliting = randomSplit
        self._getCurrentRunId()

    def readH5(self, fileName):
        raise NotImplemented()

    def getTestData(self, batchSize=None, width=None, volDepth=None, irDepth=None):
        if (len(self.testData["input"]) == 0):
            # pdb.set_trace()
            batchSize, width, volDepth, irDepth = self._checkFuncInput(batchSize, width, volDepth, irDepth)
            self.splitTestData(batchSize=batchSize, width=width, volDepth=volDepth, irDepth=irDepth)
        assert (len(self.testData["input"]) > 0 and len(self.testData["output"]) > 0), "Test data not present"
        if (self.saveProcessedData):
            suffix = 'test' + str(self.batchSize) + "_w" + str(self.segmentWidth) + '_' + str(
                self.volDepth) + '_' + str(self.irDepth)
            self._saveProcessedData(suffix, 'test')

        return np.asarray(self.testData["input"]), np.asarray(self.testData['output'])

    def readData(self, batchSize=None, twinFile=True, clean=False):
        batchSize, _, _, _ = self._checkFuncInput(batchSize)
        name, prefix, fileType, mode, rest = dbc.breakPath(self.dataFileName)  # rest=[] without '.'
        fileList = [(self.dataFileName, mode)]
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
                # pdb.set_trace()
                self._setDataFiles(np.load(path), mode)
        elif fileType.lower() == 'h5':
            raise NotImplemented()

        if (self.params is None):
            return 1
        # self._setChunkedProps(batchSize, volDepth, irDepth, sliding)

        return 0

    def delegateDataDictsFromFile(self, fileList):
        self.delegatedFromFile = True
        self.useDataPointers = False
        for i in range(len(fileList)):
            file = fileList[i]
            if ("test" in file):
                dataDict = self.testData
            else:
                dataDict = self.trainData

            mode = ""
            if ("input" in file):
                mode = "input"
            elif ("output" in file):
                mode = "output"

            try:
                if (len(mode) > 0):
                    dataDict[mode] = np.load(file)
                else:
                    raise FileNotFoundError
            except:
                raise Exception()

    def splitTestData(self, batchSize=None, width=None, volDepth=None, irDepth=None):
        batchSize, width, volDepth, irDepth = self._checkFuncInput(batchSize, width, volDepth, irDepth)
        if (self.volatilities is None):
            self.readData(self.dataFileName)
        if (len(self.inputSegments) == 0 or self.segmentWidth != width):
            self._segmentDataset(width, volDepth, irDepth, self.useDataPointers)
            self.segmentWidth = width
        dataLength = len(self.inputSegments)
        if (self.useDataPointers):
            dataLength = len(self.dataPointers["vol"])

        if (self.randomSpliting):
            pdb.set_trace()
            self.splitBooleanIndex = np.random.rand(dataLength) < (1 - self.testDataPercentage)
        else:
            self.splitBooleanIndex = (np.zeros((int(np.floor(dataLength * (1 - self.testDataPercentage))))) == 0)
            temp = np.zeros((dataLength - self.splitBooleanIndex.shape[0])) != 0
            self.splitBooleanIndex = np.append(self.splitBooleanIndex, temp)

        if (self.useDataPointers):
            testIndices = np.where(self.splitBooleanIndex == False)[0]
            inPut, outPut = self.reshapeFromPointers(width, volDepth=volDepth, irDepth=irDepth, indices=testIndices,
                                                     nbBatches=np.where(self.splitBooleanIndex == False)[0].shape[0],
                                                     train=False)
        else:
            # pdb.set_trace()
            inPut = self.inputSegments[~self.splitBooleanIndex]
            outPut = self.outputSegments[~self.splitBooleanIndex]

        if (self.datePointer and not self.randomSplitting):
            firstTestPoint = int(np.floor(dataLength * (1 - self.testDataPercentage)))
            self.testData['output'] = np.int32(np.arange(firstTestPoint, firstTestPoint + len(self.testData['output'])))
        else:
            self.testData['output'] = outPut
        self.testData['input'] = inPut

    def getNextBatch(self, batchSize=None, width=None, volDepth=None, irDepth=None, pipeline=None, randomDraw=False):
        batchSize, width, volDepth, irDepth = self._checkFuncInput(batchSize, width, volDepth, irDepth)
        if (self.lastBatchPointer == -1):
            self.lastBatchPointer = 0
            if (self.delegatedFromFile):
                modulo = len(self.trainData["input"])
                if (pipeline is not None):
                    # pdb.set_trace()
                    self.trainData["output"] = pipeline.fit_transform(self.trainData["output"])
            else:
                if ((len(self.inputSegments) == 0 and len(
                        self.dataPointers['vol']) == 0) or self.segmentWidth != width):
                    self._segmentDataset(width, volDepth, irDepth, pointers=self.useDataPointers)
                    self.segmentWidth = width
                self.trainIndices = np.where(self.splitBooleanIndex == True)[0]
                if (self.useDataPointers):
                    inPut, outPut = self.reshapeFromPointers(width, volDepth, irDepth, indices=self.trainIndices,
                                                             startPointer=0,
                                                             nbBatches=batchSize, train=True)
                    self.trainData["input"] = inPut
                    self.trainData["output"] = outPut
                    modulo = len(self.dataPointers['vol'])
                else:
                    self.trainData["input"] = self.inputSegments[self.splitBooleanIndex]
                    self.trainData["output"] = self.outputSegments[self.splitBooleanIndex]
                    if (pipeline is not None):
                        # pdb.set_trace()
                        self.trainData["output"] = pipeline.fit_transform(self.trainData["output"])
                    modulo = len(self.inputSegments)
            if (randomDraw):
                indices = np.random.randint(0, high=len(self.trainData["input"]), size=batchSize)
                trainX = self.trainData["input"][indices]
                trainY = self.trainData["output"][indices]
            else:
                trainX = self.trainData["input"][self.lastBatchPointer:batchSize]
                trainY = self.trainData["output"][self.lastBatchPointer:batchSize]
        else:
            if (self.useDataPointers):
                modulo = len(self.trainIndices)
                if (self.lastBatchPointer >= len(self.trainData["input"])):
                    batchSize, width, volDepth, irDepth = self._checkFuncInput(batchSize, width, volDepth, irDepth)
                    inPut, outPut = self.reshapeFromPointers(width, volDepth, irDepth, indices=self.trainIndices,
                                                             startPointer=self.lastBatchPointer,
                                                             nbBatches=batchSize, train=True)
                    self.trainData["input"] = np.vstack((self.trainData["input"], inPut))
                    self.trainData["output"] = np.vstack((self.trainData["output"], outPut))
                if (len(self.trainData["input"]) == len(self.dataPointers["vol"]) and self.saveProcessedData):
                    suffix = 'train' + str(self.batchSize) + "_w" + str(self.segmentWidth) + '_' + str(
                        self.volDepth) + '_' + str(self.irDepth)
                    self._saveProcessedData(suffix, 'train')
            else:
                modulo = len(self.trainData["input"])
            if (randomDraw):
                indices = np.random.randint(0, high=len(self.trainData["input"]), size=batchSize)
                trainX = self.trainData["input"][indices]
                trainY = self.trainData["output"][indices]
            else:
                trainX = self.trainData["input"][self.lastBatchPointer: self.lastBatchPointer + batchSize]
                trainY = self.trainData["output"][self.lastBatchPointer: self.lastBatchPointer + batchSize]

        self.lastBatchPointer = (self.lastBatchPointer + batchSize) % modulo

        return np.asarray(np.float32(trainX)), np.asarray(np.float32(trainY))

    def fitPipeline(self, pipeline):
        if (self.useDataPointers and len(self.trainData['output']) == 0):
            self.trainData["output"] = self.outputSegments[self.splitBooleanIndex]
        pipeline.fit_transform(self.trainData["output"])
        return pipeline

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

    def preprocess(self, x_train, y_train, x_test, y_test):
        raise NotImplemented()

    def reshapeFromPointers(self, width, volDepth, irDepth, indices, startPointer=0, nbBatches=10, train=False,
                            fromPointers=True):
        # TODO: multithread reshape
        # pdb.set_trace()
        inSegments = np.empty((0, width, volDepth + irDepth))
        targetSegments = np.empty((0, self.params.shape[0]))
        if (len(self.dataPointers['vol']) > 0):
            # if(len(self.trainIndices)):
            #     self.trainIndices = np.where(self.splitBooleanIndex == True)[0]
            volPointers = np.array(self.dataPointers['vol'])[indices[startPointer:startPointer + nbBatches]]
            irPointers = np.array(self.dataPointers['ir'])[indices[startPointer:startPointer + nbBatches]]
            paramPointers = np.array(self.dataPointers['params'])[indices[startPointer:startPointer + nbBatches]]

            for i in range(len(volPointers)):
                vol = self.volatilities[volPointers[i][0]:volPointers[i][1], volPointers[i][2]:volPointers[i][3]]
                ir = self.ir[irPointers[i][0]:irPointers[i][1], irPointers[i][2]:irPointers[i][3]]
                params = self.params[:, paramPointers[i]]
                inPut = self._mergeReashapeInput(vol=vol, ir=ir)
                inSegments = np.vstack((inSegments, inPut))
                targetSegments = np.vstack((targetSegments, params))
        else:
            raise IndexError()
        # pdb.set_trace()
        inSegments = inSegments.reshape(inSegments.shape[0], 1, width, inSegments.shape[2])
        return inSegments, targetSegments

    def _segmentDataset(self, width, volDepth, irDepth, pointers=True):
        inSegments = np.empty((0, width, volDepth + irDepth))
        targetSegments = np.empty((0, self.params.shape[0]))
        # bad memory handling, we could only keep coordinates of each chunk
        while (True):
            vol, ir, params, traversedDataset = self._buildBatch(width, volDepth, irDepth, pointers=pointers)
            if (pointers):
                self.dataPointers['vol'].append(vol)
                self.dataPointers['ir'].append(ir)
                self.dataPointers['params'].append(params)
            else:
                inPut = self._mergeReashapeInput(vol, ir)
                # pdb.set_trace()
                inSegments = np.vstack((inSegments, inPut))
                targetSegments = np.vstack((targetSegments, params))
            if (traversedDataset):
                break
        if (not pointers):
            self.inputSegments = inSegments.reshape(inSegments.shape[0], 1, width, inSegments.shape[2])
            self.outputSegments = targetSegments

    def _mergeReashapeInput(self, vol, ir):
        # pdb.set_trace()
        inPut = np.column_stack((vol.T, ir.T))
        inPut = inPut.reshape((1, inPut.shape[0], inPut.shape[1]))
        return inPut

    def _buildBatch(self, width, volDepth, irDepth, pointers=True):
        irDepthEnd = False
        volDepthEnd = False
        seriesEnd = False
        traversedFullDataset = False
        # pdb.set_trace()
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

            # print(volStartPosition, volStopPosition, startWidthPosition, endWidthPosition, irStartPosition,
            #       irStopPosition, widthEndFlag)

        if (pointers):
            volData = (volStartPosition, volStopPosition, startWidthPosition, endWidthPosition)
            irData = (irStartPosition, irStopPosition, startWidthPosition, endWidthPosition)
            params = endWidthPosition - 1
        else:
            volData, irData, params = self._getActualData(volStartPosition, volStopPosition, irStartPosition,
                                                          irStopPosition,
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
        # pdb.set_trace()
        if (newPosition > limit):  # CHECK
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

    def _checkFuncInput(self, batchSize, width=None, volDepth=None, irDepth=None):
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

    def _saveProcessedData(self, fileSuffix, data="train"):
        if (data.lower() == "train"):
            dictToSave = self.trainData
        else:
            dictToSave = self.testData

        for dataType in dictToSave:
            fileName = self.filePrefix + self._getCurrentRunId() + '_' + fileSuffix + '_' + str(dataType) + ".npy"
            np.save(fileName, dictToSave[dataType])

    def findTwinFiles(self, fileName):
        fileList = [fileName]
        name, prefix, fileType, mode, rest = dbc.breakPath(self.dataFileName)
        code = re.findall(r'([0-9]{16})', name)
        if (len(code) == 0):
            code = re.findall(r'([0-9]{15})', name)
        if (len(code) > 0):
            code = code[0]
        for subdir, dirs, files in os.walk(prefix):
            for f in files:
                if f.endswith(fileType) and code in f:
                    fileName = os.path.join(prefix, f)
                    if (fileName not in fileList):
                        fileList.append(fileName)

        return fileList

    def _getCurrentRunId(self):
        if (self.runId is None):
            timestamp = ''.join(str(dt.datetime.now().timestamp()).split('.'))
            self.runId = timestamp

        return self.runId
