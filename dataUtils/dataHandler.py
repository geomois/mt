import numpy as np
import re
import dataUtils.dbDataPreprocess as dbc
import models.instruments as inst
import os.path
import pdb
import datetime as dt


class DataHandler(object):
    def __init__(self, dataFileName='data/toyData/AH_vol.npy'
                 , testDataPercentage=0.2, batchSize=50, width=50, volDepth=156,
                 irDepth=44, sliding=True, useDataPointers=False, randomSplit=False, datePointer=False, save=False,
                 specialFilePrefix="", predictiveShape=None, targetDataPath=None, targetDataMode=None):
        target = 'params' if targetDataMode is None else targetDataMode
        if (targetDataPath is not None and targetDataMode is None):
            target = 'target'
        self.modes, self.targetName = self.setupModes(volDepth, irDepth, target)
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
        self.target = None
        self.filePrefix = None
        self.inputSegments = []
        self.outputSegments = []
        self.dataPointers = {"vol": [], "ir": [], self.targetName: []}
        self.lastBatchPointer = -1
        self.saveProcessedData = save
        self.delegatedFromFile = False
        self.runId = None
        self.randomSpliting = randomSplit
        self.specialPrefix = specialFilePrefix
        self.channelStart = None
        self.channelEnd = None
        # predictive tuple (label = "vol" or "ir", [inputWidth, outputWidth])
        self.predictive = predictiveShape
        self.targetDataPath = targetDataPath
        self.transformed = {"test": False, "train": False}
        self._getCurrentRunId()

    def setupModes(self, volDepth, irDepth, targetName):
        modes = []
        target = []
        if (volDepth > 0):
            modes.append('vol')
        if (irDepth > 0):
            modes.append('ir')
        if (targetName is not None):
            modes.append(targetName)
            target = targetName
        return modes, target

    def getTestData(self, batchSize=None, width=None, volDepth=None, irDepth=None):
        if (len(self.testData["input"]) == 0):
            batchSize, width, volDepth, irDepth = self._checkFuncInput(batchSize, width, volDepth, irDepth)
            self.splitTestData(batchSize=batchSize, width=width, volDepth=volDepth, irDepth=irDepth)
        if (self.predictive is not None):
            self._feedTransform('test')
        assert (len(self.testData["input"]) > 0 and len(self.testData["output"]) > 0), "Test data not present"
        if (self.saveProcessedData):
            suffix = 'test' + str(self.specialPrefix) + str(self.batchSize) + "_w" + str(self.segmentWidth) + '_' + str(
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
                    elif (mode == self.targetName and self.targetDataPath is not None):
                        if (os.path.isfile(path=self.targetDataPath)):
                            fileList.append((self.targetDataPath, mode))

        if fileType.lower() == 'csv':
            for path, mode in fileList:
                # next line is a patch of a patch, needs cleaning
                cleanMode = mode if mode not in self.targetName or self.targetName.lower() == 'deltair' else 'target'
                if (clean):
                    df, npy = dbc.cleanCsv(path, mode=cleanMode, toNNData=True, exportPath=True, dbFormat=True)
                else:
                    df, npy = dbc.cleanCsv(path, mode=cleanMode, toNNData=True, exportPath=True, dbFormat=False)
                self._setDataFiles(npy, mode)
        elif fileType.lower() == 'npy':
            for path, mode in fileList:
                # pdb.set_trace()
                self._setDataFiles(np.load(path), mode)

        if (self.target is None):
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
        if (self.volatilities is None):  # generalize
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

    def initializePipeline(self, pipeline):
        # pdb.set_trace()
        if (self.predictive is not None and not self.transformed['train']):
            self._feedTransform('train')
        if (self.delegatedFromFile and pipeline.steps[1][1] is not None):
            self.trainData["input"] = pipeline.fit_transform(self.trainData["input"])
        else:
            pass
            # raise Exception("Only data delegated from file can use pipeline")
        return pipeline

    def getNextBatch(self, batchSize=None, width=None, volDepth=None, irDepth=None, pipeline=None, randomDraw=False):
        batchSize, width, volDepth, irDepth = self._checkFuncInput(batchSize, width, volDepth, irDepth)
        if (self.lastBatchPointer == -1):
            self.lastBatchPointer = 0
            if (self.delegatedFromFile):
                modulo = len(self.trainData["input"])
                if (self.predictive is not None and not self.transformed['train']):
                    self._feedTransform('train')
                    # modulo = len(self.trainData["input"])
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
                    suffix = 'train' + str(self.specialPrefix) + str(self.batchSize) + "_w" + str(
                        self.segmentWidth) + '_' + str(self.volDepth) + '_' + str(self.irDepth)
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

    def _feedTransform(self, data):
        """
        Reshapes input [batch,1,width,depth] data to [(depth*batch -1),1,width,1] data
        and output [batch,<shape>] or [batch,1,width,depth] to [(width*depth*batch)-1,1]
        :param data: data pointer
        :return: reshaped  x, y
        """
        if (data.lower() == "train"):
            targetDict = self.trainData
        else:
            targetDict = self.testData
        mode = self.predictive[0]
        inWidth = int(self.predictive[1][0])
        outWidth = int(self.predictive[1][1])
        if (len(self.predictive[2])) > 1:
            channelRange = [int(self.predictive[2][0]), int(self.predictive[2][1])]
        elif (len(self.predictive[2])) > 1:
            channelRange = [0, int(self.predictive[2][1])]
        depth = 1
        if (len(self.predictive[1]) > 2):
            depth = int(self.predictive[1][2])
        targetShape = np.asarray(targetDict['input']).shape
        if (mode.lower() is self.modes[0]):
            self.channelStart = 0
            self.channelEnd = self.volDepth
        else:
            self.channelStart = self.volDepth + channelRange[0]
            self.channelEnd = targetShape[3] if channelRange[1] <= 0 else channelRange[1]

        if (outWidth > targetShape[2]):
            outWidth = targetShape[2]
        if (inWidth > targetShape[2]):
            inWidth = targetShape[2]
        if (self.targetName.lower() == 'deltair'):
            output = targetDict['output']
            window = inWidth
            targetArray = np.empty((0, 1))
            for i in range(0, targetShape[0] - window):
                for j in range(self.channelEnd):
                    targetArray = np.vstack((output[i + window, j].reshape(-1, 1), targetArray))
            targetDict['output'] = targetArray
        else:
            targetDict['output'] = self._reshapeToPredict(
                np.asarray(targetDict['input'][1:, :, :outWidth, self.channelStart:self.channelEnd])).reshape(
                (-1, 1))  # skip first

        targetDict['input'] = self._reshapeToPredict(
            np.asarray(
                targetDict['input'][:targetShape[0] - inWidth, :, :inWidth,
                self.channelStart:self.channelEnd]))  # skip last
        self.transformed[data] = True

    def _reshapeToPredict(self, array):
        o = np.empty((0, 1, array.shape[2], 1))
        for i in range(array.shape[0]):
            temp = array[i, :].T.reshape(array.shape[3], 1, array.shape[2], 1)
            o = np.vstack((o, temp))
        return o

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
        elif (mode.lower() == self.targetName.lower()):
            self.target = data

    def reshapeFromPointers(self, width, volDepth, irDepth, indices, startPointer=0, nbBatches=10, train=False,
                            fromPointers=True):
        # pdb.set_trace()
        inSegments = np.empty((0, width, volDepth + irDepth))
        targetSegments = np.empty((0, self.target.shape[0]))
        if (len(self.dataPointers['vol']) > 0):
            # if(len(self.trainIndices)):
            #     self.trainIndices = np.where(self.splitBooleanIndex == True)[0]
            volPointers = np.array(self.dataPointers['vol'])[indices[startPointer:startPointer + nbBatches]]
            irPointers = np.array(self.dataPointers['ir'])[indices[startPointer:startPointer + nbBatches]]
            targetPointers = np.array(self.dataPointers[self.targetName])[
                indices[startPointer:startPointer + nbBatches]]

            for i in range(len(volPointers)):
                vol = self.volatilities[volPointers[i][0]:volPointers[i][1], volPointers[i][2]:volPointers[i][3]]
                ir = self.ir[irPointers[i][0]:irPointers[i][1], irPointers[i][2]:irPointers[i][3]]
                target = self.target[:, targetPointers[i]]
                inPut = self._mergeReashapeInput(vol=vol, ir=ir)
                inSegments = np.vstack((inSegments, inPut))
                targetSegments = np.vstack((targetSegments, target))
        else:
            raise IndexError()
        # pdb.set_trace()
        inSegments = inSegments.reshape(inSegments.shape[0], 1, width, inSegments.shape[2])
        return inSegments, targetSegments

    def _segmentDataset(self, width, volDepth, irDepth, pointers=True):
        inSegments = np.empty((0, width, volDepth + irDepth))
        targetSegments = np.empty((0, self.target.shape[0]))
        while (True):
            vol, ir, target, traversedDataset = self._buildBatch(width, volDepth, irDepth, pointers=pointers)
            if (pointers):
                self.dataPointers['vol'].append(vol)
                self.dataPointers['ir'].append(ir)
                self.dataPointers[self.targetName].append(target)
            else:
                inPut = self._mergeReashapeInput(vol, ir)
                inSegments = np.vstack((inSegments, inPut))
                targetSegments = np.vstack((targetSegments, target))
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
        if (irDepth == 0):
            irDepthEnd = True
        else:
            irDepthEnd = False

        if (volDepth == 0):
            volDepthEnd = True
        else:
            volDepthEnd = False
        if (volDepthEnd and irDepthEnd):
            raise ValueError('Zero depths')
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

        volStartPosition, volStopPosition, volDepthEnd = self._getLimits(seriesEnd, self.prevVolStartPosition,
                                                                         self.prevVolDepthPosition, volDepth,
                                                                         self.volatilities.shape[0], volDepthEnd)

        irStartPosition, irStopPosition, irDepthEnd = self._getLimits(seriesEnd, self.prevIrStartPosition,
                                                                      self.prevIrDepthPosition, irDepth,
                                                                      self.ir.shape[0], irDepthEnd)

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
            target = endWidthPosition - 1
        else:
            volData, irData, target = self._getActualData(volStartPosition, volStopPosition, irStartPosition,
                                                          irStopPosition,
                                                          startWidthPosition, endWidthPosition)

        return volData, irData, target, traversedFullDataset

    def _getLimits(self, seriesEnd, prevStartPosition, prevDepthPosition, seriesDepth, limit, endFlag):
        if (seriesDepth <= 0):
            return 0, 0, True
        else:
            if (not seriesEnd and prevDepthPosition != -1):
                startPosition = prevStartPosition
                stopPosition = prevDepthPosition
            else:
                startPosition, stopPosition, endFlag = \
                    self._checkLimits(prevStartPosition, prevDepthPosition, seriesDepth,
                                      limit)

            return startPosition, stopPosition, endFlag

    def _getActualData(self, volStart, volStop, irStart, irStop, widthStart, widthStop):
        volData = self.volatilities[volStart:volStop, widthStart:widthStop]
        irData = self.ir[irStart:irStop, widthStart:widthStop]
        target = self.target[:, widthStop - 1]

        return volData, irData, target

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
