import numpy as np
import re
import utils.dbDataPreprocess as dbc
import models.SwaptionGenerator as inst
import os.path
import pdb
import utils.customUtils as cu
import datetime as dt

DEFAULT_TEST_DATA_PERCENTAGE = 0.2


class DataHandler(object):
    def __init__(self, dataFileName='data/'
                 , testDataPercentage=DEFAULT_TEST_DATA_PERCENTAGE, batchSize=50, width=50, volDepth=156,
                 irDepth=44, sliding=True, useDataPointers=False, randomSplit=False, datePointer=False, save=False,
                 specialFilePrefix="", predictiveShape=None, targetDataPath=None, targetDataMode=None, cropFirst=0,
                 alignedData=False, perTermScale=False):
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
        self.cropFirst = cropFirst
        self.alignedData = alignedData
        self.perTermScale = perTermScale
        self.getCurrentRunId()

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
        if (self.predictive is not None and not self.transformed['test']):
            self._feedTransform('test')
        assert (len(self.testData["input"]) > 0 and len(self.testData["output"]) > 0), "Test data not present"
        if (self.saveProcessedData):
            suffix = 'test' + str(self.specialPrefix) + str(self.batchSize) + "_w" + str(self.segmentWidth) + '_' + str(
                self.volDepth) + '_' + str(self.irDepth)
            self._saveProcessedData(suffix, 'test')

        return np.asarray(self.testData["input"]), np.asarray(self.testData['output'])

    def readData(self, batchSize=None, twinFile=True, clean=False):
        # pdb.set_trace()
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

    def delegateDataDictsFromFile(self, fileList, simplify=False):
        self.delegatedFromFile = True
        self.useDataPointers = False
        for i in range(len(fileList)):
            file = fileList[i]
            if ("test" in file):
                dataType = 'test'
                dataDict = self.testData
            else:
                dataType = 'train'
                dataDict = self.trainData

            mode = ""
            if ("input" in file):
                mode = "input"
            elif ("output" in file):
                mode = "output"
                if (len(dataDict[mode]) > 0):
                    # Skip importing output from file if the predictive shape is already used to delegate output
                    continue
            try:
                if (len(mode) > 0):
                    dataDict[mode] = np.load(file)
                    if (simplify):
                        if (self.predictive is not None):
                            if (not self.transformed[dataType]):
                                self._feedTransform(dataType)
                        dataDict[mode] = self._simplify(np.asarray(dataDict[mode]))
                else:
                    raise FileNotFoundError
            except:
                raise Exception()

    def splitTestData(self, batchSize=None, width=None, volDepth=None, irDepth=None):
        batchSize, width, volDepth, irDepth = self._checkFuncInput(batchSize, width, volDepth, irDepth)
        if (self.volatilities is None or self.ir is None):
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
            outPut = None
            if (self.target is not None):
                outPut = self.outputSegments[~self.splitBooleanIndex]

        if (self.datePointer and not self.randomSplitting):
            firstTestPoint = int(np.floor(dataLength * (1 - self.testDataPercentage)))
            self.testData['output'] = np.int32(np.arange(firstTestPoint, firstTestPoint + len(self.testData['output'])))
        else:
            self.testData['output'] = outPut
        self.testData['input'] = inPut

    def initializePipelines(self, inputPipeline=None, outPipeline=None):
        if (self.predictive is not None and not self.transformed['train']):
            self._feedTransform('train')
        if (self.predictive is not None and not self.transformed['test']):
            self._feedTransform('test')
        if (self.delegatedFromFile):
            if (outPipeline is not None):
                self.trainData["output"] = outPipeline.fit_transform(self.trainData["output"])
            if (inputPipeline is not None):
                tt = np.asarray(self.trainData['input'])
                if (len(tt.shape) > 2):
                    if (len(np.asarray(self.trainData['input']).shape) == 3):
                        tt = tt.reshape((-1, tt.shape[1]))
                        inputPipeline = inputPipeline.fit(tt)
                        for i in range(np.asarray(self.trainData["input"]).shape[2]):
                            # self.trainData["input"][:, 0, :, i] = inputPipeline.fit_transform(
                            #     self.trainData["input"][:, 0, :, i])
                            self.trainData["input"][:, :, i] = inputPipeline.transform(self.trainData["input"][:, :, i])
                    elif (len(np.asarray(self.trainData['input']).shape) == 4):
                        tt = tt.reshape((-1, tt.shape[2]))
                        inputPipeline = inputPipeline.fit(tt)
                        for i in range(np.asarray(self.trainData["input"]).shape[3]):
                            # self.trainData["input"][:, 0, :, i] = inputPipeline.fit_transform(
                            #     self.trainData["input"][:, 0, :, i])
                            self.trainData["input"][:, 0, :, i] = inputPipeline.transform(
                                self.trainData["input"][:, 0, :, i])
                else:
                    self.trainData["input"] = inputPipeline.fit_transform(self.trainData["input"])
        return inputPipeline, outPipeline

    def getNextBatch(self, batchSize=None, width=None, volDepth=None, irDepth=None, pipeline=None, randomDraw=False):
        batchSize, width, volDepth, irDepth = self._checkFuncInput(batchSize, width, volDepth, irDepth)
        if (self.lastBatchPointer == -1):
            self.lastBatchPointer = 0
            if (self.delegatedFromFile):
                modulo = len(self.trainData["input"])
                if (self.predictive is not None and not self.transformed['train']):
                    self._feedTransform('train')
                    # modulo = len(self.trainData["input"])
                if (self.saveProcessedData):
                    suffix = 'train' + str(self.specialPrefix) + str(self.batchSize) + "_w" + str(
                        self.segmentWidth) + '_' + str(self.volDepth) + '_' + str(self.irDepth)
                    self._saveProcessedData(suffix, 'train')
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
                    if (len(self.trainData["input"]) == 0):
                        self.splitTrainData(pipeline)
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

    def splitTrainData(self, pipeline=None):
        self.trainData["input"] = self.inputSegments[self.splitBooleanIndex]
        if (self.target is not None):
            self.trainData["output"] = self.outputSegments[self.splitBooleanIndex]
        if (self.predictive is not None and not self.transformed['train']):
            self._feedTransform('train')
        if (pipeline is not None):
            # pdb.set_trace()
            self.trainData["output"] = pipeline.fit_transform(self.trainData["output"])
        if (self.saveProcessedData):
            suffix = 'train' + str(self.specialPrefix) + str(self.batchSize) + "_w" + str(
                self.segmentWidth) + '_' + str(self.volDepth) + '_' + str(self.irDepth)
            self._saveProcessedData(suffix, 'train')

    def _feedTransform(self, data):
        """
        Reshapes input [batch,1,width,depth] data to [(depth*batch -1),1,width,1] data
        and output [batch,<shape>] or [batch,1,width,depth] to [(width*depth*batch)-1,1]
        :param data: data pointer
        :return: reshaped  x, y
        """
        # region unravelling input
        if (data.lower() == "train"):
            targetDict = self.trainData
        else:
            targetDict = self.testData
        mode = self.predictive[0]
        inWidth = int(self.predictive[1][0])
        outWidth = int(self.predictive[1][1])
        transformFlag = True
        if (len(self.predictive) > 2):
            transformFlag = self.predictive[3]
        if (len(self.predictive[2])) > 1:
            channelRange = [int(self.predictive[2][0]), int(self.predictive[2][1])]
        else:
            channelRange = [0, int(self.predictive[2][0])]
        outDepth = 1
        if (len(self.predictive[1]) > 2):
            outDepth = int(self.predictive[1][2])
        targetShape = np.asarray(targetDict['input']).shape

        if (mode.lower() is self.modes[0]):
            self.channelStart = 0
            self.channelEnd = self.volDepth
        else:
            self.channelStart = self.volDepth + channelRange[0]
            diff = channelRange[1] - channelRange[0]
            self.channelEnd = targetShape[3] if channelRange[1] <= 0 else self.channelStart + diff
        if (outWidth > targetShape[2]):
            outWidth = targetShape[2]
        if (inWidth > targetShape[2]):
            inWidth = targetShape[2]
        # endregion unravelling input

        if (transformFlag):
            if (self.targetName.lower() == 'deltair'):
                output = targetDict['output']
                window = inWidth
                targetArray = np.empty((0, outDepth))
                for i in range(0, targetShape[0] - window):
                    for j in range(self.channelStart, self.channelEnd, outDepth):
                        colEnd = j + outDepth
                        # rowEnd = i + window + outWidth TODO:implement future movement

                        # data is aligned to have -1 datapoint
                        targetArray = np.vstack((targetArray, output[i, j:colEnd].reshape(-1, outDepth)))
                        # targetArray = np.vstack((targetArray, output[i + window, j:colEnd].reshape(-1, outDepth)))
                targetDict['output'] = targetArray
            else:
                end = self.channelEnd
                if(outDepth < (self.channelEnd - self.channelStart)):
                    end = self.channelStart + outDepth

                targetDict['output'] = self._reshapeToPredict(
                    np.asarray(targetDict['input'][inWidth:, :, :outWidth, self.channelStart:end]),
                    outDepth).reshape((-1, outDepth))  # skip first
            inPut = targetDict['input'][:targetShape[0] - inWidth, :, :inWidth,
                    self.channelStart:self.channelEnd]  # skip last
            targetDict['input'] = self.reshapeMultiple(np.asarray(inPut), outDepth)
            self.transformed[data] = True

    def _reshapeToPredict(self, array, depth):  # CHECK
        tShape = array.reshape((-1, 1, array.shape[2], depth)).shape
        o = np.empty(tShape)
        # o = np.empty((0, 1, array.shape[2], 1))
        for i in range(array.shape[0]):
            temp = array[i, :].T.reshape(-1, 1, array.shape[2], depth)
            # o = np.vstack((o, temp))
            o[i] = temp
        return o

    def reshapeMultiple(self, array, depth):  # CHECK
        # o = np.empty((0, 1, array.shape[2], depth))
        tShape = array.reshape((-1, 1, array.shape[2], depth)).shape
        if (tShape == array.shape):
            return array
        o = np.empty(tShape)
        for i in range(array.shape[0]):
            for j in range(self.channelStart, self.channelEnd, depth):
                colEnd = j + depth
                temp = array[i, :, :, j:colEnd].reshape((1, 1, array.shape[2], depth))
                # o = np.vstack((o, temp))
                o[(i * array.shape[3]) + j] = temp
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
        if (self.targetName.lower() == "deltair" and mode.lower() != self.targetName.lower()):
            data = data[:, self.cropFirst:]
        if (self.perTermScale):
            data = cu.rScale(data, self)
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
        targetSegments = None
        if (self.target is not None):
            targetSegments = np.empty((0, self.target.shape[0]))
        while (True):
            vol, ir, target, traversedDataset = self._buildBatch(width, volDepth, irDepth, pointers=pointers)
            if (pointers):
                self.dataPointers['vol'].append(vol)
                self.dataPointers['ir'].append(ir)
                self.dataPointers[self.targetName].append(target)
            else:
                if (vol is None and ir is None and target is None):
                    pass
                else:
                    inPut = self._mergeReashapeInput(vol, ir)
                    inSegments = np.vstack((inSegments, inPut))
                    if (self.target is not None):
                        targetSegments = np.vstack((targetSegments, target))
            if (traversedDataset):
                # inSegments = inSegments[:inSegments.shape[0] - 3]  # last are leftovers
                # targetSegments = targetSegments[:targetSegments.shape[0] - 3]  # CHECK
                break
        if (not pointers):
            self.inputSegments = inSegments.reshape(inSegments.shape[0], 1, width, inSegments.shape[2])
            if (self.target is not None):
                self.outputSegments = targetSegments

    def _mergeReashapeInput(self, vol, ir):
        inPut = None
        if (vol is None):
            inPut = ir.T
        if (ir is None):
            inPut = vol.T
        inPut = np.column_stack((vol.T, ir.T)) if inPut is None else inPut
        inPut = inPut.reshape((1, inPut.shape[0], inPut.shape[1]))
        return inPut

    def forceSimplify(self, mode="p"):
        """
        :param mode: p -> preserve data depth, l -> use only last time-point
        :return:
        """
        if (not self.delegatedFromFile):
            # if (self.predictive is not None):
            if (self.volatilities is None or self.ir is None):
                self.splitTestData()
                self.splitTrainData()
            if (not self.transformed['train']):
                self._feedTransform('train')
            if (not self.transformed['test']):
                self._feedTransform('test')
        self.trainData['input'] = self._simplify(self.trainData['input'])
        self.trainData['output'] = self._simplify(self.trainData['output'])
        self.testData['input'] = self._simplify(self.testData['input'])
        self.testData['output'] = self._simplify(self.testData['output'])
        if (mode.lower() == 'l'):
            if (len(self.trainData['input'].shape) == 3):
                self.trainData['input'] = self.trainData['input'][:, -1, :]
                self.testData['input'] = self.testData['input'][:, -1, :]
            elif (len(self.trainData['input'].shape) == 4):
                self.trainData['input'] = self.trainData['input'][:, :, -1, :]
                self.testData['input'] = self.testData['input'][:, :, -1, :]

    def _simplify(self, x):
        x = np.asarray(x)
        if (len(x.shape) > 3):
            if (x.shape[1] == 1):
                if (x.shape[2] == 1):
                    x = x.reshape(x.shape[0], x.shape[3])
                else:
                    x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
            if (x.shape[1] == 1 and x.shape[3] == 1):
                x = x.reshape(x.shape[0], x.shape[2])
        return x

    def _buildBatch(self, width, volDepth, irDepth, pointers=True):
        irData = volData = target = None
        targetEnd = False
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

        try:
            if (self.volatilities is None):
                targetShape = self.ir.shape
            elif (self.ir is None):
                targetShape = self.volatilities.shape
            else:
                targetShape = self.volatilities.shape
        except:
            targetShape = None
            raise Exception("Volatilities or Ir should be supplied")

        startWidthPosition, endWidthPosition, widthEndFlag = \
            self._checkLimits(self.prevWidthStartPosition, self.prevWidthStopPosition, step, targetShape[1])

        if (widthEndFlag):
            self.endOfSeriesCount += 1

        if (volDepth != 0):
            volStartPosition, volStopPosition, volDepthEnd = self._getLimits(seriesEnd, self.prevVolStartPosition,
                                                                             self.prevVolDepthPosition, volDepth,
                                                                             self.volatilities.shape[0], volDepthEnd)
            self.prevVolDepthPosition = volStopPosition
            self.prevVolStartPosition = volStartPosition
            if (pointers):
                volData = (volStartPosition, volStopPosition, startWidthPosition, endWidthPosition)
            else:
                volData = self.volatilities[volStartPosition:volStopPosition, startWidthPosition:endWidthPosition]

        if (irDepth != 0):
            irStartPosition, irStopPosition, irDepthEnd = self._getLimits(seriesEnd, self.prevIrStartPosition,
                                                                          self.prevIrDepthPosition, irDepth,
                                                                          self.ir.shape[0], irDepthEnd)
            self.prevIrDepthPosition = irStopPosition
            self.prevIrStartPosition = irStartPosition

            if (pointers):
                irData = (irStartPosition, irStopPosition, startWidthPosition, endWidthPosition)
            else:
                irData = self.ir[irStartPosition:irStopPosition, startWidthPosition:endWidthPosition]

        if (self.target is not None):
            if (self.alignedData):
                targetPos = startWidthPosition
            else:
                if (self.targetName.lower() == 'deltair'):
                    targetPos = endWidthPosition
                else:
                    targetPos = endWidthPosition - 1

            if (pointers):
                target = targetPos
            else:
                if (targetPos >= self.target.shape[1]):
                    # pos = self.target.shape[1] - 1
                    irData = volData = target = None
                    targetEnd = True
                else:
                    target = self.target[:, targetPos]

        self.prevWidthStopPosition = endWidthPosition
        self.prevWidthStartPosition = startWidthPosition

        if ((seriesEnd and irDepthEnd and volDepthEnd) or targetEnd):
            self.endOfSeriesCount = 0
            self.prevIrStartPosition = -1
            self.prevVolStartPosition = -1
            self.prevVolDepthPosition = -1
            self.prevIrDepthPosition = -1
            self.prevWidthStopPosition = -1
            self.prevWidthStartPosition = -1
            traversedFullDataset = True

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
            fileName = self.filePrefix + self.getCurrentRunId() + '_' + fileSuffix + '_' + str(dataType) + ".npy"
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

    def getCurrentRunId(self):
        if (self.runId is None):
            timestamp = ''.join(str(dt.datetime.now().timestamp()).split('.'))
            self.runId = timestamp

        return self.runId
