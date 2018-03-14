import QuantLib as ql
import matplotlib.pyplot as plt
from datetime import date
from matplotlib.dates import MonthLocator, DateFormatter
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.externals import joblib
import re, pickle, os, pdb
from utils.FunctionTransformer import FunctionTransformerWithInverse
from scipy.integrate import simps, trapz

optionList = ["dropout_rate", "architecture", "use_calibration_loss", "currency", "gpu_memory_fraction", "suffix",
              "decay_steps", "weight_reg_strength", "irFileName", "model_dir", "historyStart", "test_frequency",
              "channel_range", "learning_rate", "batch_width", "conv_ir_depth", "batch_size", "fullyConnectedNodes",
              "model", "activation", "extend_training", "use_pipeline", "skip", "weight_reg", "weight_init_scale",
              "conv_vol_depth", "futureIncrement", "use_cpu", "with_gradient", "target", "volFileName", "chained_model",
              "data_ir_depth", "processedData", "data_dir", "scaler", "exportForwardRates", "log_dir",
              "predictiveShape", "max_steps", "full_test", "compare", "dnn_hidden_units", "data_vol_depth",
              "historyEnd", "print_frequency", "checkpoint_freq", "nn_model", "checkpoint_dir", "paramsFileName",
              "optimizer", "is_train", "pipeline", "calculate_gradient", "no_transform", "decay_rate",
              "saveProcessedData", "calibrate_sigma", "irType", "calibrate", "weight_init", "decay_staircase",
              "outDims", "input_dims", "output_dims"]

dateInDays = {"swap": [365, 730, 1095, 1460, 1825, 2190, 2555, 2920, 3285, 3650, 5475, 7300, 9125],
              "libor": [0, 1, 2, 7, 14, 30, 61, 91, 122, 152, 182, 213, 243, 273, 304, 335, 365, 395, 425, 456,
                        486, 516, 547, 577, 607, 638, 668, 698, 730, 1095, 1460, 1825, 2190, 2555, 2920, 3285,
                        3650, 4380, 5475, 7300, 9125, 10950, 14600, 18250],
              "eonia": [30, 61, 91, 122, 152, 182, 213, 243, 273, 304, 335, 365, 547, 730, 1095, 1460, 1825,
                        2190, 2555, 2920, 3285, 3650, 4380, 5475, 7300, 9125, 10950, 12775, 14600, 18250],
              "clean": [30, 61, 91, 122, 152, 182, 213, 243, 273, 304, 335, 365, 547, 730, 1095, 1460, 1825,
                        2190, 2555, 2920, 3285, 3650]
              }
index_Map_Libor_Eonia = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 27, 28, 29, 30, 31, 32, 33, 34, 35]

Dt = 0.001


def toDatetime(d):
    return date(d.year(), d.month(), d.dayOfMonth())


def formatRate(r):
    return '%.4f %%' % (r.rate() * 100.0)


def plotCurve(today, *curves):
    fig, ax = plt.subplots()
    plt.rc('lines', linewidth=3)
    dates = [today + ql.Period(i, ql.Weeks) for i in range(0, 52 * 5)]
    for c in curves:
        validDates = [d for d in dates if d >= c.referenceDate()]
        rates = [c.forwardRate(d, d + 1, ql.Actual360(), ql.Simple).rate() for d in validDates]
        ax.plot_date([toDatetime(d) for d in validDates], rates, '-')

    ax.set_xlim(toDatetime(min(dates)), toDatetime(max(dates)))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[6, 12]))
    ax.xaxis.set_major_formatter(DateFormatter("%b '%y"))
    ax.set_ylim(0.0, 0.01)
    ax.autoscale_view()
    ax.xaxis.grid(True, 'major')
    ax.xaxis.grid(False, 'minor')
    fig.autofmt_xdate()
    plt.show()


def createCurve(refDate, tenors, rates, irType='libor'):
    if (irType.lower() == 'libor'):
        irType = ql.GBPLibor(ql.Period(6, ql.Months))
    elif (irType.lower() == 'euribor'):
        irType = ql.Euribor(ql.Period(6, ql.Months))

    helpers = [ql.SwapRateHelper(ql.QuoteHandle(ql.SimpleQuote(rate)),
                                 ql.Period(*tenor), ql.TARGET(), ql.Annual, ql.Unadjusted,
                                 ql.Thirty360, irType)
               for tenor, rate in zip(tenors, rates)]
    curve = ql.MonotonicCubicZeroCurve(0, ql.TARGET(), helpers, ql.Actual360)
    return curve


def getImpliedForwardCurve(futureDate, curve):
    '''
    :param futureDate: future evaluation date
    :param curve: a QuantLib curve
    :return:
    '''
    refDate = curve.nodes()[0][0]
    ql.Settings.instance().evaluationDate = refDate
    dates, rates = zip(*curve.nodes())
    pdb.set_trace()
    initCurve = ql.ZeroCurve(dates, rates, curve.dayCounter())
    impliedCurve = ql.ImpliedTermStructure(ql.YieldTermStructureHandle(initCurve), futureDate)
    return impliedCurve


def transformDerivatives(derivative, channelStart, channelEnd, xShape):
    derivative = np.asarray(derivative[0])
    step = channelEnd - channelStart
    if (len(xShape) > 2):
        if (xShape[3] is not 1):
            derivative = reshapeMultiple(derivative, 1, channelStart, channelEnd).reshape((-1, xShape[2]))
        else:
            derivative = derivative.reshape((-1, xShape[2]))
        datapoints = int(derivative.shape[0] / step)
        der = np.zeros((step, datapoints))
    else:
        derivative = derivative.reshape(-1, derivative.shape[0])
        der = np.zeros(derivative.shape)

    depth = derivative.shape[1]
    lin = np.linspace(1, derivative.shape[1], num=derivative.shape[1])[::-1]
    dt = np.linspace(0.0027, derivative.shape[1] * 0.0027, num=derivative.shape[1])[::-1]
    times = np.asarray(dateInDays['libor'])
    denom = derivative.shape[1]
    depthScaler = Dt * denom
    # pdb.set_trace()
    for i in range(step):
        temp = []
        for j in range(i, derivative.shape[0], step):
            # integral = simps(-np.log(1 - derivative[j]) / dt, -lin) / denom  # 1st
            # integral = simps(-np.log(np.abs(derivative[j])), -dt) / denom  # abs
            if derivative[j].shape[0] == 1:
                integral = (np.log(1 - derivative[j]) / dt) * depthScaler  # 1st
            else:
                integral = simps(np.log(1 - derivative[j]) / dt, -lin * depthScaler)  # 1st cnn
                # integral = simps(np.log(1 - derivative[j]) / dt, -lin * 0.01)  # 1st lstm
                # integral = simps(-np.log(np.abs(derivative[j])), -dt) / denom  # abs
                # integral = simps(np.log(1 - derivative[0]), -lin) / denom  # 2nd
                # integral = -np.log(1 - derivative[j][-1]) / 0.0027  # last
            temp.append(integral)
            der[i] = temp
    return der


def reshapeMultiple(array, depth, start, end):
    cc = []
    o = np.empty((array.shape[0] * array.shape[3], 1, array.shape[2], depth))
    for i in range(array.shape[0]):
        for j in range(start, end, depth):
            colEnd = j + depth
            temp = array[i, :, :, j:colEnd].reshape((1, 1, array.shape[2], depth))
            cc.append((i * array.shape[3]) + j)
            o[(i * array.shape[3]) + j] = temp
    # return np.abs(o)
    return o


#

def loadSavedScaler(path, identifier=None):
    if (os.path.isfile(path)):
        temp = joblib.load(path)
        if (type(temp) == StandardScaler or type(temp) == MinMaxScaler):
            transformFunc = FunctionTransformerWithInverse(func=None, inv_func=None)
            pp = Pipeline([('funcTrm', transformFunc), ('scaler', temp)])
            return pp
        return None

    pklList = []
    for subdir, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.pkl'):
                temp = joblib.load(subdir + "/" + f)
                if (type(temp) == StandardScaler or type(temp) == MinMaxScaler):
                    transformFunc = FunctionTransformerWithInverse(func=None, inv_func=None)
                    pp = Pipeline([('funcTrm', transformFunc), ('scaler', temp)])
                    index = int(re.findall(r"([0-9][0-9]?)", f)[0])
                    pklList.append((index, pp))
                elif (type(temp) == Pipeline):
                    pklList.append(temp)
    if (len(pklList) == 0):
        raise Exception('Empty pipeline folder')
    pklList = sorted(pklList, key=lambda tup: tup[0])
    pklList = [i[1] for i in pklList]
    return pklList


def splitFileName(path):
    fileName = ''.join(re.findall(r'(/)(\w+)', path).pop())
    directory = path.split(fileName)[0]
    return fileName, directory


def save_obj(obj, name):
    with open(name, 'wb+') as f:
        pickle.dump(obj, f)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def prepareProcData(mode='ir', scaleParams=False, dataFileName='data/toyData/AH_vol.csv', targetDataPath=None,
                    targetDataMode=None, specialFilePrefix=None, predictiveShape=None, volDepth=156, irDepth=44,
                    width=30, cropFirst=0, alignedData=False):
    import utils.dataHandler as dh
    # argetDataPath = 'exports/AH_ir_Delta_fDays365.csv'
    # targetDataMode = 'deltair'
    # specialFilePrefix = '_perTermSTANDARD_pfw365_'
    # example:
    # import utils.customUtils as cu
    # cu.prepareProcData(scaleParams=False, dataFileName='data/ownData/AH_eonia2005_ir.csv',targetDataPath='exports/eonia_Delta_fDays365.csv',targetDataMode='deltair',specialFilePrefix='_eonia_pfw365_',volDepth=0,irDepth = 22, width = 30)
    # cu.prepareProcData(scaleParams=True, dataFileName='data/ownData/AH_eonia2005_ir.csv',
    #                    targetDataPath='exports/EONIA_delta_per100alpha.csv', targetDataMode='deltair',
    #                    specialFilePrefix='_perTermStdPer100Delta_', volDepth=0, irDepth=22, width=30, cropFirst=70,
    # alignedData=True)
    dd = dh.DataHandler(dataFileName=dataFileName, volDepth=volDepth, irDepth=irDepth, width=width,
                        useDataPointers=False, save=True, predictiveShape=predictiveShape,
                        specialFilePrefix=specialFilePrefix, cropFirst=cropFirst, alignedData=alignedData,
                        targetDataPath=targetDataPath, targetDataMode=targetDataMode)
    if (scaleParams):
        dd.readData(dd.dataFileName)
        sc = StandardScaler()
        if (mode.lower() == 'ir'):
            array = dd.ir
        elif (mode.lower() == 'vol'):
            array = dd.volatilities

        temp = np.empty((0, array.shape[1]))
        for i in range(array.shape[0]):
            scaled = sc.fit_transform(array[i, :].reshape((-1, 1)))
            suffix = 'exports/perTermScaler' + str(i) + str(dd.specialPrefix) + str(dd.batchSize) + "_w" + str(
                dd.segmentWidth) + '_' + str(dd.volDepth) + '_' + str(dd.irDepth)
            joblib.dump(sc, suffix + ".pkl", compress=1)
            temp = np.vstack((temp, scaled[:, 0]))
        array = temp
    _ = dd.getTestData()
    _ = dd.getNextBatch()
    suffix = 'train' + str(dd.specialPrefix) + str(dd.batchSize) + "_w" + str(dd.segmentWidth) + '_' + str(
        dd.volDepth) + '_' + str(dd.irDepth)
    dd._saveProcessedData(suffix, 'train')


def rScale(array, dataHandler):
    dd = dataHandler
    sc = StandardScaler()
    temp = np.empty((0, array.shape[1]))
    for i in range(array.shape[0]):
        scaled = sc.fit_transform(array[i, :].reshape((-1, 1)))
        suffix = 'exports/perTermScaler' + str(i) + str(dd.specialPrefix) + str(dd.batchSize) + "_w" + str(
            dd.segmentWidth) + '_' + str(dd.volDepth) + '_' + str(dd.irDepth)
        joblib.dump(sc, suffix + ".pkl", compress=1)
        temp = np.vstack((temp, scaled[:, 0]))
    array = temp
    return array
