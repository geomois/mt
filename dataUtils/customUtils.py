import QuantLib as ql
import matplotlib.pyplot as plt
from datetime import date
from matplotlib.dates import MonthLocator, DateFormatter
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.externals import joblib
import os
import models.instruments as inst
import pdb


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
    initCurve = ql.ZeroCurve(dates, rates, curve.dayCounter())
    impliedCurve = ql.ImpliedTermStructure(ql.YieldTermStructureHandle(initCurve), futureDate)
    return impliedCurve


def transformDerivatives(derivative, channelStart, channelEnd, testX):
    derivative = np.asarray(derivative[0])
    step = channelEnd - channelStart
    if (testX.shape[3] is not 1):
        derivative = reshapeMultiple(derivative, 1, channelStart, channelEnd).reshape((-1, testX.shape[2]))
    else:
        derivative = derivative.reshape((-1, testX.shape[2]))

    datapoints = int(derivative.shape[0] / step)
    der = np.empty((0, datapoints))
    for i in range(step):
        temp = []
        for j in range(i, derivative.shape[0], step):
            temp.append(np.abs(np.average(derivative[j])))
        der = np.vstack((der, temp))
    return der


def reshapeMultiple(array, depth, start, end):
    o = np.empty((0, 1, array.shape[2], depth))
    for i in range(array.shape[0]):
        for j in range(start, end, depth):
            colEnd = j + depth
            temp = array[i, :, :, j:colEnd].reshape((1, 1, array.shape[2], depth))
            o = np.vstack((o, temp))
    return o

def loadSavedScaler(path, identifier=None):
    pklList = []
    for subdir, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.pkl'):
                temp = joblib.load(subdir + "/" + f)
                if (type(temp) == StandardScaler or type(temp) == MinMaxScaler):
                    transformFunc = inst.FunctionTransformerWithInverse(func=None, inv_func=None)
                    pp = pipeline = Pipeline([('funcTrm', transformFunc), ('scaler', temp)])
                    pklList.append(pp)
                elif (type(temp) == Pipeline):
                    pklList.append(temp)
    if (len(pklList) == 0):
        raise Exception('Empty pipeline folder')

    return pklList