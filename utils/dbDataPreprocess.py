import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from dateutil.relativedelta import relativedelta as rd
import QuantLib as ql
import re, math, datetime
from collections import deque
import scipy.stats as stats
import pdb


def toAHDateSequence(dFrame, columns=['X'], shortDate=True):
    dat = dFrame.copy()
    if (not shortDate):
        dat = shortTenorProcess(dFrame, columns)
    series = dat.X.as_matrix()
    for i in range(len(series)):
        currentDate = pd.to_datetime(series[i]).date()
        num = series[i]  # columns generalize
        targetDate = currentDate + rd(months=int(num % 1 * 100)) + rd(years=int(math.floor(num)))
        diff = (targetDate - currentDate).days
        series[i] = diff
    return dat


def toAHFileFormat(fileName, targetFile=None, keyName=None, save=False):
    cols = pd.read_csv(fileName, nrows=1).columns
    data = pd.read_csv(fileName, parse_dates=[0], infer_datetime_format=True, index_col=cols.tolist()[:-1])
    if (save):
        assert (keyName is not None), "Error while saving file, invalid key name"
        assert (targetFile is not None), "Error while saving file, invalid target file name"
        with pd.HDFStore(targetFile) as store:
            store[keyName] = data
            store.close()
    return data


def shortTenorToQLDate(evaluationDate, series):
    assert type(evaluationDate) == datetime.date, "evaluationDate should be of type datetime.date"
    # return [evaluationDate + rd(months=int(num%1*100)) + rd(years=int(math.floor(num))) for num in series]
    qlDates = []
    for num in series:
        d = evaluationDate + rd(months=int(num % 1 * 100)) + rd(years=int(math.floor(num)))
        qlDates.append(ql.Date(d.day, d.month, d.year))
    return qlDates


def simplePlotVals(vals, ax, end=0, labels=["meanErrorConv", "meanErrorFnn", "meanErrorQL"]):
    if (ax is None):
        ax = np.arange((vals[0].shape[0]))
    for i in range(vals.shape[0]):
        plt.plot(ax, vals[i][:, :end], label=labels[i])
    plt.grid()
    plt.legend(loc=9, bbox_to_anchor=(1.0, 1.0))


def plotGroups(dat, xLabels, groupByColumn='X', columnToPlot='Z', sortValue='Date', scatter=False, start=-1, end=10):
    g = dat.groupby(groupByColumn)
    ax = xLabels
    if (ax is None):
        ax = np.asarray(pd.to_datetime(pd.unique(dat.Date)))
    counter = 0
    for i in g.groups:
        if start - counter < 0 or start == -1:
            if scatter:
                plt.plot(ax, g.get_group(i).sort_values(sortValue, ascending=True)[columnToPlot], label=i, marker='.',
                         linestyle='None')
            else:
                plt.plot(ax, g.get_group(i).sort_values(sortValue, ascending=True)[columnToPlot], label=i)
        if counter > end + start and start != -1:
            break
        counter += 1
    plt.legend(loc=9, bbox_to_anchor=(1.0, 1.0))
    plt.show()


def writeToFile(dat, sortCol, groupCol, name='smth.csv', indexFlag=False):
    g = dat.sort_values(sortCol, ascending=True).groupby(groupCol)  # groupby unneeded
    g.head(len(dat)).copy().to_csv(name, mode='w', index=indexFlag)


# region nnData
term0 = 'Term'
term1 = 'OptionTerm'
term2 = 'SwapTerm'


def breakPath(path):
    try:
        split = deque(re.findall(r'(.*/)?((\w+)?(_)(\w+)?(\.)(\w+))', path)[0])
    except:
        print('Snap! split error')
        raise RuntimeError()

    prefix = split.popleft()
    fileName = split.popleft()
    fileType = split.pop()
    split.pop()  # pop '.' of file type
    mode = split.pop()  # pop the mode part
    return fileName, prefix, fileType, mode, split


def shortTenorProcess(dFrame, columns=['X', 'Y']):
    dat = dFrame.copy()
    for j in range(len(columns)):
        series = np.asarray(dat[columns[j]])
        for i in range(0, len(series)):
            new = re.findall(r'(\d+)(\w)', series[i])
            if len(new) > 1:
                if new[0][1].lower() == 'm':
                    num = int(new[1][0])
                    # series[i]=(int(new[1][0])/10)#discard the first part
                    # 0.11 <- 11 months, 1.02<-14 months, 0.03 <- 3 months
                    series[i] = num / 100 if num < 12 else 1 + (num % 12) / 100
                else:
                    series[i] = int(new[0][0]) + (int(new[1][0]) / 100)
            elif len(new) == 1:
                series[i] = int(new[0][0])
                if new[0][1].lower() == 'm':
                    series[i] = series[i] / 100 if series[i] < 12 else 1 + (series[i] % 12 / 100)
    return dat


def cleanCsv(path, mode='vol', toNNData=False, exportPath=None, dbFormat=False):
    '''
    :param path:
    :param mode:
    :param toNNData:
    :param exportPath:
    :param dbFormat:
    :return: pandas DataFrame, numpy array
    '''
    # pdb.set_trace()
    df = pd.DataFrame()
    try:
        df = pd.read_csv(path, header=0)
    except:
        print("Error reading file")
        return
    # pdb.set_trace()
    df = shortClean(df, hasDuplicates=True)
    targetFlag = False
    if (mode != 'target'):
        df, terms = fixColumnNamesAndFieldFormats(df, dbFormat=dbFormat)
    else:
        terms = [i for i in df.columns if (i.lower() != 'date')]
        targetFlag = True

    nnData = None
    if (toNNData):
        if (exportPath is True):
            name, prefix, fileType, mode, rest = breakPath(path)
            exportPath = prefix + ''.join(rest) + mode + '.npy'
        nnData = dfToNNData(df, terms, targetDirectory=exportPath, target=targetFlag)

    return df, nnData


# concatenate the Term columns into tuple (OptionTerm,SwapTerm)

def fixColumnNamesAndFieldFormats(df, dbFormat=False):
    columns = []
    for c in df.columns:
        if (not ('date' in c.lower() or 'value' in c.lower() or 'z' in c.lower())):
            columns.append(c)

    # pdb.set_trace()
    if (dbFormat):
        df = shortTenorProcess(df, columns=columns)

    newName = ''
    for c in columns:
        if (c.lower() == 'x'):
            newName = 'Term'  # IR
            if (len(columns) > 1 or c.lower() == 'option term'):
                newName = term1  # Swaptions
            df = df.rename(index=str, columns={c: newName})
        elif (c.lower() == 'y' or c.lower() == 'swap term'):
            df = df.rename(index=str, columns={c: term2})

    terms = []
    if (len(columns) > 1):
        df = df.sort_values(['Date', term1, term2])
        terms = [term1, term2]
    else:
        df = df.sort_values(['Date', term0])
        terms = [term0]

    return df, terms


def dfToNNData(dFrame, groupingColumns, targetDirectory=None, target=False):
    """
    :param dFrame: pandas dataframe with columns [Date, Term, Value] or [Date, Term1, Term2, Value]
                    or [Date, Alpha, Sigma]
    :param targetDirectory: directory to save numpy array
    :param groupingColumns:
    :param target:
    :return:
    """
    assert ('Date' in dFrame.columns), "No date in dataframe"
    dat = None
    low = [i.lower() for i in groupingColumns]
    # if ('alpha' in low or 'sigma' in low):
    if (target):
        dat = np.zeros((dFrame.shape[0]))
        for c in groupingColumns:
            dat = np.vstack((dat, dFrame[c].copy().as_matrix()))
    else:
        g = dFrame.groupby(groupingColumns)
        keyList = list(g.groups.keys())
        if (type(keyList[0]) == tuple):
            keyList = sorted(keyList, key=lambda tup: (tup[0], tup[1]))
        else:
            keyList = sorted(keyList)
        dat = np.empty((g.get_group(keyList[0]).Value.shape[0]))
        # pdb.set_trace()
        for key in keyList:
            dat = np.vstack((dat, g.get_group(key).Value.copy().as_matrix()))

    dat = dat[1:, :]
    # pdb.set_trace()
    if (targetDirectory is not None):
        np.save(targetDirectory, dat)
    return dat


def shortClean(dFrame, columns=['MDT_ID', 'MDE_ID', 'CURVE_ID_1', 'CURVE_ID_2', 'CURVE_ID_3', 'GMDB_SYMBOL', 'PNT_ID',
                                'CURVE_UNIFIED', 'Z1', 'MONEYNESS', "FutureDate"],
               hasDuplicates=False):
    dat = dFrame.copy()
    if (hasDuplicates):
        dat = dropDuplicates(dat)
    for i in range(len(columns)):
        if columns[i] in dat.columns:
            dat = dat.drop(columns[i], axis=1)
    if ('MTM_DATE' in dat.columns):
        dat = dat.rename(index=str, columns={'MTM_DATE': "Date"})
    dat.Date = pd.to_datetime(dat.Date)
    return dat.sort_values('Date').copy()


# endregion nnData

def toCalibrationTuple(list):
    CalibrationData = namedtuple("CalibrationData", "opt,swap,volatility")
    data = [CalibrationData(float(x), float(y), float(z)) for x, y, z in list]
    return data


def getSampleData():
    tvol = pd.read_csv('../Data/swaptionEURvolClean.csv')
    tvol.drop('Unnamed: 0', axis=1)
    g = tvol.sort_values(['Date', 'X', 'Y'], ascending=True).groupby('Date')
    temp = g.get_group('2015-09-11').head(len(g.get_group('2015-09-11'))).as_matrix([['X', 'Y', 'Z']])
    calibrationVolatilities = toCalibrationTuple(temp)
    eur = pd.read_csv('../Data/EUR6M.csv')
    eur = shortClean(eur)
    teur = shortTenorProcess(eur, ['X'])
    g1 = teur.sort_values(['Date', 'X']).groupby('Date')

    series = g1.get_group('2015-09-11').head(50)
    spotRates = series.Z.as_matrix()
    evalDate = datetime.date(2015, 9, 11)
    spotDates = shortTenorToQLDate(evalDate, series)

    return spotDates, spotRates, calibrationVolatilities


def dropDuplicates(groupedData, subset=['PNT_ID'], forceUnique='MDE_ID'):
    if ('PNT_ID' not in groupedData.columns):
        return groupedData
    else:
        dat = groupedData.drop_duplicates(subset=subset, keep='first').copy()
        df = pd.DataFrame()
        g = dat.groupby('Date')
        for i in g.groups:
            temp = g.get_group(i)
            uniq = pd.unique(temp[forceUnique])
            if uniq.shape[0] > 1:
                temp = temp[temp[forceUnique] == uniq[0]]
            df = df.append(temp.copy())
        return df


def correlations(xDFrame, yDFrame, xGroup=['Term'], yGroup=['OptionTerm', 'SwapTerm'], valueTerm='Value'):
    '''
    Normally yDFrame and xDFrame should have same number of columns, if not we crop the larger
    :param xDFrame: e.g. IR curve
    :param yDFrame: e.g. swaption volatilities
    :param xGroup:
    :param yGroup:
    :param valueTerm:
    :return:
    '''
    # assert (xSeries.shape[1] == ySeries.shape[1]), "x and y series have the same number of columns"
    gx = xDFrame.groupby(xGroup)
    gy = yDFrame.groupby(yGroup)
    minLength = gx.get_group(list(gx.groups.keys())[0]).shape[0]
    if (minLength > gy.get_group(list(gy.groups.keys())[0]).shape[0]):
        minLength = gy.get_group(list(gy.groups.keys())[0]).shape[0]

    index = (np.arange(gy.first().shape[0] * gx.first().shape[0]))
    cols = [i for i in xGroup]
    [cols.append(i) for i in yGroup]
    cols.append('Correlation')
    cols.append('P-value')
    corr = pd.DataFrame(index=index, columns=cols)
    count = 0
    for i in gx.groups:
        datX = gx.get_group(i)[valueTerm][:minLength]
        for j in gy.groups:
            datY = gy.get_group(j)[valueTerm][:minLength]
            pearson = stats.pearsonr(datX, datY)
            temp = [x for x in i] if type(i) == tuple else [i]
            [temp.append(x) for x in j]
            temp.append(pearson[0])
            temp.append(pearson[1])
            corr.loc[count] = temp
            count += 1
            # print(i,'x',j,' = ',stats.pearsonr(datX,datY))
    return corr

    # df[df.someColumn.isin(someList)] #keep data that are also in somelist
    # df[~df.someColumn.isin(someList)] #keep data that are NOT in somelist

# Plot errors
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# ind=0
# with pd.HDFStore('../../../data/data.h5') as store:
#     er = store['/Errors/IR/SWO/GBP/Hull_White_analytic_formulae']
#     params = store['/Models/IR/SWO/GBP/Hull_White_analytic_formulae']
# emp = np.load("001sigmas.npy")
# first = np.load("firstSigmas.npy")
# long = np.load("147kVolatile_avgSigmas.npy")
# err = er.as_matrix()
# errA = []
# for i in range(err.shape[0]):
#     errA.append(np.average(np.abs(err[i])))
# dates = er.index.tolist()
# ddates = np.asarray(pd.to_datetime(dates[29:]))
# plt.figure()
# plt.grid()
# plt.plot(ddates, errA[29:], label='default QL error')
# plt.plot(ddates, emp[:, ind], label='0.001 error')
# plt.plot(ddates, long[:, ind], label='@147k error')
# plt.plot(ddates, first[:, ind], label='@6900k error')
# plt.legend(loc=9, bbox_to_anchor=(1.0, 1.0))


# import pandas as pd
# import numpy as np
# eun = pd.read_csv("data/eoniaFullFilled100.csv",header = 0)
# term = eun.Term.as_matrix()
# eur = eun.drop("Term",axis=1)
# eur.dropna(axis=0,how='any')
# dat = pd.DataFrame(columns=['Date', "Term", "Value"])
# for col in eur.columns:
#     datesList = [col for y in range(len(term))]
#     dd = np.asarray([pd.to_datetime(datesList), term, eur[col]]).T
#     temp = pd.DataFrame(dd, columns=['Date', "Term", "Value"])
#     dat = dat.append(temp)
