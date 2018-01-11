import pandas as pd
from sklearn.decomposition import PCA
from utils.customUtils import *
import pdb

data_dir = '../data/'
h5file = data_dir + 'data.h5'
h5_ts_node = 'TS'
h5_tsc_node = 'TSC'
swo_gbp_tskey = h5_ts_node + '/SWO/GBP'
ois_gbp_tskey = h5_ts_node + '/IRC/GBP/OIS'
l6m_gbp_tskey = h5_ts_node + '/IRC/GBP/L6M'

# 1-10, 15, 20, 25
dateInDays = {"swap": [365, 730, 1095, 1460, 1825, 2190, 2555, 2920, 3285, 3650, 5475, 7300, 9125],
              "libor": [0, 1, 2, 7, 14, 30, 61, 91, 122, 152, 182, 213, 243, 273, 304, 335, 365, 395, 425, 456,
                        486, 516, 547, 577, 607, 638, 668, 698, 730, 1095, 1460, 1825, 2190, 2555, 2920, 3285,
                        3650, 4380, 5475, 7300, 9125, 10950, 14600, 18250],
              "eonia": [30, 61, 91, 122, 152, 182, 213, 243, 273, 304, 335, 365, 547, 730, 1095, 1460, 1825,
                        2190, 2555, 2920, 3285, 3650, 4380, 5475, 7300, 9125, 10950, 12775, 14600, 18250],
              "clean": [30, 61, 91, 122, 152, 182, 213, 243, 273, 304, 335, 365]
              }


class TimeSeriesData(object):
    def __init__(self, key, file_name=h5file, data=None):
        # pdb.set_trace()
        if (data is None):
            self._data = from_hdf5(key, file_name)
        else:
            self._data = data

        fr = self._data.iloc[0]
        fd = self._data.loc[fr.name[0]].index
        if hasattr(fd, 'levels'):
            self._levels = fd.levels
            self._axisSize = tuple([len(x) for x in self._levels])
        else:
            self._levels = [fd]
            self._axisSize = (len(fd),)
        self._dates = self._data.index.levels[0]
        self._pipeline = None

    def __getitem__(self, ddate):
        return self.__getimpl(ddate)

    def __getimpl(self, ddate):
        data = self._data.loc[ddate].as_matrix()
        data.shape = self._axisSize
        return data

    def axis(self, i):
        return self._levels[i]

    def dates(self):
        return self._dates

    def intersection(self, rest):
        return self._dates.intersection(rest._dates)

    def to_matrix(self, *args):
        if len(args) > 0:
            dates = args[0]
        else:
            dates = self._data.index.levels[0]
        nbrDates = len(dates)
        mat = np.zeros((nbrDates,) + self._axisSize)
        for iDate in range(nbrDates):
            mat[iDate] = self.__getimpl(dates[iDate])

        return mat

    def pca(self, **kwargs):
        pdb.set_trace()
        if 'n_components' in kwargs:
            nComp = kwargs['n_components']
        else:
            nComp = 0.995

        if 'dates' in kwargs:
            mat = self.to_matrix(kwargs['dates'])
        else:
            mat = self.to_matrix()
        scaler = StandardScaler()
        pca = PCA(n_components=nComp)
        self._pipeline = Pipeline([('scaler', scaler), ('pca', pca)])
        self._pipeline.fit(mat)

        if 'file' in kwargs:
            tofile(kwargs['file'], self._pipeline)

        return self._pipeline


def read_csv(file_name):
    # For swaptions Data is assumed to come in the form
    # <Date, format YYYY-mm-dd>,<Option Term>,<Swaption Term>, <Value>
    # For term structures Data is assumed to come in the form
    # <Date>, <Term>, <Value>
    # /TS/SWO/GBP
    # /TS/IRC/GBP/L6M
    # /TS/IRC/GBP/OIS
    # /Models/IR/SWO/GBP/G2pp
    # /Models/IR/SWO/GBP/G2pp_local
    # /Models/IR/SWO/GBP/Hull_White_analytic_formulae
    # /Errors/IR/SWO/GBP/G2pp
    # /Errors/IR/SWO/GBP/G2pp_local
    # /Errors/IR/SWO/GBP/Hull_White_analytic_formulae
    cols = pd.read_csv(file_name, nrows=1).columns
    return pd.read_csv(file_name, parse_dates=[0], infer_datetime_format=True, index_col=cols.tolist()[:-1])


def store_hdf5(file_name, key, val):
    with pd.HDFStore(file_name) as store:
        store[key] = val
        store.close()


def csv_to_hdf5(file_name, key, hdf5file_name):
    res = read_csv(file_name)
    store_hdf5(hdf5file_name, key, res)


def from_hdf5(key, file_name=h5file):
    with pd.HDFStore(file_name) as store:
        return store[key]


def tofile(file_name, model):
    joblib.dump(model, file_name)


def fromfile(file_name):
    return joblib.load(file_name)


def gbp_to_hdf5():
    swo_csvfile = data_dir + 'swaption_gbp_20130101_20160601.csv'
    ois_csvfile = data_dir + 'ois_gbp_20130101_20160601.csv'
    l6m_csvfile = data_dir + 'libor_6m_gbp_20130101_20160601.csv'
    csv_to_hdf5(swo_csvfile, swo_gbp_tskey, h5file)
    csv_to_hdf5(ois_csvfile, ois_gbp_tskey, h5file)
    csv_to_hdf5(l6m_csvfile, l6m_gbp_tskey, h5file)
