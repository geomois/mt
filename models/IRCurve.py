import string, pdb
from utils.ahUtils import *
from utils.customUtils import getImpliedForwardCurve
import utils.dbDataPreprocess as dbc

seed = 1027
# Interest Rate Curve hdf5 node
h5_irc_node = 'IRC'


class IRCurve(du.TimeSeriesData):
    '''Class for accessing IR curve data and instantianting QuantLib instances
    The curves are instantiated as interpolated zero curves using the monotonic
    cubic interpolation.
    '''

    def __init__(self, ccy, tenor, parentNode=du.h5_ts_node, data=None):
        if tenor in string.digits:
            tenor = '_' + tenor
        self.ccy = ccy.upper()
        self.tenor = tenor
        self.name = h5_irc_node + '_' + ccy + '_' + tenor
        self.name = self.name.lower()
        self.key_ts = parentNode + '/' + h5_irc_node + '/' + self.ccy + '/' + tenor.upper()
        self._daycounter = ql.Actual360()
        self.values = None
        super(IRCurve, self).__init__(self.key_ts, file_name=du.h5file, data=data)

    def __getitem__(self, date):
        data = super(IRCurve, self).__getitem__(date)
        ts = pd.Timestamp(date)
        refdate = ql.Date(ts.day, ts.month, ts.year)
        return self.__curveimpl(refdate, data)

    def rebuild(self, date, vals):
        if self._pipeline is None:
            self.pca()
        refdate = ql.Date(date.day, date.month, date.year)
        # data = self._pipeline.inverse_transform(vals)[0] OLD
        data = self._pipeline.inverse_transform(vals)[:]
        return (data, self.__curveimpl(refdate, data))

    def build(self, date, data):
        return self.__curveimpl(date, data)

    def __curveimpl(self, refdate, values):
        self.values = values
        ql.Settings.instance().evaluationDate = refdate
        dates = [refdate + int(d) for d in self.axis(0)]
        # pdb.set_trace()
        return ql.MonotonicCubicZeroCurve(dates, values, self._daycounter)

    def calcForward(self, path=None, futureIncrementInDays=180):
        fwCurves = pd.DataFrame(columns=["Date", "FutureDate", "Tenor", "Rate"])
        levels = np.asarray(self._levels)[0]
        for ddate in self._dates:
            ts = pd.Timestamp(ddate)
            refDate = ql.Date(ts.day, ts.month, ts.year)
            futureDate = refDate + ql.Period(futureIncrementInDays, ql.Days)
            curve = getImpliedForwardCurve(futureDate, self.__getitem__(ddate))
            fwRates = []
            for T in levels:
                tenor = (T / 365.0) if (T / 365.0) <= curve.maxTime() else curve.maxTime()
                refD = pd.to_datetime(ql.Date.to_date(refDate))
                fD = pd.to_datetime(ql.Date.to_date(futureDate))
                fwRates.append((refD, fD, T, curve.zeroRate(tenor, ql.Continuous).rate()))
            fwCurves = fwCurves.append(pd.DataFrame(fwRates, columns=fwCurves.columns.tolist()))
        if (path is not None):
            fwCurves.to_csv(path, index=False)


def getIRCurves(modelMap=hullwhite_analytic, currency='GBP', irType='Libor', pNode=du.h5_ts_node, irFileName=None):
    # 'GBP','EUR','USD'
    # 'libor','euribor','ois'
    index = None
    if (str(currency).lower() == 'gbp'):
        if (str(irType).lower() == 'libor'):
            index = ql.GBPLibor(ql.Period(6, ql.Months))
        elif (str(irType).lower() == 'ois'):
            index = ql.Sonia()
    elif (str(currency).lower() == 'eur'):
        if (str(irType).lower() == 'euribor'):
            index = ql.Euribor(ql.Period(6, ql.Months))
        elif (str(irType).lower() == 'ois'):
            index = ql.Eonia()
    elif (str(currency).lower() == 'usd'):
        if (str(irType).lower() == 'libor'):
            index = ql.USDLibor(ql.Period(6, ql.Months))
        elif (str(irType).lower() == 'ois'):
            index = ql.FedFunds()

    if (irFileName is None):
        irc = IRCurve(index.currency().code(), to_tenor(index, irType), parentNode=pNode)
    else:
        irc = IRCurve(index.currency().code(), to_tenor(index, irType), parentNode=pNode,
                      data=dbc.toAHFileFormat(irFileName))
    return irc
