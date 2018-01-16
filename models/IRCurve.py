import string, pdb
from utils.ahUtils import *
from utils.customUtils import getImpliedForwardCurve
import utils.dbDataPreprocess as dbc
from sklearn.linear_model import LinearRegression

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

    def calcForward(self, path=None, futureIncrementInDays=365):
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
        return fwCurves

    @staticmethod
    def zeta(x, T):
        levels = T
        t0 = x.dates()[0]
        delta = ql.Period(1, ql.Days)

        zDelta = x.zeroRate(t0 + delta, x.dayCounter(), ql.Continuous).rate()
        expSums = np.zeros((len(levels)))
        for i in range(1, len(levels) + 1):
            cSum = 0
            for j in levels[:i]:  # include last tenor
                ddate = t0 + ql.Period(int(j), ql.Days) + delta
                if (ddate > x.maxDate()):
                    ddate = x.maxDate()
                cSum += x.zeroRate(ddate, x.dayCounter(), ql.Continuous).rate()
            cSum = np.exp(-cSum)
            expSums[i - 1] = cSum
        return expSums, np.exp(-zDelta)

    def getHWForwardRate(self, ddate, T, deltaT):
        z1, zDelta = self.zeta(self.__getitem__(ddate), T)
        fw = (-np.log(z1) - np.log(zDelta)) / deltaT
        return fw

    def calibrateStatic(self):
        import statsmodels.api as sm
        # lr = LinearRegression()
        curves = np.asarray(self.getAll())
        irPre = curves[:, :curves.shape[1] - 1]
        irAft = curves[:, 1:]
        levelParams = []
        print("Fitting params")
        pdb.set_trace()
        for i in range(irPre.shape[0]):
        	ls = sm.OLS(irPre[i,:200].reshape(-1, 1), irAft[i,:200].reshape(-1, 1))
        	res = ls.fit()
        	levelParams.append([res.params[0], res.bse[0]])
        levelParams = np.asarray(levelParams)
        return levelParams[:, 0], levelParams[:, 1]

    # paper http://www.ressources-actuarielles.net/EXT/ISFA/1226.nsf/0/b92869fc0331450dc1256dc500576be4/$FILE/SEPP%20numerical%20implementation%20Hull&White.pdf
    def calcThetaHW(self, path=None):
        thetaFrame = pd.DataFrame(columns=["Date", "Tenor", "Rate"])
        deltaT = 1 # 0.00277778  # 1 Day
        levels = np.asarray(self._levels)[0]
        firstCurve = self.__getitem__(self._dates[0])
        T = levels
        # for t in levels:
        # T.append((t / 365.0) if (t / 365.0) <= firstCurve.maxTime() else firstCurve.maxTime())

        alpha0, sigma = self.calibrateStatic()
        pdb.set_trace()
        alpha = -np.log(alpha0) / deltaT
        sigma = np.power(sigma,2) *(-2 * np.log(alpha0)) / deltaT * (1 - np.power(alpha, 2))
        theta = np.zeros((len(self._dates), len(levels)))
        for i in range(len(self._dates)):
            ddate = self._dates[i]
            fw = self.getHWForwardRate(ddate, T, deltaT)
            fwPlus = self.getHWForwardRate(ddate, T + deltaT, deltaT)
            fwMinus = self.getHWForwardRate(ddate, T - deltaT, deltaT)
            add1 = alpha * fw
            add2 = (sigma / (2 * alpha)) * (1 - np.exp(-alpha * deltaT))  # (T/365.0)
            dtFw = (fwPlus - fwMinus) / (2 * deltaT)
            pdb.set_trace()
            theta[i] = dtFw + add1 + add2
            refDate = pd.to_datetime(ddate)
            tRate = [(refDate, T[j], theta[i][j]) for j in range(len(T))]
            thetaFrame = thetaFrame.append(pd.DataFrame(tRate, columns=thetaFrame.columns.tolist()))
        pdb.set_trace()
        if (path is not None):
            thetaFrame.to_csv(path, index=False)
        return thetaFrame


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
