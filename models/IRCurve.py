import string, pdb
from utils.ahUtils import *
from utils.customUtils import getImpliedForwardCurve
import utils.dbDataPreprocess as dbc
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

seed = 1027
h5_irc_node = 'IRC'


class IRCurve(du.TimeSeriesData):
    def __init__(self, ccy, tenor, parentNode=du.h5_ts_node, data=None):
        if tenor in string.digits:
            tenor = '_' + tenor
        self.ccy = ccy.upper()
        self.tenor = tenor
        self.name = h5_irc_node + '_' + ccy + '_' + tenor
        self.name = self.name.lower()
        self.key_ts = parentNode + '/' + h5_irc_node + '/' + self.ccy + '/' + tenor.upper()
        if (ccy.lower() == 'gbp'):
            self._daycounter = ql.Actual365Fixed()
        else:
            self._daycounter = ql.Actual360()
        self.values = None
        self.curveArray = None
        super(IRCurve, self).__init__(self.key_ts, file_name=du.h5file, data=data)

    def __getitem__(self, date):
        data = super(IRCurve, self).__getitem__(date)
        ts = pd.Timestamp(date)
        refdate = ql.Date(ts.day, ts.month, ts.year)
        return self.__curveimpl(refdate, data)

    def getValues(self, date):
        data = super(IRCurve, self).__getitem__(date)
        return data

    def rebuild(self, date, vals):
        if self._pipeline is None:
            self.pca()
        refdate = ql.Date(date.day, date.month, date.year)
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

    def curveimpl(self, refdate, values):
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
        pdb.set_trace()
        x.forwardRate()
        expSums = np.zeros((len(levels)))
        # intergrating
        from scipy.integrate import simps
        for i in range(0, len(levels)):
            cSum = np.zeros((50))
            steps = 50
            if (levels[i] == 0):
                sampleX = [0]
            else:
                sampleX = np.linspace(0, levels[i], steps)
            sampleDates = [t0 + ql.Period(int(num), ql.Days) + delta for num in sampleX]
            for ddate in sampleDates:
                if (ddate > x.maxDate()):
                    ddate = x.maxDate()
                cSum[i] = x.zeroRate(ddate, x.dayCounter(), ql.Continuous).rate()
            if (levels[i] == 0):
                integral = np.exp(-cSum[0])
            else:
                integral = np.exp(-simps(cSum, sampleX))
            expSums[i] = integral
        return expSums, np.exp(-zDelta)

    def getHWForwardRate(self, ddate, T, deltaT):
        z1, zDelta = self.zeta(self.__getitem__(ddate), T)
        fw = (-np.log(z1) - np.log(zDelta)) / deltaT
        return fw

    def calibrateStatic(self):
        import statsmodels.api as sm
        lr = LinearRegression()
        curves = np.asarray(self.getAll())
        irPre = curves[:, :curves.shape[1] - 1]
        irAft = curves[:, 1:]
        levelParams = []
        print("Fitting params")
        for i in range(irPre.shape[0]):
            # lr.fit(irPre[i, :50].reshape(-1, 1), irAft[i, :50].reshape(-1, 1))
            # levelParams.append([lr.coef_[0, 0], lr.intercept_[0]])
            ls = sm.OLS(irAft[i, :50].reshape(-1, 1), irPre[i, :50].reshape(-1, 1))
            res = ls.fit()
            levelParams.append([res.params[0], res.bse[0]])
            # pdb.set_trace()
            # res = ls.fit_regularized(alpha=0.00001)
            # levelParams.append(res.params[0])
        levelParams = np.asarray(levelParams)
        pdb.set_trace()
        return levelParams[:, 0], levelParams[:, 1]

    def calibrateStaticAlpha(self, start, end, diff=80, short=False):
        lr = LinearRegression()
        if (self.curveArray is None):
            self.curveArray = np.asarray(self.getAll())
        irPre = self.curveArray[:, :self.curveArray.shape[1] - 1]
        irAft = self.curveArray[:, 1:]
        if end > irAft.shape[1]:
            end = irAft.shape[1]
            start = irAft.shape[1] - diff
        levelParams = []
        sigma = []
        maturities = irPre.shape[0]
        if (short):
            maturities = 1  # use first maturity only
        for i in range(maturities):
            # lr.fit(irPre[i, start:end].reshape(-1, 1), irAft[i, start:end].reshape(-1, 1))
            # levelParams.append(lr.coef_[0, 0])
            # sigma.append(lr.intercept_[0])
            levelParams.append(np.corrcoef(irPre[i, start:end], irAft[i, start:end])[0, 1])
        levelParams = np.average(1 - np.asarray(levelParams), axis=0)

        print(str(start) + "-" + str(end) + " Fit params: ", levelParams)
        return levelParams, np.asarray(sigma)

    # paper http://www.ressources-actuarielles.net/EXT/ISFA/1226.nsf/0/b92869fc0331450dc1256dc500576be4/$FILE/SEPP%20numerical%20implementation%20Hull&White.pdf
    def calcThetaHW(self, path=None, prime=False, skip=30):
        thetaFrame = pd.DataFrame(columns=["Date", "Term", "Value"])
        deltaFrame = pd.DataFrame(columns=["Date", "Term", "Value"])
        levels = np.asarray(self._levels)[0]
        firstCurve = self.__getitem__(self._dates[0])
        T = levels
        theta = np.zeros((len(self._dates), len(levels)))
        if (prime):
            calcTheta = self._getPrime
            calcAlpha = lambda x, y, z: None
            skip = -1
        else:
            calcTheta = self._getTheta
            calcAlpha = self.calibrateStaticAlpha

        for i, ddate in enumerate(self._dates):
            if (i <= skip):
                continue
            alpha = calcAlpha(i - skip, i, skip)
            curve = self.__getitem__(ddate)
            theta[i] = calcTheta(levels, curve, alpha)
            refDate = pd.to_datetime(ddate)
            tRate = [(refDate, T[j], theta[i][j]) for j in range(len(T))]

            try:
                nextDate = self._dates[i + 1]
                nextFrame = self._getSubOfNext(levels, ddate, nextDate, theta[i], deltaFrame.columns.tolist())
                deltaFrame = deltaFrame.append(nextFrame)
            except:
                pass
            thetaFrame = thetaFrame.append(pd.DataFrame(tRate, columns=thetaFrame.columns.tolist()))
        if (path is not None):
            thetaFrame.to_csv(path, index=False)
            deltaFrame.to_csv(path + "delta", index=False)

        return thetaFrame, theta

    def getInstForward(self, path=None, skip=0):
        currentInstFw = pd.DataFrame(columns=["Date", "Term", "Value"])
        levels = np.asarray(self._levels)[0]
        instForward = np.zeros((len(self._dates), len(levels)))
        for i, ddate in enumerate(self._dates):
            if (i <= skip):
                continue
            try:
                curve = self.__getitem__(ddate)
                fwT = self.curveToArray(levels, curve, instant=True)
                nextFrame = self._getSubOfNext(levels, ddate, ddate, fwT, currentInstFw.columns.tolist())
                currentInstFw = currentInstFw.append(nextFrame)
            except:
                pass
        if (path is not None):
            currentInstFw.to_csv(path, index=False)

        return currentInstFw

    def _getSubOfNext(self, levels, currentDate, nextDate, rateList, columns):
        rT1 = self.__getitem__(nextDate)
        refDate = pd.to_datetime(currentDate)
        deltaRate = [(refDate, levels[j], (rT1.nodes()[j][1] - rateList[j])) for j in range(len(levels))]
        deltaFrame = pd.DataFrame(deltaRate, columns=columns)
        return deltaFrame

    def _getTheta(self, levels, curve, alpha=0.01):
        fw = self.curveToArray(levels, curve)
        dtFw = self._getPrime(levels, curve)
        theta = dtFw + alpha * fw
        return theta

    def _getPrime(self, levels, curve, alpha=None):
        shift = 0.0001
        fwMinus = self.curveToArray(levels, curve, instant=True, delta=(-1 * shift))
        fwPlus = self.curveToArray(levels, curve, instant=True, delta=shift)
        prime = (fwPlus - fwMinus) / (2 * shift)
        # pdb.set_trace()
        return prime

    def curveToArray(self, levels, curve, instant=False, delta=0):
        # Delta is expected to be in dayss
        # deltaY = delta / 365
        firstDate = curve.dates()[0]
        fwRates = []
        for T in levels.tolist():
            if (T <= 365):
                start = 0
                if (firstDate + ql.Period(T + int(np.round(delta, decimals=0)), ql.Days) < firstDate or T + delta < 0):
                    delta = 0
            else:
                start = (T - 365) / 365

            tenor = (T / 365.0) if (T / 365.0) <= curve.maxTime() else curve.maxTime()

            if delta >= 1:
                deltaY = delta / 365  # or 0.0001
            else:
                deltaY = delta
            end = tenor + deltaY
            if (start >= end):
                end = tenor
            if (instant):
                fwRates.append(curve.forwardRate(tenor + delta, tenor + delta, ql.Continuous).rate())
            else:
                fwRates.append(curve.forwardRate(start, end, ql.Continuous).rate())

        return np.asarray(fwRates)

    def calibrate_alpha(self, predictive_model, modelName, dates=None, dataLength=1, skip=0, plot=False):
        outcome = []
        outcomeStatic = []
        if dates is None:
            dates = self._dates
        ir = np.zeros((dataLength, self.getValues(dates[0]).shape[0]))
        print('Calibrating mean reversion on historic data...')
        for i, ddate in enumerate(dates):
            if (skip == -1):
                if (i % 80 == 0):
                    alpha = self.calibrateStaticAlpha(i, i + 80)
            if (i < skip):
                if (i + dataLength - 1 >= skip):
                    ir[i] = self.getValues(ddate)
                continue
            if (i + 1 < dataLength):
                ir[i] = self.getValues(ddate)
                continue
            if (dataLength == 1):
                params = predictive_model.predict(vol=None, ir=ir)
            else:
                ir[-1] = self.getValues(ddate)
                if (type(predictive_model) == list):
                    out = []
                    for j in range(len(predictive_model)):
                        out.append(predictive_model[j].predict(vol=None, ir=ir[:, j:j + 1])[0])
                    params = np.average(out).reshape((-1, 1))
                else:
                    params = predictive_model.predict(vol=None, ir=ir)
                ir[:-1] = ir[1:]

            params = [params[0, 0]]
            outcome.append(params)

        if (len(outcomeStatic) > 1):
            np.save("sigmaStatic30.npy", outcomeStatic)
        outcome = np.asarray(outcome).reshape(-1)
        outcome = np.append(np.zeros(dataLength - 1), outcome)
        if (plot):
            plt.plot(outcome, label=modelName)

        print("Finished calibrating mean reversion")
        return outcome


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
            index = ql.USDLibor(ql.Period(3, ql.Months))
        elif (str(irType).lower() == 'ois'):
            index = ql.FedFunds()

    if (irFileName is None):
        irc = IRCurve(index.currency().code(), to_tenor(index, irType), parentNode=pNode)
    else:
        irc = IRCurve(index.currency().code(), to_tenor(index, irType), parentNode=pNode,
                      data=dbc.toAHFileFormat(irFileName))
    return irc
