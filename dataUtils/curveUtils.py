import QuantLib as ql
import matplotlib.pyplot as plt
from datetime import date
from matplotlib.dates import MonthLocator, DateFormatter
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
