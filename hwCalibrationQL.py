import QuantLib as ql
from collections import namedtuple
import math

def create_swaption_helpers(data, index, term_structure, engine):
    swaptions = []
    fixed_leg_tenor = ql.Period(1, ql.Years)
    fixed_leg_daycounter = ql.Actual360()
    floating_leg_daycounter = ql.Actual360()
    for d in data:
        vol_handle = ql.QuoteHandle(ql.SimpleQuote(d.volatility))
        helper = ql.SwaptionHelper(ql.Period(d.start, ql.Years),
                                   ql.Period(d.length, ql.Years),
                                   vol_handle,
                                   index,
                                   fixed_leg_tenor,
                                   fixed_leg_daycounter,
                                   floating_leg_daycounter,
                                   term_structure
                                   )
        helper.setPricingEngine(engine)
        swaptions.append(helper)
    return swaptions    

def calibration_report(swaptions, data):
    print ("-"*82)
    print ("%15s %15s %15s %15s %15s" % \
    ("Model Price", "Market Price", "Implied Vol", "Market Vol", "Rel Error"))
    print ("-"*82)
    cum_err = 0.0
    for i, s in enumerate(swaptions):
        model_price = s.modelValue()
        market_vol = data[i].volatility
        black_price = s.blackPrice(market_vol)
        rel_error = model_price/black_price - 1.0
        implied_vol = s.impliedVolatility(model_price,
                                          1e-5, 50, 0.0, 0.50)
        rel_error2 = implied_vol/market_vol-1.0
        cum_err += rel_error2*rel_error2
        
        print ("%15.5f %15.5f %15.5f %15.5f %15.5f" % \
        (model_price, black_price, implied_vol, market_vol, rel_error))
    print ("-"*82)
    print ("Cumulative Error : %15.5f" % math.sqrt(cum_err))

def runTest():
    today = ql.Date(17, ql.October, 2017)
    settlement= ql.Date(23,ql.October,2017)
    ql.Settings.instance().evaluationDate = today
    term_structure = ql.YieldTermStructureHandle(
        ql.FlatForward(settlement,0.04875825,ql.Actual365Fixed())
        )
    index = ql.Euribor1Y(term_structure)
    CalibrationData = namedtuple("CalibrationData", 
                                "start, length, volatility")
    data = [CalibrationData(1, 5, 0.1148),
            CalibrationData(2, 4, 0.1108),
            CalibrationData(3, 3, 0.1070),
            CalibrationData(4, 2, 0.1021),
            CalibrationData(5, 1, 0.1000 )]

    #For known spot rates
    # ql.Settings.instance().evaluationDate = today
    # spotDates = [ql.Date(15, 1, 2015), ql.Date(15, 7, 2015), ql.Date(15, 1, 2016)]
    # spotRates = [0.0, 0.005, 0.007]
    # dayCount = ql.Thirty360()
    # calendar = ql.UnitedStates()
    # interpolation = ql.Linear()
    # compounding = ql.Compounded
    # compoundingFrequency = ql.Annual
    # spotCurve = ql.ZeroCurve(spotDates, spotRates, dayCount, calendar, interpolation,compounding, compoundingFrequency)
    # spotCurveHandle = ql.YieldTermStructureHandle(spotCurve)

    #Calibrating mean reversion and volatility
    model = ql.HullWhite(term_structure)
    engine = ql.JamshidianSwaptionEngine(model)
    swaptions = create_swaption_helpers(data, index, term_structure, engine)

    optimization_method = ql.LevenbergMarquardt(1.0e-8,1.0e-8,1.0e-8)
    end_criteria = ql.EndCriteria(10000, 100, 1e-6, 1e-8, 1e-8)
    model.calibrate(swaptions, optimization_method, end_criteria)

    a, sigma = model.params()
    print ("a = %6.5f, sigma = %6.5f" % (a, sigma))
    calibration_report(swaptions, data)

    #Calibrating Volatility With Fixed Reversion
    constrained_model = ql.HullWhite(term_structure, 0.05, 0.001)
    engine = ql.JamshidianSwaptionEngine(constrained_model)
    swaptions = create_swaption_helpers(data, index, term_structure, engine)

    optimization_method = ql.LevenbergMarquardt(1.0e-8,1.0e-8,1.0e-8)
    end_criteria = ql.EndCriteria(10000, 100, 1e-6, 1e-8, 1e-8)
    constrained_model.calibrate(swaptions, optimization_method, end_criteria, ql.NoConstraint(), [], [True, False])

    a, sigma = constrained_model.params()
    print ("a = %6.5f, sigma = %6.5f" % (a, sigma))
    calibration_report(swaptions, data)

    #Black Karasinski Model
    model = ql.BlackKarasinski(term_structure)
    engine = ql.TreeSwaptionEngine(model, 100)
    swaptions = create_swaption_helpers(data, index, term_structure, engine)

    optimization_method = ql.LevenbergMarquardt(1.0e-8,1.0e-8,1.0e-8)
    end_criteria = ql.EndCriteria(10000, 100, 1e-6, 1e-8, 1e-8)
    model.calibrate(swaptions, optimization_method, end_criteria)

    a, sigma =  model.params()
    print ("a = %6.5f, sigma = %6.5f" % (a, sigma))
    calibration_report(swaptions, data)