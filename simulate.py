import QuantLib as ql
import matplotlib.pyplot as plt
import numpy as np
import models.IRCurve as irc
import pdb
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def generate_paths(num_paths, timestep, sequence):
    arr = np.zeros((num_paths, timestep + 1))
    for i in range(num_paths):
        sample_path = sequence.next()
        path = sample_path.value()
        time = [path.time(j) for j in range(len(path))]
        value = [path[j] for j in range(len(path))]
        arr[i, :] = np.array(value)
    return np.array(time), arr


def calcMean(forward, sigma, a, time):
    fw = []
    for t in time:
        fw.append(forward.zeroRate(t, ql.Continuous).rate())
    mm = fw + (1 / 2.0) * np.power(sigma / a * (1.0 - np.exp(-a * time)), 2)
    return mm


def runSimulations(predictive_model, modelName, swo=None, dataLength=1, simNumber=1000):
    dateList = [swo._dates[430]]  # [swo._dates[50], swo._dates[430], swo._dates[550], swo._dates[650]]
    for l in dateList:
        ddate = l
        refDate = ql.Date(ddate.day, ddate.month, ddate.year)
        months = 50
        timestep = months * 30
        length = (months * 30) / 365  # in years
        # params, ir = runModel(swo, ddate, dataLength, 0, predictive_model)
        # alpha = params[0]
        # sigma = params[1]
        swo.set_date(l)
        ir = swo._ircurve
        avg = np.zeros((4, 1 + (months * 30)))
        a = [0.0001, 0.001, 0.01, 0.1]
        for i in range(len(a)):
            alpha = a[i]
            sigma = 0.01
            # levels = np.asarray(ir._levels)[0]
            # fw = ir.curveToArray(levels, ir[ddate])
            # plt.plot(fw)
            # forward_rate = ir.curveimpl(refDate, fw)
            ql.Settings.instance().evaluationDate = refDate
            # spot_curve = ir[ddate]
            spot_curve = ql.FlatForward(refDate, ql.QuoteHandle(ql.SimpleQuote(0.05)), ql.Actual360())
            spot_curve_handle = ql.YieldTermStructureHandle(spot_curve)
            hw_process = ql.HullWhiteProcess(spot_curve_handle, alpha, sigma)
            rng = ql.GaussianRandomSequenceGenerator(
                ql.UniformRandomSequenceGenerator(timestep, ql.UniformRandomGenerator()))
            seq = ql.GaussianPathGenerator(hw_process, length, timestep, rng, False)
            time, paths = generate_paths(simNumber, timestep, seq)
            # runModel(paths, predictive_model)
            # pdb.set_trace()
            avg[i] = np.average(paths, axis=0)
            plt.plot(avg[i],label=a[i])
        plt.legend()

        print(((avg[0]-avg[1])**2).mean())
        print(((avg[1] - avg[2]) ** 2).mean())
        print(((avg[2] - avg[3]) ** 2).mean())
        # print(np.average(paths,axis=0))
        # plotSimulations(paths[:int(np.floor(len(paths) / 2))], timestep, time)
        # plotVariance(paths, timestep, time, alpha, sigma)
        # plotMean(paths, timestep, time, alpha, sigma, forward_rate)


def runModel(paths, predictive_model):
    # paths = np.average(paths, axis=0)
    # paths = np.repeat(paths.reshape(-1,1)[:], 44, axis=1)
    sc = StandardScaler()
    # scaled = sc.fit_transform(paths.reshape(-1, 1))
    # scaled = np.repeat(scaled[:], 44, axis=1)

    method = ql.LevenbergMarquardt()
    end_criteria = ql.EndCriteria(250, 200, 1e-7, 1e-7, 1e-7)
    constraint = ql.PositiveConstraint()
    der = np.zeros((paths.shape[0], 6))
    for j in range(paths.shape[0]):
        scaled = sc.fit_transform(paths[j, :].reshape(-1, 1))
        scaled = np.repeat(scaled[:], 44, axis=1)
        params = []
        s = 30
        for i in range(0, scaled.shape[0] - s, s):
            dataDict = {'vol': np.empty((0, scaled.shape[1])), 'ir': scaled[i:i + s, :]}
            params.append(predictive_model.predict(vol=dataDict['vol'], ir=dataDict['ir']))
        der[j] = params
    print(params)
    pdb.set_trace()
    return params


# def runModel(swo, refDate, dataLength, skip, predictive_model):
#     dates = swo._dates
#     method = ql.LevenbergMarquardt()
#     end_criteria = ql.EndCriteria(250, 200, 1e-7, 1e-7, 1e-7)
#     constraint = ql.PositiveConstraint()
#     swo.set_date(dates[0])
#     dataDict = {'vol': np.empty((0, swo.values.shape[0])), 'ir': np.empty((0, swo._ircurve.values.shape[0]))}
#     for i, ddate in enumerate(dates):
#         swo.set_date(ddate)
#         if (i < skip):
#             if (i + dataLength - 1 >= skip):
#                 dataDict['vol'] = np.vstack((dataDict['vol'], swo.values))
#                 dataDict['ir'] = np.vstack((dataDict['ir'], swo._ircurve.values))
#             continue
#         if (i + 1 < dataLength):
#             dataDict['vol'] = np.vstack((dataDict['vol'], swo.values))
#             dataDict['ir'] = np.vstack((dataDict['ir'], swo._ircurve.values))
#             continue
#
#         dataDict['vol'] = np.vstack((dataDict['vol'], swo.values))
#         dataDict['ir'] = np.vstack((dataDict['ir'], swo._ircurve.values))
#         if (refDate == ddate):
#             if (type(predictive_model) == list):
#                 out = []
#                 for j in range(len(predictive_model)):
#                     out.append(predictive_model[j].predict(vol=dataDict['vol'], ir=dataDict['ir'][:, j:j + 1])[0])
#                 params = np.abs(np.average(out)).reshape((-1, 1))
#             else:
#                 params = np.abs(predictive_model.predict(vol=dataDict['vol'], ir=dataDict['ir']))
#         dataDict['vol'] = np.delete(dataDict['vol'], (0), axis=0)
#         dataDict['ir'] = np.delete(dataDict['ir'], (0), axis=0)
#         if (refDate == ddate):
#             params = [[params[0, 0], 0]]  # shape (1,2)
#             swo.model.setParams(ql.Array(params[0]))
#             swo.model.calibrate(swo.helpers, method, end_criteria, constraint, [], [True, False])
#             paramsC = np.asarray(swo.model.params())
#             break
#
#     return paramsC, swo._ircurve


def plotSimulations(paths, timestep, time):
    num_paths = len(paths)
    for i in range(num_paths):
        plt.plot(time, paths[i, :], lw=0.8, alpha=0.6)
    plt.title("Hull-White Short Rate Simulation")


def plotVariance(paths, timestep, time, a, sigma):
    plt.figure()
    vol = [np.var(paths[:, i]) for i in range(timestep + 1)]
    plt.plot(time, vol, "r-.", lw=3, alpha=0.6, label="simulation")
    plt.plot(time, sigma * sigma / (2 * a) * (1.0 - np.exp(-2.0 * a * np.array(time))), "b-", lw=2, alpha=0.5,
             label="model")
    plt.title("Variance of Short Rates")
    plt.legend()


def plotMean(paths, timestep, time, a, sigma, forwardRate):
    plt.figure()
    avg = [np.mean(paths[:, i]) for i in range(timestep + 1)]
    plt.plot(time, avg, "r-.", lw=3, alpha=0.6, label="simulation")
    plt.plot(time, calcMean(forwardRate, sigma, a, time), "b-", lw=2, alpha=0.6, label="model")
    plt.title("Mean of Short Rates")
    plt.legend()
