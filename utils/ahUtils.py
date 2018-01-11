import numpy as np
import QuantLib as ql
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import empirical_covariance
from scipy.linalg import sqrtm
import pandas as pd
import utils.data_utils as du

seed = 1027

h5_model_node = 'Models'
h5_error_node = 'Errors'
'''
Sampling functions
'''


def random_normal_draw(history, nb_samples, **kwargs):
    """Random normal distributed draws

    Arguments:
        history: numpy 2D array, with history along axis=0 and parameters
            along axis=1
        nb_samples: number of samples to draw

    Returns:
        numpy 2D array, with samples along axis=0 and parameters along axis=1
    """
    scaler = StandardScaler()
    scaler.fit(history)
    scaled = scaler.transform(history)
    # sqrt_cov = sqrtm(empirical_covariance(scaled)).real
    sqrt_cov = float(sqrtm(empirical_covariance(scaled)))

    # Draw correlated random variables
    # draws are generated transposed for convenience of the dot operation
    # pdb.set_trace()
    draws = np.random.standard_normal((history.shape[1], nb_samples))
    draws = np.dot(sqrt_cov, draws)
    draws = np.transpose(draws)
    return scaler.inverse_transform(draws)


'''
Dictionary defining model
It requires 4 parameters:
    name
    model: a function that creates a model. It takes as parameter a 
            yield curve handle
    engine: a function that creates an engine. It takes as paramters a
            calibration model and a yield curve handle
    sampler: a sampling function taking a history and a number of samples
            to produce

Optional parameters:
    transformation: a preprocessing function
    inverse_transformation: a postprocessing function
    method: an optimization object

'''
hullwhite_analytic = {'name': 'Hull-White (analytic formulae)',
                      'model': ql.HullWhite,
                      'engine': ql.JamshidianSwaptionEngine,
                      'transformation': np.log,
                      'inverse_transformation': np.exp,
                      'sampler': random_normal_draw}


def g2_transformation(x):
    if isinstance(x, pd.DataFrame):
        x = x.values
    y = np.zeros_like(x)
    y[:, :-1] = np.log(x[:, :-1])
    y[:, -1] = x[:, -1]
    return y


def g2_inverse_transformation(x):
    y = np.zeros_like(x)
    y[:, :-1] = np.exp(x[:, :-1])
    y[:, -1] = np.clip(x[:, -1], -1.0, +1.0)
    return y


def g2_method():
    n = 5
    lower = ql.Array(n, 1e-9);
    upper = ql.Array(n, 1.0);
    lower[n - 1] = -1.0;
    upper[n - 1] = 1.0;
    constraint = ql.NonhomogeneousBoundaryConstraint(lower, upper);

    maxSteps = 5000;
    staticSteps = 600;
    initialTemp = 50.0;
    finalTemp = 0.001;
    sampler = ql.SamplerMirrorGaussian(lower, upper, seed);
    probability = ql.ProbabilityBoltzmannDownhill(seed);
    temperature = ql.TemperatureExponential(initialTemp, n);
    method = ql.MirrorGaussianSimulatedAnnealing(sampler, probability, temperature,
                                                 ql.ReannealingTrivial(),
                                                 initialTemp, finalTemp)
    criteria = ql.EndCriteria(maxSteps, staticSteps, 1.0e-8, 1.0e-8, 1.0e-8);
    return (method, criteria, constraint)


def g2_method_local():
    method = ql.LevenbergMarquardt()
    criteria = ql.EndCriteria(250, 200, 1e-7, 1e-7, 1e-7)
    lower = ql.Array(5, 1e-9)
    upper = ql.Array(5, 1.0)
    lower[4] = -1.0
    constraint = ql.NonhomogeneousBoundaryConstraint(lower, upper)
    return (method, criteria, constraint)


g2 = {'name': 'G2++',
      'model': ql.G2,
      'engine': lambda model, _: ql.G2SwaptionEngine(model, 6.0, 16),
      'transformation': g2_transformation,
      'inverse_transformation': g2_inverse_transformation,
      'method': g2_method(),
      'sampler': random_normal_draw}

g2_local = {'name': 'G2++_local',
            'model': ql.G2,
            'engine': lambda model, _: ql.G2SwaptionEngine(model, 6.0, 16),
            'transformation': g2_transformation,
            'inverse_transformation': g2_inverse_transformation,
            'method': g2_method_local(),
            'sampler': random_normal_draw}


def local_hw_map(swo, date, pointA, pointB, off_x=0.1, off_y=0.1,
                 low_x=1e-8, low_y=1e-8, nb_points=20):
    assert (len(pointA) == 2 and len(pointB) == 2)
    if pointA[0] > pointB[0]:
        max_x = pointA[0]
        min_x = pointB[0]
    else:
        min_x = pointA[0]
        max_x = pointB[0]
    if pointA[1] > pointB[1]:
        max_y = pointA[1]
        min_y = pointB[1]
    else:
        min_y = pointA[1]
        max_y = pointB[1]
    off_x = (max_x - min_x) * off_x
    off_y = (max_y - min_y) * off_y
    max_x += off_x
    min_x -= off_x
    max_y += off_y
    min_y -= off_y
    if min_x <= low_x:
        min_x = low_x
    if min_y <= low_y:
        min_y = low_y

    assert (min_x <= max_x and min_y <= max_y)
    rx = np.linspace(min_x, max_x, nb_points)
    ry = np.linspace(min_y, max_y, nb_points)
    xx, yy = np.meshgrid(rx, ry)

    result = np.empty(xx.shape)
    result.fill(np.nan)
    swo.set_date(date)
    for i, x in enumerate(rx):
        for j, y in enumerate(ry):
            swo.model.setParams(ql.Array([x, y]))
            result[i, j] = swo.model.value(swo.model.params(), swo.helpers)

    return (xx, yy, result)


def to_tenor(index, irType):
    frequency = index.tenor().frequency()
    if (irType.lower() == 'ois'):
        return 'OIS'
    elif (irType.lower() == 'euribor'):
        return 'E%dM' % index.tenor().length()
    elif (irType.lower() == 'libor'):
        return 'L%dM' % index.tenor().length()


def format_vol(v, digits=2):
    fformat = '%%.%df %%%%' % digits
    return fformat % (v * 100)


def format_price(p, digits=2):
    fformat = '%%.%df' % digits
    return fformat % p


def proper_name(name):
    name = name.replace(" ", "_")
    name = name.replace("(", "")
    name = name.replace(")", "")
    name = name.replace(",", "_")
    name = name.replace("-", "_")
    name = name.replace("+", "p")
    return name


def flatten_name(name, node=h5_model_node, risk_factor='IR'):
    name = proper_name(name)
    return node + '/' + risk_factor + '/' + name


def postfix(size, with_error, history_start, history_end, history_part):
    if with_error:
        file_name = '_adj_err'
    else:
        file_name = '_unadj_err'
    file_name += '_s' + str(size)
    if history_start is not None:
        file_name += '_' + str(history_start) + '-' + str(history_end)
    else:
        file_name += '_' + str(history_part)

    return file_name


def sample_file_name(swo, size, with_error, history_start, history_end, history_part):
    file_name = flatten_name(swo.name).lower().replace('/', '_')
    file_name = du.data_dir + file_name

    if with_error:
        file_name += '_adj_err'
    else:
        file_name += '_unadj_err'
    file_name += '_s' + str(size)
    if history_start is not None:
        file_name += '_' + str(history_start) + '-' + str(history_end)
    else:
        file_name += '_' + str(history_part)

    return file_name
