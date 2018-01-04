from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
import numpy as np
import QuantLib as ql
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.covariance import empirical_covariance
from sklearn.pipeline import Pipeline
from scipy.linalg import sqrtm
import pandas as pd

seed = 1027
class FunctionTransformerWithInverse(BaseEstimator, TransformerMixin):
    def __init__(self, func=None, inv_func=None, validate=False,
                 accept_sparse=False, pass_y=False):
        self.validate = validate
        self.accept_sparse = accept_sparse
        self.pass_y = pass_y
        self.func = func
        self.inv_func = inv_func

    def fit(self, X, y=None):
        if self.validate:
            check_array(X, self.accept_sparse)
        return self

    def transform(self, X, y=None):
        if self.validate:
            X = check_array(X, self.accept_sparse)
        if self.func is None:
            return X
        return self.func(X)

    def inverse_transform(self, X, y=None):
        if self.validate:
            X = check_array(X, self.accept_sparse)
        if self.inv_func is None:
            return X
        return self.inv_func(X)


def retrieve_swo_train_set(file_name, transform=True, func=None, inv_func=None,
                           valid_size=0.2, test_size=0.2, total_size=1.0,
                           scaler=MinMaxScaler(), concatenated=True):
    # Randomly selects points from dataset
    # To make it reproducible
    np.random.seed(seed)

    x_swo = np.load(file_name + '_x_swo.npy')
    x_ir = np.load(file_name + '_x_ir.npy')
    y = np.load(file_name + '_y.npy')

    train_size = total_size - valid_size - test_size
    assert (train_size > 0. and train_size <= 1.)
    assert (valid_size >= 0. and valid_size <= 1.)
    assert (test_size >= 0. and test_size <= 1.)
    total_sample = y.shape[0]
    print("Total sample: %d" % total_sample)
    train_sample = int(np.ceil(total_sample * train_size))
    assert (train_sample > 0)
    valid_sample = int(np.floor(total_sample * valid_size))
    test_sample = int(np.floor(total_sample * test_size))
    total_sample -= train_sample
    if total_sample - valid_sample < 0:
        valid_sample = 0
        test_sample = 0
    else:
        total_sample -= valid_sample
        if total_sample - test_sample < 0:
            test_sample = 0

    index = np.arange(y.shape[0])
    np.random.shuffle(index)
    x_swo_train = x_swo[index[:train_sample]]
    x_ir_train = x_ir[index[:train_sample]]
    if concatenated:
        x_train = np.concatenate((x_swo_train, x_ir_train), axis=1)
    else:
        x_train = (x_swo_train, x_ir_train)
    y_train = y[index[:train_sample]]

    if valid_sample == 0:
        x_valid = None
        y_valid = None
    else:
        x_swo_valid = x_swo[index[train_sample:train_sample + valid_sample]]
        x_ir_valid = x_ir[index[train_sample:train_sample + valid_sample]]
        if concatenated:
            x_valid = np.concatenate((x_swo_valid, x_ir_valid), axis=1)
        else:
            x_valid = (x_swo_valid, x_ir_valid)
        y_valid = y[index[train_sample:train_sample + valid_sample]]

    if test_sample == 0:
        x_test = None
        y_test = None
    else:
        x_swo_test = x_swo[index[train_sample + valid_sample:train_sample + valid_sample + test_sample]]
        x_ir_test = x_ir[index[train_sample + valid_sample:train_sample + valid_sample + test_sample]]
        if concatenated:
            x_test = np.concatenate((x_swo_test, x_ir_test), axis=1)
        else:
            x_test = (x_swo_test, x_ir_test)
        y_test = y[index[train_sample + valid_sample:train_sample + valid_sample + test_sample]]

    if transform:
        if func is not None or inv_func is not None:
            funcTrm = FunctionTransformerWithInverse(func=func,
                                                     inv_func=inv_func)
            pipeline = Pipeline([('funcTrm', funcTrm), ('scaler', scaler)])
        else:
            pipeline = scaler

        y_train = pipeline.fit_transform(y_train)
        if y_valid is not None:
            y_valid = pipeline.transform(y_valid)
        if y_test is not None:
            y_test = pipeline.transform(y_test)
    else:
        print('No transform requested')
        pipeline = None

    return {'x_train': x_train,
            'y_train': y_train,
            'x_valid': x_valid,
            'y_valid': y_valid,
            'x_test': x_test,
            'y_test': y_test,
            'transform': pipeline}


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
    sqrt_cov = sqrtm(empirical_covariance(scaled)).real

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


# g2_vae = {'name' : 'G2++',
#       'model' : ql.G2,
#       'engine' : lambda model, _: ql.G2SwaptionEngine(model, 6.0, 16),
#       'transformation' : g2_transformation,
#       'inverse_transformation' : g2_inverse_transformation,
#       'method': g2_method(),
#       'sampler': vae.sample_from_generator,
#       'file_name': 'g2pp_vae'}

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
