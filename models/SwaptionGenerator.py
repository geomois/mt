import utils.data_utils as du
from models.IRCurve import IRCurve
from utils import dbDataPreprocess as dbc
from utils.customUtils import *
from utils.ahUtils import *
import pdb


class SwaptionGen(du.TimeSeriesData):
    '''
    SwaptionGen provides different functionality for working with QuantLib
    swaptions.
    Input parameters:
    index - IR index from which many conventions will be taken. A clone of the
            original index will be kept.
    modelF - Function to instantiate the model. The function should take only
             one parameter: a term structure handle
    engineF - Function to instantiate the swaption pricing engine. The function
              should take two parameters: a model and a term structure handle
        Methods:
    SwaptionGen['<date, format yyyy-mm-dd>'] returns the surface data for the day
    SwaptionGen.set_date(date) adjusts the swaption helpers to use the quotes
    from that date SwaptionGen.update_quotes(volas) adjusts the swaption helpers
    to use the given quotes
    SwaptionGen.calibrate_history(name) calibrates the model on each day and
    saves the results to the HDF5 file. The table has node Models/IR/<name>
    The index of the table is the dates
    The columns are:
        orig_evals - the number of evaluations needed to start from default params
        OptimEvals - the number of evaluations needed if starting from optimized solution
        orig_objective - objective value starting from default parameters
        orig_mean_error - mean error starting from default parameters
        HistEvals - the number of evaluations needed if starting from yesterday's params
        HistObjective - objective value starting from yesterday's params
        HistMeanError - mean error starting from yesterday's params
        orig_params - optimized parameters starting from default parameters
        HistParams - optimized parameters starting from yesterday's params
    '''

    def __init__(self, index, model_dict,
                 error_type=ql.CalibrationHelper.ImpliedVolError,
                 parentNode=du.h5_ts_node, irType='libor', volData=None, irData=None, **kwargs):
        self.customIndex = index
        self._model_dict = model_dict
        if not 'model' in self._model_dict \
                or not 'engine' in self._model_dict \
                or not 'name' in self._model_dict:
            raise RuntimeError('Missing parameters')
        self.ccy = index.currency().code()
        # pdb.set_trace()
        self.model_name = self._model_dict['name'].replace("/", "")
        self.name = 'SWO ' + self.ccy
        if 'file_name' in self._model_dict:
            self.name += ' ' + self._model_dict['file_name']
        else:
            self.name += ' ' + self.model_name

        self.key_ts = parentNode + '/SWO/' + self.ccy
        self.key_model = flatten_name('SWO/' + self.ccy + '/' + self.model_name)
        self.key_error = flatten_name('SWO/' + self.ccy + '/' + self.model_name, node=h5_error_node)
        super(SwaptionGen, self).__init__(self.key_ts, du.h5file, data=volData)

        # Yield Curve and dates
        self._ircurve = IRCurve(self.ccy, to_tenor(index, irType), parentNode=parentNode, data=irData)
        self._dates = self.intersection(self._ircurve)
        self.refdate = ql.Date(self._dates[0].day, self._dates[0].month, self._dates[0].year)
        ql.Settings.instance().evaluationDate = self.refdate
        self._term_structure = ql.RelinkableYieldTermStructureHandle()
        self._term_structure.linkTo(self._ircurve[self._dates[0]])

        # Index, swaption model and pricing engine
        self.index = index.clone(self._term_structure)
        self.model = self._model_dict['model'](self._term_structure)
        self.engine = self._model_dict['engine'](self.model, self._term_structure)
        # self.setupModel()

        # Quotes & calibration helpers
        volas = self.__getitem__(self._dates[0])
        volas.shape = (volas.shape[0] * volas.shape[1],)
        # pdb.set_trace()
        self._quotes = [ql.SimpleQuote(vola) for vola in volas]
        self.error_type = error_type
        self.__create_helpers()

        # Standard calibration nick-nacks
        if 'method' in self._model_dict:
            self.method = self._model_dict['method'][0]
            self.end_criteria = self._model_dict['method'][1]
            self.constraint = self._model_dict['method'][2]
        else:
            self.method = ql.LevenbergMarquardt()
            self.end_criteria = ql.EndCriteria(1000, 250, 1e-7, 1e-7, 1e-7)
            self.constraint = ql.PositiveConstraint()

        self._default_params = self.model.params()
        self._sampler = self._model_dict['sampler']

        # Output transformations
        if 'transformation' in self._model_dict:
            self._transformation = self._model_dict['transformation']
        else:
            self._transformation = None

        if 'inverse_transformation' in self._model_dict:
            self._inverse_transform = self._model_dict['inverse_transformation']
        else:
            self._inverse_transform = None

        # Misc
        self.ok_format = '%12s |%12s |%12s |%12s |%12s'
        self.err_format = '%2s |%12s |'
        self.values = None

    def setupModel(self, alpha=0, sigma=0):
        # Index, swaption model and pricing engine
        self.index = self.customIndex.clone(self._term_structure)
        self.model = self._model_dict['model'](self._term_structure, alpha, sigma)
        self.engine = self._model_dict['engine'](self.model, self._term_structure)

    def set_date(self, ddate):
        # pdb.set_trace()
        # Set Reference Date
        ts = pd.Timestamp(ddate)
        dt = ql.Date(ts.day, ts.month, ts.year)
        if dt != self.refdate or self.values is None:
            self.refdate = dt
            # Update term structure
            self._term_structure.linkTo(self._ircurve[ddate])

            # Update quotes
            volas = self.__getitem__(ddate)
            volas.shape = (volas.shape[0] * volas.shape[1],)
            self.update_quotes(volas)

    def update_quotes(self, volas):
        self.values = volas
        [quote.setValue(vola) for vola, quote in zip(volas, self._quotes)]

    @staticmethod
    def lengthInMonths(length):
        return (int(length % 1 * 100) + int(np.floor(length)) * 12)

    def __create_helpers(self):
        mg = np.meshgrid(self.axis(0).values, self.axis(1).values)
        mg[0].shape = (mg[0].shape[0] * mg[0].shape[1],)
        mg[1].shape = (mg[1].shape[0] * mg[1].shape[1],)
        self._maturities = mg[0]
        self._lengths = mg[1]
        # Create swaption helpers
        # pdb.set_trace()
        self.helpers = [ql.SwaptionHelper(ql.Period(int(maturity), ql.Years),
                                          ql.Period(self.lengthInMonths(length), ql.Months),
                                          ql.QuoteHandle(quote),
                                          self.index, self.index.tenor(),
                                          self.index.dayCounter(),
                                          self.index.dayCounter(),
                                          self._term_structure,
                                          self.error_type)
                        for maturity, length, quote in zip(mg[0], mg[1], self._quotes)]
        # Set pricing engine
        for swaption in self.helpers:
            swaption.setPricingEngine(self.engine)

    def __history(self, history_start, history_end, history_part, with_error):
        if history_start is not None and history_end is not None:
            assert (history_start >= 0)
            assert (history_start < len(self._dates))
            assert (history_end > history_start)
            assert (history_end <= len(self._dates))
            dates = self._dates[history_start:history_end]
        else:
            assert (history_part > 0)
            assert (history_part <= 1.0)
            dates = self._dates[:len(self._dates) * history_part]

        # Get history of parameters
        nb_swo_params = len(self._default_params)
        columns_orig = ['OrigParam%d' % x for x in range(nb_swo_params)]
        columns_hist = ['HistParam%d' % x for x in range(nb_swo_params)]
        df_model = pd.HDFStore(du.h5file)[self.key_model]
        # Pick the best of the two
        swo_param_history_orig = df_model.loc[dates][columns_orig]
        swo_param_history_hist = df_model.loc[dates][columns_hist]
        # pdb.set_trace()
        orig_vs_hist = df_model.loc[dates]['OrigObjective'].values < df_model.loc[dates]['HistObjective'].values
        swo_param_history = swo_param_history_orig.copy(deep=True)
        swo_param_history[~orig_vs_hist] = swo_param_history_hist[~orig_vs_hist]

        # Get history of errors
        if with_error:
            df_error = pd.HDFStore(du.h5file)[self.key_error]
            swo_error_history = df_error.loc[dates]
        else:
            swo_error_history = None
        return (dates, swo_param_history, swo_error_history)

    def train_history(self, *kwargs):
        # Retrieves training data from history
        if 'history_start' in kwargs:
            history_start = kwargs['history_start']
            history_end = kwargs['history_end']
            history_part = None
        else:
            history_start = None
            history_end = None
            if 'history_part' in kwargs:
                history_part = kwargs['history_part']
            else:
                history_part = 0.4

        if 'save' in kwargs and kwargs['save']:
            if 'file_name' in kwargs:
                file_name = kwargs['file_name']
            else:
                file_name = sample_file_name(self, 0, True, history_start,
                                             history_end, history_part)
            print('Saving to file %s' % file_name)

        (dates, y, swo_error) = \
            self.__history(history_start, history_end, history_part, True)
        x_ir = self._ircurve.to_matrix(dates)

        # Calculate volatilities according to different conditions
        nb_instruments = len(self.helpers)
        nb_dates = len(dates)
        x_swo = np.zeros((nb_dates, nb_instruments), np.float32)

        for row in range(nb_dates):
            # Set term structure
            self.set_date(dates[row])
            self.model.setParams(ql.Array(y[row, :].tolist()))
            for swaption in range(nb_instruments):
                try:
                    NPV = self.helpers[swaption].modelValue()
                    vola = self.helpers[swaption].impliedVolatility(NPV, 1.0e-6, 1000, 0.0001, 2.50)
                    x_swo[row, swaption] = np.clip(vola - swo_error[row, swaption], 0., np.inf)
                except RuntimeError as e:
                    print('Exception (%s) for (sample, maturity, length): (%s, %s, %s)' % (
                        e, row, self._maturities[swaption], self._lengths[swaption]))

        if 'save' in kwargs and kwargs['save']:
            if 'append' in kwargs and kwargs['append']:
                try:
                    x_swo_l = np.load(file_name + '_x_swo.npy')
                    x_ir_l = np.load(file_name + '_x_ir.npy')
                    y_l = np.load(file_name + '_y.npy')
                    x_swo = np.concatenate((x_swo_l, x_swo), axis=0)
                    x_ir = np.concatenate((x_ir_l, x_ir), axis=0)
                    y = np.concatenate((y_l, y), axis=0)
                except Exception as e:
                    print(e)

            np.save(file_name + '_x_swo', x_swo)
            np.save(file_name + '_x_ir', x_ir)
            np.save(file_name + '_y', y)
        return (x_swo, x_ir, y)

    def calibrate_history(self, start=0, end=-1, fileName=du.h5file, csvFilePath=None):
        clean = True
        if (end == -1):
            end = len(self._dates)

        prev_params = self._default_params
        nb_params = len(prev_params)
        nb_instruments = len(self.helpers)
        columns = ['OrigEvals', 'OptimEvals', 'OrigObjective', 'OrigMeanError',
                   'HistEvals', 'HistObjective', 'HistMeanError',
                   'HistObjectivePrior', 'HistMeanErrorPrior']
        size_cols = len(columns)
        columns = columns + ['OrigParam' + str(x) for x in range(nb_params)]
        columns = columns + ['HistParam' + str(x) for x in range(nb_params)]
        if clean:
            # pdb.set_trace()
            rows_model = np.empty((len(self._dates), len(columns)))
            rows_model.fill(np.nan)
            rows_error = np.zeros((len(self._dates), nb_instruments))
        else:
            # pdb.set_trace()
            rows_error = []
            with pd.HDFStore(fileName) as store:
                df_model = store[self.key_model]
                if len(df_model.columns) != len(columns) or \
                        len(df_model.index) != len(self._dates):
                    raise RuntimeError("Incompatible file")
                rows_model = df_model.values

                df_error = store[self.key_error]
                if len(df_error.columns) != nb_instruments or \
                        len(df_error.index) != len(self._dates):
                    raise RuntimeError("Incompatible file")
                rows_error = df_error.values

        header = self.ok_format % ('maturity', 'length', 'volatility', 'implied', 'error')
        dblrule = '=' * len(header)
        # pdb.set_trace()
        for iDate in range(start, end):
            self.model.setParams(self._default_params)
            # Return tuple (date, orig_evals, optim_evals, hist_evals,
            # orig_objective, orig_mean_error, hist_objective, hist_mean_error,
            # original params, hist parameters, errors)
            try:
                # pdb.set_trace()
                res = self.calibrate(self._dates[iDate], prev_params)
            except RuntimeError as e:
                print('')
                print(dblrule)
                print(self.name + " " + str(self._dates[iDate]))
                print("Error: %s" % e)
                print(dblrule)
                res = (self._dates[iDate], -1, -1, -1, -1, -1, -1, -1, -1, -1,
                       self._default_params, self._default_params, np.zeros((1, nb_instruments)))
            rows_model[iDate, 0:size_cols] = res[1:size_cols + 1]
            rows_model[iDate, size_cols:size_cols + nb_params] = res[size_cols + 1]
            rows_model[iDate, size_cols + nb_params:size_cols + nb_params * 2] = res[size_cols + 2]
            rows_error[iDate, :] = res[-1]
            prev_params = self.model.params()

        df_model = pd.DataFrame(rows_model, index=self._dates, columns=columns)
        df_error = pd.DataFrame(rows_error, index=self._dates)

        if (fileName is not None):
            with pd.HDFStore(fileName) as store:
                store[self.key_model] = df_model
                store[self.key_error] = df_error

        if (csvFilePath is not None):
            df_model[['OrigParam0', 'OrigParam1']].to_csv(csvFilePath)

        return df_model[['OrigParam0', 'OrigParam1']].copy()

    def calibrateOnce(self, ddate, alpha):
        self.set_date(ddate)
        try:
            pdb.set_trace()
            self.model.setParams(ql.Array([alpha, self._default_params[1]]))
            self.model.calibrate(self.helpers, self.method,
                                 self.end_criteria, self.constraint, [], [True, False])
            # calibrate only sigma

            params = self.model.params()
            orig_objective = self.model.value(params, self.helpers)
            orig_mean_error, errors = self.__errors()
        except RuntimeError as e:
            pdb.set_trace()
            print("Error: %s" % e)
            orig_objective = float("inf")
            orig_mean_error = float("inf")
            errors = np.zeros((1, len(self.helpers)))
        avgError, _ = self.__errors()

        return avgError

    def __errors(self, part=None):
        if (part is not None):
            # part = [0, len(self.helpers)]
            # TODO:map ir tenors to helpers' tenors
            pass
        total_error = 0.0
        with_exception = 0
        errors = np.zeros((1, len(self.helpers)))
        for swaption in range(len(self.helpers)):
            vol = self._quotes[swaption].value()
            NPV = self.helpers[swaption].modelValue()
            try:
                implied = self.helpers[swaption].impliedVolatility(NPV, 1.0e-4, 1000, 0.001, 1.80)
                errors[0, swaption] = vol - implied
                total_error += abs(errors[0, swaption])
            except RuntimeError:
                with_exception = with_exception + 1
        denom = len(self.helpers) - with_exception
        if denom == 0:
            average_error = float('inf')
        else:
            average_error = total_error / denom

        return average_error, errors

    def calibrate(self, ddate, *args):
        name = self.model_name
        header = self.ok_format % ('maturity', 'length', 'volatility', 'implied', 'error')
        rule = '-' * len(header)
        dblrule = '=' * len(header)

        print('')
        print(dblrule)
        print(name + " " + str(ddate))
        print(rule)

        self.set_date(ddate)
        try:
            self.model.calibrate(self.helpers, self.method,
                                 self.end_criteria, self.constraint)

            params = self.model.params()
            orig_objective = self.model.value(params, self.helpers)
            print('Parameters   : %s ' % self.model.params())
            print('Objective    : %s ' % orig_objective)
            # pdb.set_trace()
            orig_mean_error, errors = self.__errors()
            print('Average error: %s ' % format_vol(orig_mean_error, 4))
            print(dblrule)
        except RuntimeError as e:
            print("Error: %s" % e)
            print(dblrule)
            orig_objective = float("inf")
            orig_mean_error = float("inf")
            errors = np.zeros((1, len(self.helpers)))

        if 'functionEvaluation' in dir(self.model):
            with_evals = True
            orig_evals = self.model.functionEvaluation()
        else:
            with_evals = False
            orig_evals = -1
        orig_params = np.array([v for v in self.model.params()])
        if len(args) > 0:
            # Recalibrate using optimized parameters
            try:
                self.model.calibrate(self.helpers, self.method,
                                     self.end_criteria, self.constraint)
                optim_evals = self.model.functionEvaluation() if with_evals else -1
            except RuntimeError as e:
                optim_evals = -1

            # Recalibrate using previous day's parameters
            self.model.setParams(args[0])
            try:
                hist_objective_prior = self.model.value(self.model.params(), self.helpers)
                hist_mean_error_prior, _ = self.__errors()
                self.model.calibrate(self.helpers, self.method,
                                     self.end_criteria, self.constraint)
                hist_objective = self.model.value(self.model.params(), self.helpers)
                hist_mean_error, errors_hist = self.__errors()
                if hist_objective < orig_objective:
                    errors = errors_hist
            except RuntimeError:
                hist_objective_prior = float("inf")
                hist_mean_error_prior = float("inf")
                hist_objective = float("inf")
                hist_mean_error = float("inf")

            hist_evals = self.model.functionEvaluation() if with_evals else -1
            hist_params = np.array([v for v in self.model.params()])

            # Return tuple (date, orig_evals, optim_evals, hist_evals,
            # orig_objective, orig_mean_error, hist_objective, hist_mean_error,
            # hist_objective_prior, hist_mean_error_prior,
            # original params, hist parameters, errors)
            return (ddate, orig_evals, optim_evals, orig_objective, orig_mean_error,
                    hist_evals, hist_objective, hist_mean_error, hist_objective_prior,
                    hist_mean_error_prior, orig_params, hist_params, errors)

        return (ddate, orig_evals, orig_objective, orig_mean_error, orig_params, errors)

    def evaluate(self, params, irValues, ddate):
        self.refdate = ql.Date(ddate.day, ddate.month, ddate.year)
        _, curve = self._ircurve.build(self.refdate, irValues)
        self._term_structure.linkTo(curve)
        qlParams = ql.Array(params.tolist())
        self.model.setParams(qlParams)
        return self.__errors()

    def errors(self, predictive_model, ddate):
        with pd.HDFStore(du.h5file) as store:
            df_error = store[self.key_error]
            orig_errors = df_error.loc[ddate]
            store.close()

        self.refdate = ql.Date(1, 1, 1901)
        self.set_date(ddate)
        params = predictive_model.predict((self.values, self._ircurve.values))
        self.model.setParams(ql.Array(params.tolist()[0]))
        _, errors = self.__errors()
        return (orig_errors, errors)

    def history_heatmap(self, predictive_model, dates=None):
        self.refdate = ql.Date(1, 1, 1901)
        if dates is None:
            dates = self._dates

        errors_mat = np.zeros((len(dates), len(self.helpers)))
        for i, ddate in enumerate(dates):
            ddate = self._dates[i]
            self.set_date(ddate)
            params = predictive_model.predict((self.values, self._ircurve.values))
            self.model.setParams(ql.Array(params.tolist()[0]))
            _, errors_mat[i, :] = self.__errors()

        return errors_mat

    def objective_values(self, predictive_model, date_start, date_end):
        dates = self._dates[date_start:date_end]
        objective_predict = np.empty((len(dates),))
        volas_predict = np.empty((len(dates),))
        for i, ddate in enumerate(dates):
            self.set_date(ddate)
            params = predictive_model.predict((self.values, self._ircurve.values))
            self.model.setParams(ql.Array(params.tolist()[0]))
            volas_predict[i], _ = self.__errors()
            try:
                objective_predict[i] = self.model.value(self.model.params(), self.helpers)
            except RuntimeError:
                objective_predict[i] = np.nan

        return (objective_predict, volas_predict)

    def objective_shape(self, predictive_model, ddate, nb_samples_x=100, nb_samples_y=100):
        store = pd.HDFStore(du.h5file)
        df = store[self.key_model]
        store.close()
        self.set_date(ddate)

        params_predict = predictive_model.predict((self.values, self._ircurve.values))
        params_predict = params_predict.reshape((params_predict.shape[1],))
        self.model.setParams(ql.Array(params_predict.tolist()))
        print("Predict value = %f" % self.model.value(self.model.params(), self.helpers))
        orig_objective = df.ix[ddate, 'orig_objective']
        hist_objective = df.ix[ddate, 'HistObjective']
        if orig_objective < hist_objective:
            name = 'OrigParam'
        else:
            name = 'HistParam'

        params_calib = np.array([df.ix[ddate, name + '0'],
                                 df.ix[ddate, name + '1'],
                                 df.ix[ddate, name + '2'],
                                 df.ix[ddate, name + '3'],
                                 df.ix[ddate, name + '4']])
        self.model.setParams(ql.Array(params_calib.tolist()))
        print("Calib value = %f" % self.model.value(self.model.params(), self.helpers))

        params_optim = np.array(self._default_params)
        self.model.setParams(self._default_params)
        print("Optim value = %f" % self.model.value(self.model.params(), self.helpers))

        # The intention is to sample the plane that joins the three points:
        # params_predict, params_calib, and params_optim
        # A point on that plane can be described by Q(alpha, beta)
        # Q(alpha, beta) = params_predict+(alpha-beta(A*B)/(A*A))A+beta B
        # with
        A = params_calib - params_predict
        # and
        B = params_optim - params_predict

        Aa = np.sqrt(np.dot(A, A))
        Ba = np.dot(A, B) / Aa
        lim_alpha = np.array([np.min((Ba, 0)), np.max((Ba / Aa, 1))])
        da = lim_alpha[1] - lim_alpha[0]
        lim_alpha += np.array([-1.0, 1.0]) * da / 10
        lim_beta = np.array([-0.1, 1.1])

        ls_alpha = np.linspace(lim_alpha[0], lim_alpha[1], nb_samples_x)
        ls_beta = np.linspace(lim_beta[0], lim_beta[1], nb_samples_y)
        xv, xy = np.meshgrid(ls_alpha, ls_beta)
        sh = xv.shape
        samples = [params_predict + (alpha - beta * Ba / Aa) * A + beta * B
                   for alpha, beta in zip(xv.reshape((-1,)), xy.reshape((-1,)))]

        objectives = np.empty((len(samples),))
        for i, params in enumerate(samples):
            try:
                self.model.setParams(ql.Array(params.tolist()))
                objectives[i] = self.model.value(self.model.params(), self.helpers)
            except RuntimeError:
                objectives[i] = np.nan

        return (objectives.reshape(sh), lim_alpha, lim_beta)

    def calibrate_sigma(self, predictive_model, modelName, dates=None, dataLength=1, session=None, x_pl=None, part=None,
                        skip=0):
        store = pd.HDFStore(du.h5file)
        df = store[self.key_model]
        outcome = []
        store.close()
        part = [0, len(self.helpers)] if part == -1 else part
        self.refdate = ql.Date(1, 1, 1901)
        vals = np.zeros((len(df.index), 4))
        values = np.zeros((len(df.index), 13))
        if dates is None:
            dates = self._dates

        method = ql.LevenbergMarquardt()
        end_criteria = ql.EndCriteria(250, 200, 1e-7, 1e-7, 1e-7)
        lower = ql.Array(5, 1e-9)
        upper = ql.Array(5, 1.0)
        lower[4] = -1.0
        # constraint = ql.NonhomogeneousBoundaryConstraint(lower, upper)
        constraint = ql.PositiveConstraint()
        self.set_date(dates[0])
        dataDict = {'vol': np.empty((0, self.values.shape[0])), 'ir': np.empty((0, self._ircurve.values.shape[0]))}
        # pdb.set_trace()
        for i, ddate in enumerate(dates):
            if (i < skip):
                if (i + dataLength - 1 >= skip):
                    self.set_date(ddate)
                    dataDict['vol'] = np.vstack((dataDict['vol'], self.values))
                    dataDict['ir'] = np.vstack((dataDict['ir'], self._ircurve.values))
                continue
            if (i + 1 < dataLength):
                self.set_date(ddate)
                dataDict['vol'] = np.vstack((dataDict['vol'], self.values))
                dataDict['ir'] = np.vstack((dataDict['ir'], self._ircurve.values))
                continue
            self.set_date(ddate)
            if (dataLength == 1):
                params = predictive_model.predict(vol=dataDict['vol'], ir=dataDict['ir'])
            else:
                dataDict['vol'] = np.vstack((dataDict['vol'], self.values))
                dataDict['ir'] = np.vstack((dataDict['ir'], self._ircurve.values))
                params = predictive_model.predict(vol=dataDict['vol'], ir=dataDict['ir'])
                dataDict['vol'] = np.delete(dataDict['vol'], (0), axis=0)
                dataDict['ir'] = np.delete(dataDict['ir'], (0), axis=0)

            # self.setupModel(alpha=params[0])
            params = [[params[0, 0], 0]]  # shape (1,2)
            self.model.setParams(ql.Array(params[0]))
            self.model.calibrate(self.helpers, method, end_criteria, constraint, [], [True, False])  # keep alpha as is
            meanErrorAfter, _ = self.__errors(part=part)
            paramsC = self.model.params()
            paramsC = np.append(np.asarray(paramsC), meanErrorAfter)
            outcome.append(paramsC)
            # print('\n', i, paramsC, '\n')
            # try:
            #     objectiveAfter = self.model.value(self.model.params(), self.helpers)
            # except RuntimeError:
            #     objectiveAfter = np.nan
            print('\n', i, paramsC, '\n')
        return outcome

    def compare_history(self, predictive_model, modelName, dates=None, plot_results=True, dataLength=1, skip=0,
                        fullTest=False):
        store = pd.HDFStore(du.h5file)
        df = store[self.key_model]
        store.close()
        self.refdate = ql.Date(1, 1, 1901)
        vals = np.zeros((len(df.index), 4))
        values = np.zeros((len(df.index), 13))
        if dates is None:
            dates = self._dates

        method = ql.LevenbergMarquardt()
        end_criteria = ql.EndCriteria(250, 200, 1e-7, 1e-7, 1e-7)
        lower = ql.Array(5, 1e-9)
        upper = ql.Array(5, 1.0)
        lower[4] = -1.0
        # constraint = ql.NonhomogeneousBoundaryConstraint(lower, upper)
        constraint = ql.PositiveConstraint()
        self.set_date(dates[0])
        dataDict = {'vol': np.empty((0, self.values.shape[0])), 'ir': np.empty((0, self._ircurve.values.shape[0]))}
        # pdb.set_trace()
        paramsList = np.empty((0, 3))
        for i, ddate in enumerate(dates):
            if (i < skip):
                if (i + dataLength - 1 >= skip):
                    self.set_date(ddate)
                    dataDict['vol'] = np.vstack((dataDict['vol'], self.values))
                    dataDict['ir'] = np.vstack((dataDict['ir'], self._ircurve.values))
                continue
            if (i + 1 < dataLength):
                self.set_date(ddate)
                dataDict['vol'] = np.vstack((dataDict['vol'], self.values))
                dataDict['ir'] = np.vstack((dataDict['ir'], self._ircurve.values))
                continue

            self.set_date(ddate)
            if (dataLength == 1):
                params = predictive_model.predict(self.values, self._ircurve.values)
            else:
                dataDict['vol'] = np.vstack((dataDict['vol'], self.values))
                dataDict['ir'] = np.vstack((dataDict['ir'], self._ircurve.values))
                params = predictive_model.predict(dataDict['vol'], dataDict['ir'])
                print('\n', i, params, '\n')
                dataDict['vol'] = np.delete(dataDict['vol'], (0), axis=0)
                dataDict['ir'] = np.delete(dataDict['ir'], (0), axis=0)

            if (type(params) == list):
                self.model.setParams(ql.Array(params[0]))
            else:
                self.model.setParams(ql.Array(params.tolist()[0]))

            meanErrorPrior, _ = self.__errors()
            temp = np.asarray(params)
            temp = np.append(temp[0], meanErrorPrior).reshape((-1, 3))
            paramsList = np.vstack((paramsList, temp))
            try:
                objectivePrior = self.model.value(self.model.params(), self.helpers)
            except RuntimeError:
                objectivePrior = np.nan
            if (fullTest):
                self.model.calibrate(self.helpers, method, end_criteria, constraint)
                meanErrorAfter, _ = self.__errors()
                paramsC = self.model.params()
                try:
                    objectiveAfter = self.model.value(self.model.params(), self.helpers)
                except RuntimeError:
                    objectiveAfter = np.nan
            else:
                meanErrorAfter = np.nan
                paramsC = np.empty((5))
                paramsC[:] = np.nan
                objectiveAfter = np.nan
                meanErrorAfter = np.nan

            orig_mean_error = df.ix[ddate, 'OrigMeanError']
            hist_mean_error = df.ix[ddate, 'HistMeanError']
            orig_objective = df.ix[ddate, 'OrigObjective']
            hist_objective = df.ix[ddate, 'HistObjective']

            # pdb.set_trace()
            values[i, 0] = orig_mean_error
            values[i, 1] = hist_mean_error
            values[i, 2] = meanErrorPrior
            values[i, 3] = orig_objective
            values[i, 4] = hist_objective
            values[i, 5] = objectivePrior
            values[i, 6] = meanErrorAfter
            values[i, 7] = objectiveAfter
            extensiveFlag = False
            if orig_objective < hist_objective:
                values[i, 8] = df.ix[ddate, 'OrigParam0']
                values[i, 9] = df.ix[ddate, 'OrigParam1']
                if 'OrigParam2' in df.columns:
                    values[i, 10] = df.ix[ddate, 'OrigParam2']
                    values[i, 11] = df.ix[ddate, 'OrigParam3']
                    values[i, 12] = df.ix[ddate, 'OrigParam4']
                    extensiveFlag = True
            else:
                values[i, 8] = df.ix[ddate, 'HistParam0']
                values[i, 9] = df.ix[ddate, 'HistParam1']
                if 'HistParam2' in df.columns:
                    values[i, 10] = df.ix[ddate, 'HistParam2']
                    values[i, 11] = df.ix[ddate, 'HistParam3']
                    values[i, 12] = df.ix[ddate, 'HistParam4']
                    extensiveFlag = True

            print('Date=%s' % ddate)
            print('Vola: Orig=%s Hist=%s ModelPrior=%s ModelAfter=%s' % (
                orig_mean_error, hist_mean_error, meanErrorPrior, meanErrorAfter))
            print('NPV:  Orig=%s Hist=%s Model=%s ModelAfter=%s' % (
                orig_objective, hist_objective, objectivePrior, objectiveAfter))
            print('Param0: Cal:%s , Model:%s, Cal-Mod:%s' % (values[i, 8], params[0][0], paramsC[0]))
            print('Param1: Cal:%s , Model:%s, Cal-Mod:%s' % (values[i, 9], params[0][1], paramsC[1]))
            if extensiveFlag:
                print('Param2: Cal:%s , Model:%s, Cal-Mod:%s' % (values[i, 10], params[0][2], paramsC[2]))
                print('Param3: Cal:%s , Model:%s, Cal-Mod:%s' % (values[i, 11], params[0][3], paramsC[3]))
                print('Param4: Cal:%s , Model:%s, Cal-Mod:%s' % (values[i, 12], params[0][4], paramsC[4]))

            vals[i, 0] = (meanErrorPrior - orig_mean_error) / orig_mean_error * 100.0
            vals[i, 1] = (meanErrorPrior - hist_mean_error) / hist_mean_error * 100.0
            vals[i, 2] = (meanErrorAfter - orig_mean_error) / orig_mean_error * 100.0
            vals[i, 3] = (meanErrorAfter - hist_mean_error) / hist_mean_error * 100.0

            print('      impO=%s impH=%s impAfterO=%s impAfterH=%s' % (
                vals[i, 0], vals[i, 1], vals[i, 2], vals[i, 3]))
        if plot_results:
            r = range(vals.shape[0])
            fig = plt.figure(figsize=(16, 16))
            f1 = fig.add_subplot(211)
            f1.plot(r, vals[:, 0])
            f2 = fig.add_subplot(212)
            f2.plot(r, vals[:, 1])
            plt.savefig(modelName + '.png')
        return (dates, values, vals, paramsList)

def get_swaptiongen(modelMap=hullwhite_analytic, currency='GBP', irType='Libor', pNode=du.h5_ts_node,
                    volFileName=None, irFileName=None):
    # ,'GBP','EUR','USD','CNY'
    # ,'libor','euribor','shibor','ois'
    index = None
    if (str(currency).lower() == 'gbp' and str(irType).lower() == 'libor'):
        index = ql.GBPLibor(ql.Period(6, ql.Months))
    elif (str(currency).lower() == 'eur' and str(irType).lower() == 'euribor'):
        index = ql.Euribor(ql.Period(6, ql.Months))
        # pNode=du.h5_tsc_node #custom

    if (volFileName is None or irFileName is None):
        swo = SwaptionGen(index, modelMap, parentNode=pNode, irType=irType)
    else:
        swo = SwaptionGen(index, modelMap, irType=irType,
                          volData=dbc.toAHFileFormat(volFileName), irData=dbc.toAHFileFormat(irFileName))
    return swo


def default_calibrate_history(model_dict, *args):
    swo = get_swaptiongen(model_dict, 'gbp', 'libor')
    swo.calibrate_history(*args)
    return swo


def setDataFileName(file):
    du.h5file = file
