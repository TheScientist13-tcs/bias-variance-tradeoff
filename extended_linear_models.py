from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


class PolynomialRegression(LinearRegression):
    def __init__(
        self, degree, fit_intercept=True, copy_X=True, n_jobs=None, positive=False
    ):
        super().__init__(
            fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs, positive=positive
        )
        self.degree = degree
        self.coef_ = None
        self.intercept_ = None
        self._reg = None

    def fit(self, x, y, sample_weight=None):
        X = self._create_dummy_matrix(x, self.degree)
        reg = LinearRegression(
            fit_intercept=self.fit_intercept, positive=self.positive
        ).fit(X, y)
        self._reg = reg
        self.coef_ = reg.coef_
        self.intercept_ = reg.intercept_
        return self

    def score(self, x, y):
        X = self._create_dummy_matrix(x, self.degree)
        r_sq = self._reg.score(X, y)
        calc_adj_r_sq = lambda r2: 1 - (
            (1 - r2) * (len(x) - 1) / (len(x) - self.degree - 1)
        )
        adj_r_sq = calc_adj_r_sq(r_sq)
        return {"r_sq": r_sq, "adj_r_sq": adj_r_sq}

    def predict(self, x):
        X = self._create_dummy_matrix(x, self.degree)
        return self._reg.predict(X)

    def _create_dummy_matrix(self, x, degree):
        result_matrix = np.zeros([len(x), degree])
        for j in range(degree):
            if j == 0:
                result_matrix[:, j] = x
            else:
                result_matrix[:, j] = [x_i ** (j + 1) for x_i in x]
        return result_matrix


class RidgePolynomialRegression(Ridge):
    def __init__(
        self,
        degree,
        alpha_,
        fit_intercept=True,
        copy_X=True,
        positive=False,
        random_state=None,
    ):
        if degree < 1:
            raise Exception("Value Error: degree must be > 0")
        super().__init__(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            positive=positive,
            random_state=random_state,
        )
        self.alpha_ = alpha_
        self.degree = degree
        self.coef_ = None
        self.intercept_ = None
        self._reg = None

    def fit(self, x, y, sample_weight=None):
        X = self._create_dummy_matrix(x, self.degree)
        reg = Ridge(
            alpha=self.alpha_, fit_intercept=self.fit_intercept, positive=self.positive
        ).fit(X, y)
        self._reg = reg
        self.coef_ = reg.coef_
        self.intercept_ = reg.intercept_
        return self

    def score(self, x, y):
        X = self._create_dummy_matrix(x, self.degree)
        r_sq = self._reg.score(X, y)
        calc_adj_r_sq = lambda r2: 1 - (
            (1 - r2) * (len(x) - 1) / (len(x) - self.degree - 1)
        )
        adj_r_sq = calc_adj_r_sq(r_sq)
        return {"r_sq": r_sq, "adj_r_sq": adj_r_sq}

    def predict(self, x):
        X = self._create_dummy_matrix(x, self.degree)
        return self._reg.predict(X)

    def _create_dummy_matrix(self, x, degree):
        result_matrix = np.zeros([len(x), degree])
        for j in range(degree):
            if j == 0:
                result_matrix[:, j] = x
            else:
                result_matrix[:, j] = [x_i ** (j + 1) for x_i in x]
        return result_matrix


class LassoPolynomialRegression(Lasso):
    def __init__(
        self,
        degree,
        alpha_,
        fit_intercept=True,
        copy_X=True,
        positive=False,
        random_state=None,
    ):
        if degree < 1:
            raise Exception("Value Error: degree must be > 0")
        super().__init__(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            positive=positive,
            random_state=random_state,
        )
        self.alpha_ = alpha_
        self.degree = degree
        self.coef_ = None
        self.intercept_ = None
        self._reg = None

    def fit(self, x, y, sample_weight=None):
        X = self._create_dummy_matrix(x, self.degree)
        reg = Lasso(
            alpha=self.alpha_, fit_intercept=self.fit_intercept, positive=self.positive
        ).fit(X, y)
        self._reg = reg
        self.coef_ = reg.coef_
        self.intercept_ = reg.intercept_
        return self

    def score(self, x, y):
        X = self._create_dummy_matrix(x, self.degree)
        r_sq = self._reg.score(X, y)
        calc_adj_r_sq = lambda r2: 1 - (
            (1 - r2) * (len(x) - 1) / (len(x) - self.degree - 1)
        )
        adj_r_sq = calc_adj_r_sq(r_sq)
        return {"r_sq": r_sq, "adj_r_sq": adj_r_sq}

    def predict(self, x):
        X = self._create_dummy_matrix(x, self.degree)
        return self._reg.predict(X)

    def _create_dummy_matrix(self, x, degree):
        result_matrix = np.zeros([len(x), degree])
        for j in range(degree):
            if j == 0:
                result_matrix[:, j] = x
            else:
                result_matrix[:, j] = [x_i ** (j + 1) for x_i in x]
        return result_matrix


class CubicSplines(LinearRegression):
    def __init__(
        self, knot_locs, fit_intercept=True, copy_X=True, n_jobs=None, positive=False
    ):
        super().__init__(
            fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs, positive=positive
        )
        self.knot_locs = knot_locs
        self.coef_ = None
        self.intercept_ = None
        self._reg = None

    def fit(self, x, y, sample_weight=None):
        X = self._create_dummy_matrix(x, self.knot_locs)
        reg = LinearRegression(
            fit_intercept=self.fit_intercept, positive=self.positive
        ).fit(X, y)
        self._reg = reg
        self.coef_ = reg.coef_
        self.intercept_ = reg.intercept_
        return self

    def score(self, x, y):
        X = self._create_dummy_matrix(x, self.knot_locs)
        r_sq = self._reg.score(X, y)
        p = len(self.knot_locs) + 4
        calc_adj_r_sq = lambda r2: 1 - ((1 - r2) * (len(x) - 1) / (len(x) - p - 1))
        adj_r_sq = calc_adj_r_sq(r_sq)
        return {"r_sq": r_sq, "adj_r_sq": adj_r_sq}

    def predict(self, x):
        X = self._create_dummy_matrix(x, self.knot_locs)
        return self._reg.predict(X)

    def _create_dummy_matrix(self, X, knot_locs):
        nrows = X.shape[0]
        c1 = np.ones([nrows, 1])
        c2 = np.array(list(map(self._cubic_spline_bases(knot_locs)[1], X))).reshape(
            -1, 1
        )
        c3 = np.array(list(map(self._cubic_spline_bases(knot_locs)[2], X))).reshape(
            -1, 1
        )
        c4 = np.array(list(map(self._cubic_spline_bases(knot_locs)[3], X))).reshape(
            -1, 1
        )
        X_wrapped = np.concatenate((c1, c2, c3, c4), axis=1)
        for m in range(len(knot_locs)):
            c_m = np.array(
                list(map(self._cubic_spline_bases(knot_locs)[4 + m], X))
            ).reshape(-1, 1)
            X_wrapped = np.concatenate((X_wrapped, c_m), axis=1)
        return X_wrapped

    def _create_basis(self, xi):
        return lambda x: np.power(np.max([0, x - xi]), 3)

    def _cubic_spline_bases(self, knot_locs):
        h1 = lambda x: 1
        h2 = lambda x: x
        h3 = lambda x: np.power(x, 2)
        h4 = lambda x: np.power(x, 3)
        bases_funcs = [h1, h2, h3, h4]
        for i in range(len(knot_locs)):
            bases_funcs.append(self._create_basis(knot_locs[i]))
        return bases_funcs
