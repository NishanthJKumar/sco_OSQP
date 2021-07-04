import numpy as np


class OSQPQuadraticObj(object):
    def __init__(self, osqp_vars1, osqp_vars2, coeffs):
        """
        A wrapper class for a quadratic objective. The particular objective represented is:

        0.5 * np.linalg.multi_dot(self.osqp_vars1, self.osqp_vars2, self.coeffs)
        """
        assert osqp_vars1.shape == osqp_vars2.shape == coeffs.shape
        # Together with the above condition, this implicitly implies all these arrays must
        # have shape of length 1
        assert len(osqp_vars1.shape) == 1
        self.osqp_vars1 = osqp_vars1
        self.osqp_vars2 = osqp_vars2
        self.coeffs = coeffs

    def get_all_vars(self):
        return self.osqp_vars1.tolist() + self.osqp_vars2.tolist()
