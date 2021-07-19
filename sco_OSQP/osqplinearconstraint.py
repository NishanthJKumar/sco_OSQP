import numpy as np


class OSQPLinearConstraint(object):
    def __init__(self, osqp_vars, coeffs, lb, ub):
        """
        A wrapper class for a linear constraint. The particular constraint represented is:

        lb <= np.dot(self.osqp_vars, self.coeffs) <= ub
        """
        assert osqp_vars.shape == coeffs.shape
        self.osqp_vars = osqp_vars
        self.coeffs = coeffs
        self.lb = lb
        self.ub = ub

    def __repr__(self):
        return f"OSQPLinearConstraint with osqp_vars={self._osqp_vars}, coeffs={self.coeffs}, lb = {self.lb}, ub = {self.ub}"

    def get_all_vars(self):
        return self.osqp_vars.tolist()
