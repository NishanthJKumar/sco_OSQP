import numpy as np


class OSQPVar(object):
    """
    A class representing a variable within OSQP that will be used to construct the
    QP that will be passed to the solver. The lb and ub are used to enforce a
    trust region on each variable
    """

    def __init__(self, var_name, lb=-np.inf, ub=np.inf, val=None):
        self.var_name = var_name
        self._lower_bound = lb
        self._upper_bound = ub
        self.val = val

    def __repr__(self):
        return f"OSQPVar with name {self.var_name}"
        # return f"OSQPVar with name {self.var_name}, lb={self._lower_bound}, ub={self._upper_bound}, val={self.val}"

    def get_lower_bound(self):
        return self._lower_bound

    def set_lower_bound(self, lb_val):
        assert isinstance(lb_val, float)
        assert not np.isnan(lb_val)
        self._lower_bound = lb_val

    def get_upper_bound(self):
        return self._upper_bound

    def set_upper_bound(self, ub_val):
        assert isinstance(ub_val, float)
        assert not np.isnan(ub_val)
        self._upper_bound = ub_val
