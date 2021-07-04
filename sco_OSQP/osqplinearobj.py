class OSQPLinearObj(object):
    def __init__(self, osqp_var, coeff):
        """
        A wrapper class for a linear objective. The particular constraint represented is:

        coeff * osqp_var
        """
        self.osqp_var = osqp_var
        self.coeff = coeff

    def get_all_vars(self):
        return [self.osqp_var]