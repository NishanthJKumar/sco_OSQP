import time
from collections import defaultdict

import numpy as np
import osqp
import scipy

from sco_OSQP.osqplinearconstraint import OSQPLinearConstraint
from sco_OSQP.osqplinearobj import OSQPLinearObj
from sco_OSQP.osqpquadraticobj import OSQPQuadraticObj

from .expr import *


class Prob(object):
    """
    Sequential convex programming problem with a scalar objective. A solution is
    found using the l1 penalty method.
    """

    def __init__(self, callback=None):
        """
        _vars: variables in this problem
        _osqp_vars: a set of all the osqp_vars in this problem. This will be used to
            construct the x vector whenever a call is made to the OSQP solver.

        _quad_obj_exprs: list of quadratic expressions in the objective
        _nonquad_obj_exprs: list of non-quadratic expressions in the objective
        _approx_obj_exprs: list of approximation of the non-quadratic
            expressions in the objective

        _nonlin_cnt_exprs: list of non-linear constraint expressions
        _penalty_exprs: list of penalty term expressions (approximations of the
            non-linear constraint expressions in _nonlin_cnt_exprs)

        _osqp_quad_objs: list of OSQPQuadraticObjs that keep track of the quadratic objectives
            that will be passed to the QP
        _osqp_lin_objs: list of OSQPLinearObjectives that keep track of the linear objectives
            that will be passed to the QP
        _osqp_lin_cnt_exprs: list of OSQPLinearConstraints that keep track of the linear constraints
            that will be passed to the QP

        _osqp_penalty_cnts: list of Gurobi constraints that are generated when
            adding the hinge and absolute value terms from the penalty terms.
        _pgm: Positive Gurobi variable manager provides a lazy way of generating
            positive Gurobi variables so that there are less model updates.

        _bexpr_to_osqp_expr: dictionary that caches quadratic bound expressions
            with their corresponding Gurobi expression
        """
        self._vars = set()
        self._osqp_vars = set()
        if callback is not None:
            self._callback = callback
        else:

            def do_nothing():
                pass

            self._callback = do_nothing

        self._quad_obj_exprs = []
        self._nonquad_obj_exprs = []
        self._approx_obj_exprs = []

        self._nonlin_cnt_exprs = []

        # These are lists of OSQPQuadraticObj's, OSQPLinearObj's and OSQPLinearConstraints
        # that will directly be used to construct the P, q and A matrices that define
        # the final QP to be solved.
        self._osqp_quad_objs = []
        self._osqp_lin_objs = []
        self._osqp_lin_cnt_exprs = []

        # list of constraints that will hold the hinge constraints
        # for each non-linear constraint, is a pair of constraints
        # for an eq constraint
        self.hinge_created = False

        self._penalty_exprs = []
        # self._osqp_penalty_cnts = []  # hinge and abs value constraints

        ## group-id (str) -> cnt-set (set of constraints)
        self._cnt_groups = defaultdict(set)
        self._cnt_groups_overlap = defaultdict(set)
        self._penalty_groups = []
        self.nonconverged_groups = []
        self.gid2ind = {}

    def add_obj_expr(self, bound_expr):
        """
        Adds a bound expression (bound_expr) to the objective. If the objective
        is quadratic, is it added to _quad_obj_exprs. Otherwise, it is added
        to self._nonquad_obj_exprs.

        bound_expr's var is added to self._vars so that a trust region can be
        added to var.
        """
        expr = bound_expr.expr
        if isinstance(expr, AffExpr) or isinstance(expr, QuadExpr):
            self._quad_obj_exprs.append(bound_expr)
        else:
            self._nonquad_obj_exprs.append(bound_expr)
        self.add_var(bound_expr.var)

    def add_var(self, var):
        self._vars.add(var)

    def add_osqp_var(self, osqp_var):
        self._osqp_vars.add(osqp_var)

    def add_cnt_expr(self, bound_expr, group_ids=None):
        """
        Adds a bound expression (bound_expr) to the problem's constraints.
        If the constraint is linear, it is added directly to the model.
        Otherwise, the constraint is added by appending bound_expr to
        self._nonlin_cnt_exprs.

        bound_expr's var is added to self._vars so that a trust region can be
        added to var.
        """
        comp_expr = bound_expr.expr
        expr = comp_expr.expr
        var = bound_expr.var
        assert isinstance(comp_expr, CompExpr)
        if isinstance(expr, AffExpr):
            if isinstance(comp_expr, EqExpr):
                self._add_osqp_cnt_from_aff_expr(expr, var, "eq", comp_expr.val)
            elif isinstance(comp_expr, LEqExpr):
                self._add_osqp_cnt_from_aff_expr(expr, var, "leq", comp_expr.val)
        else:
            self._nonlin_cnt_exprs.append(bound_expr)
            self._reset_hinge_cnts()

            if group_ids is None:
                group_ids = ["all"]
            for gid in group_ids:
                self._cnt_groups[gid].add(bound_expr)
                for other in group_ids:
                    if other == gid:
                        continue
                    self._cnt_groups_overlap[gid].add(other)

        self.add_var(var)

    def optimize(self):
        """
        Calls the OSQP optimizer on the current QP approximation with a given
        penalty coefficient.
        """
        # First, we need to setup the problem as described here: https://osqp.org/docs/solver/index.html
        # Specifically, we need to start by constructing the x vector that contains all the
        # OSQPVars that are part of the QP. This will take the form of a mapping from OSQPVar to
        # index within the x vector.
        var_to_index_dict = {}
        idx = 0
        for osqp_var in self._osqp_vars:
            var_to_index_dict[osqp_var] = idx
            idx += 1
        num_osqp_vars = len(self._osqp_vars)

        # Construct the q-vector by looping through all the linear objectives
        q_vec = np.zeros(idx)
        for lin_obj in self._osqp_lin_objs:
            q_vec[var_to_index_dict[lin_obj.osqp_var]] = lin_obj.coeff

        # Next, construct the P-matrix by looping through all quadratic objectives

        # Since P must be upper-triangular, the shape must be (num_osqp_vars, num_osqp_vars)
        P_mat = np.zeros((num_osqp_vars, num_osqp_vars))
        for quad_obj in self._osqp_quad_objs:
            for i in range(quad_obj.coeffs.shape[0]):
                var1_index = var_to_index_dict[quad_obj.osqp_vars1[i]]
                var2_index = var_to_index_dict[quad_obj.osqp_vars2[i]]
                # To ensure P_mat is upper-triangular, we need to sort these indices in
                # ascending order.
                idx1, idx2 = sorted((var1_index, var2_index))
                P_mat[idx1, idx2] += quad_obj.coeffs[i]

        # Next, setup the A-matrix and l and u vectors
        num_var_constraints = sum(
            osqp_vars.shape[0]
            for var in self._vars
            for osqp_vars in var.get_osqp_vars()
        )
        A_mat = np.zeros(
            (num_var_constraints + len(self._osqp_lin_cnt_exprs), num_osqp_vars)
        )
        l_vec = np.zeros(num_var_constraints + len(self._osqp_lin_cnt_exprs))
        u_vec = np.zeros(num_var_constraints + len(self._osqp_lin_cnt_exprs))
        # First add all the linear constraints
        row_num = 0
        for lin_constraint in self._osqp_lin_cnt_exprs:
            l_vec[row_num] = lin_constraint.lb
            u_vec[row_num] = lin_constraint.ub
            for i in range(lin_constraint.coeffs.shape[0]):
                A_mat[
                    row_num, var_to_index_dict[lin_constraint.osqp_vars[i]]
                ] = lin_constraint.coeffs[i]
            row_num += 1

        # Then, add the trust regions for every variable as constraints
        # for var in self._vars:
        for var in self._vars:
            osqp_vars = var.get_osqp_vars()
            for osqp_var_i in range(osqp_vars.shape[0]):
                A_mat[row_num, var_to_index_dict[osqp_vars[osqp_var_i, 0]]] = 1.0
                l_vec[row_num] = osqp_vars[osqp_var_i, 0].get_lower_bound()
                u_vec[row_num] = osqp_vars[osqp_var_i, 0].get_upper_bound()
                row_num += 1

        # Finally, construct the matrices and call the OSQP Solver!
        P_mat = scipy.sparse.csc_matrix(P_mat)
        A_mat = scipy.sparse.csc_matrix(A_mat)
        m = osqp.OSQP()
        m.setup(P=P_mat, q=q_vec, A=A_mat, l=l_vec, u=u_vec)
        solve_res = m.solve()

        # If the solve succeeded, update all the variables with these new values, then
        # run he callback before returning true
        self._update_osqp_vars(var_to_index_dict, solve_res.x)
        self._update_vars()
        self._callback()

    def _reset_hinge_cnts(self):
        ## reset the hinge_cnts
        self.hinge_created = False

    def _add_osqp_objs_and_cnts_from_expr(self, bound_expr):
        """
        Uses AffExpr, QuadExpr, HingeExpr and AbsExpr to extract
        OSQP solver compatible data structures. Depending on the expression type,
        appends elements to self._osqp_quad_objs, or self._osqp_lin_objs.
        """
        expr = bound_expr.expr
        var = bound_expr.var

        if isinstance(expr, AffExpr):
            self._add_to_lin_objs_and_cnts_from_aff_expr(expr, var)
        elif isinstance(expr, QuadExpr):
            self._add_to_quad_and_lin_objs_from_quad_expr(expr, var)
        elif isinstance(expr, HingeExpr):
            self._add_to_lin_objs_and_cnts_from_hinge_expr(expr, var)
        elif isinstance(expr, AbsExpr):
            self._add_to_lin_objs_and_cnts_from_abs_expr(expr, var)
        elif isinstance(expr, CompExpr):
            raise Exception(
                "Comparison Expressions cannot be converted to \
                OSQP problem objectives; use _add_osqp_cnt_from_aff_expr \
                instead"
            )
        else:
            raise Exception(
                "This type of Expression cannot be converted to\
                an OSQP objective."
            )

    def _add_to_lin_objs_and_cnts_from_aff_expr(self, expr, var):
        raise NotImplementedError

    # def _hinge_expr_to_grb_expr(self, hinge_expr, var):
    #     aff_expr = hinge_expr.expr
    #     assert isinstance(aff_expr, AffExpr)
    #     grb_expr, _ = self._aff_expr_to_grb_expr(aff_expr, var)
    #     grb_hinge = self._pgm.get_array(grb_expr.shape)
    #     cnts = self._add_np_array_grb_cnt(grb_expr, GRB.LESS_EQUAL, grb_hinge)
    #     return grb_hinge, cnts

    def _add_to_lin_objs_and_cnts_from_hinge_expr(self, expr, var):
        raise NotImplementedError

    # def _abs_expr_to_grb_expr(self, abs_expr, var):
    #     aff_expr = abs_expr.expr
    #     assert isinstance(aff_expr, AffExpr)
    #     grb_expr, _ = self._aff_expr_to_grb_expr(aff_expr, var)
    #     pos = self._pgm.get_array(grb_expr.shape)
    #     neg = self._pgm.get_array(grb_expr.shape)
    #     cnts = self._add_np_array_grb_cnt(grb_expr, GRB.EQUAL, pos - neg)
    #     return pos + neg, cnts

    def _add_to_lin_objs_and_cnts_from_abs_expr(self, expr, var):
        raise NotImplementedError

    def _add_osqp_cnt_from_aff_expr(self, aff_expr, var, cnt_type, cnt_val):
        """
        Uses aff_expr to create OSQPLinearConstraints of cnt_type that are then
        appended to self._osqp_lin_cnt_exprs
        """
        osqp_vars = var.get_osqp_vars()
        A_mat = aff_expr.A
        b_vec = aff_expr.b
        for i in range(A_mat.shape[0]):
            (inds,) = np.nonzero(A_mat[i, :])
            # If the constraint to be added is an equality constraint,
            # compute the upper and lower bounds
            if cnt_type is "eq":
                # the upper and lower bounds must be equal, and they must be
                # whatever the cnt_val was minus the constant term
                curr_lb = cnt_val[i] - b_vec[i]
                curr_ub = cnt_val[i] - b_vec[i]
            elif cnt_type is "leq":
                # only the upper bound needs to be set; the lower bound is negative
                # infinity
                curr_lb = -np.inf
                curr_ub = cnt_val[i] - b_vec[i]
            else:
                raise NotImplementedError

            curr_cnt_expr = OSQPLinearConstraint(
                osqp_vars[inds, 0], A_mat[i, inds], curr_lb, curr_ub
            )
            self._osqp_lin_cnt_exprs.append(curr_cnt_expr)

    def _add_to_quad_and_lin_objs_from_quad_expr(self, quad_expr, var):
        x = var.get_osqp_vars()
        Q = quad_expr.Q
        rows, cols = x.shape
        assert cols == 1
        inds = np.nonzero(Q)
        coeffs = 2 * Q[inds]  # Need to multiply by 2 because OSQP expects 0.5*x.T*Q*x
        v1 = x[inds[0], 0]
        v2 = x[inds[1], 0]
        # Create the new QuadraticObj term and append it to the problem's running list of
        # such terms
        self._osqp_quad_objs.append(OSQPQuadraticObj(v1, v2, coeffs))
        inds = np.nonzero(quad_expr.A)
        lin_coeffs = quad_expr.A[inds]
        # Because quad_expr.A is of shape (1,2), inds[1] corresponds to the nonzero
        # vars
        lin_vars = x[inds[1], 0]
        assert lin_coeffs.shape == lin_vars.shape
        for lin_var, lin_coeff in zip(lin_vars.tolist(), lin_coeffs.tolist()):
            self._osqp_lin_objs.append(OSQPLinearObj(lin_var, lin_coeff))

    def find_closest_feasible_point(self):
        """
        Finds the closest point (l2 norm) to the initialization that satisfies
        the linear constraints.
        """
        # Store a mapping of variables to coefficients within the q vector of the
        # OSQP problem to be formed
        var_to_q_arr_val_dict = {}
        q_arr = np.array([])

        # Loop thru all variables available. For each variable, get the values that
        # are not nan. The linear objective coefficients are simply -2 * val, for each val in these
        # values, and the quadratic objective coefficient is 1.
        # Use this fact to update var_to_index_dict and q_arr
        for var in self._vars:
            osqp_vars = var.get_osqp_vars()
            val = var.get_value()
            if val is not None:
                assert osqp_vars.shape == val.shape
                inds = np.where(~np.isnan(val))
                val = val[inds]
                nonnan_osqp_vars = osqp_vars[inds]
                val_arr = val.flatten()
                for i, nonnan_osqp_var in enumerate(
                    nonnan_osqp_vars.flatten().tolist()
                ):
                    # We may see the same variable name multiple times!
                    # We need to account for this possibility with dict.get()
                    if var_to_q_arr_val_dict.get(nonnan_osqp_var) is not None:
                        var_to_q_arr_val_dict[nonnan_osqp_var] += -2 * val_arr[i]
                    else:
                        var_to_q_arr_val_dict[nonnan_osqp_var] = -2 * val_arr[i]

        # Now that we've constructed a mapping from each variable to its linear objective value
        # (q_arr_val), we can construct the final P matrix and q vector needed to define the QP
        num_vars_in_prob = len(self._osqp_vars)
        P_mat = np.zeros((num_vars_in_prob, num_vars_in_prob))
        q_arr = np.zeros(num_vars_in_prob)
        var_to_osqp_indices_dict = {}
        for i, var in enumerate(self._osqp_vars):
            var_to_osqp_indices_dict[var] = i

        for i, var in enumerate(var_to_q_arr_val_dict.keys()):
            # Set the index corresponding to the variable to 2 in P_mat to offset the
            # 1/2 constant
            var_index = var_to_osqp_indices_dict[var]
            P_mat[var_index, var_index] = 2.0
            q_arr[var_index] = var_to_q_arr_val_dict[var]

        # Solve the QP using OSQP
        P_mat = scipy.sparse.csc_matrix(P_mat)
        m = osqp.OSQP()
        m.setup(P=P_mat, q=q_arr)

        solve_res = m.solve()

        # If the solve failed, just return False
        if solve_res.info.status_val != 1:
            return False

        # If the solve succeeded, update all the variables with these new values, then
        # run he callback before returning true
        self._update_osqp_vars(var_to_osqp_indices_dict, solve_res.x)
        self._update_vars()
        self._callback()
        return True

    def update_obj(self, penalty_coeff=0.0):
        self._reset_osqp_objs()
        self._lazy_spawn_osqp_cnts()
        for bound_expr in self._quad_obj_exprs + self._approx_obj_exprs:
            self._add_osqp_objs_and_cnts_from_expr(bound_expr)

        for i, bound_expr in enumerate(self._penalty_exprs):
            # TODO: Get these next two lines to run when necessary
            grb_expr = self._update_nonlin_cnt(bound_expr, i).flatten()
            grb_exprs.extend(grb_expr * penalty_coeff)

    def _reset_osqp_objs(self):
        """Resets the quadratic and linear objectives in preparation for the
        definition of a new OSQP problem"""
        self._osqp_quad_objs = []
        self._osqp_lin_objs = []

    def _lazy_spawn_osqp_cnts(self):
        if not self.hinge_created:
            self._osqp_penalty_cnts = []
            self._osqp_penalty_exprs = []
            self._osqp_nz = []
            for bound_expr in self._penalty_exprs:
                self._osqp_nz.append(np.nonzero(bound_expr.expr.expr.A))
                self._add_osqp_objs_and_cnts_from_expr(bound_expr)
                self._osqp_penalty_cnts.append(grb_cnts)
                self._osqp_penalty_exprs.append(grb_expr)
            self.hinge_created = True

    # @profile
    def _update_nonlin_cnt(self, bexpr, ind):
        # TODO: Port this to OSQP
        expr, var = bexpr.expr, bexpr.var
        if isinstance(expr, HingeExpr) or isinstance(expr, AbsExpr):
            aff_expr = expr.expr
            assert isinstance(aff_expr, AffExpr)
            A, b = aff_expr.A, aff_expr.b
            cnts = self._osqp_penalty_cnts[ind]
            grb_expr = self._osqp_penalty_exprs[ind]
            old_nz = self._osqp_nz[ind]
            grb_vars = var.get_grb_vars()
            nz = np.nonzero(A)

            for i in range(A.shape[0]):
                ## add negative b to rhs because it
                ## changes sides of the ineq/eq
                cnts[i].setAttr("rhs", -b[i, 0])

            for idx in range(old_nz[0].shape[0]):
                i, j = old_nz[0][idx], old_nz[1][idx]
                self._model.chgCoeff(cnts[i], grb_vars[j, 0], 0)

            ## then set the non-zero values
            for idx in range(nz[0].shape[0]):
                i, j = nz[0][idx], nz[1][idx]
                self._model.chgCoeff(cnts[i], grb_vars[j, 0], A[i, j])

            self._osqp_nz[ind] = nz
            return grb_expr
        else:
            raise NotImplementedError

    def add_trust_region(self, trust_region_size):
        """
        Adds the trust region for every variable
        """
        for var in self._vars:
            var.add_trust_region(trust_region_size)

    # @profile
    def convexify(self):
        """
        Convexifies the optimization problem by computing a QP approximation
        A quadratic approximation of the non-quadratic objective terms
        (self._nonquad_obj_exprs) is saved in self._approx_obj_exprs.
        The penalty approximation of the non-linear constraints
        (self._nonlin_cnt_exprs) is saved in self._penalty_exprs
        """
        self._approx_obj_exprs = [
            bexpr.convexify(degree=2) for bexpr in self._nonquad_obj_exprs
        ]
        self._penalty_exprs = [
            bexpr.convexify(degree=1) for bexpr in self._nonlin_cnt_exprs
        ]
        self._penalty_groups = []
        gids = sorted(self._cnt_groups.keys())
        self.gid2ind = {}
        for i, gid in enumerate(gids):
            self.gid2ind[gid] = i
            cur_bexprs = [bexpr.convexify(degree=1) for bexpr in self._cnt_groups[gid]]
            self._penalty_groups.append(cur_bexprs)

    # #@profile
    def get_value(self, penalty_coeff, vectorize=False):
        """
        Returns the current value of the penalty objective.
        The penalty objective is computed by summing up all the values of the
        quadratic objective expressions (self._quad_obj_exprs), the
        non-quadratic objective expressions and the penalty coeff multiplied
        by the constraint violations (computed using _nonlin_cnt_exprs)

        if vectorize=True, then this returns a vector of constraint
        violations -- 1 per group id.
        """
        if vectorize:
            gids = sorted(self._cnt_groups.keys())
            value = np.zeros(len(gids))
            for i, gid in enumerate(gids):
                value[i] = np.sum(
                    np.sum(
                        [
                            np.sum(self._compute_cnt_violation(bexpr))
                            for bexpr in self._cnt_groups[gid]
                        ]
                    )
                )
            return value
        value = 0.0
        for bound_expr in self._quad_obj_exprs + self._nonquad_obj_exprs:
            value += np.sum(np.sum(bound_expr.eval()))
        for bound_expr in self._nonlin_cnt_exprs:
            cnt_vio = self._compute_cnt_violation(bound_expr)
            value += penalty_coeff * np.sum(cnt_vio)
        return value

    # @profile
    def _compute_cnt_violation(self, bexpr):
        comp_expr = bexpr.expr
        var_val = bexpr.var.get_value()
        if isinstance(comp_expr, EqExpr):
            return np.absolute(comp_expr.expr.eval(var_val) - comp_expr.val)
        elif isinstance(comp_expr, LEqExpr):
            v = comp_expr.expr.eval(var_val) - comp_expr.val
            zeros = np.zeros(v.shape)
            return np.maximum(v, zeros)

    def get_max_cnt_violation(self):
        """
        Returns the the maximum amount a non-linear constraint is violated.
        Linear constraints are assumed to be satisfied because they are added
        directly to the model and QP solvers can deal with them.
        """
        max_vio = 0.0
        for bound_expr in self._nonlin_cnt_exprs:
            cnt_vio = self._compute_cnt_violation(bound_expr)
            cnt_max_vio = np.amax(cnt_vio)
            max_vio = np.maximum(max_vio, cnt_max_vio)
        return max_vio

    def get_approx_value(self, penalty_coeff, vectorize=False):
        """
        Returns the current value of the penalty QP approximation by summing
        up the expression values for the quadratic objective terms
        (_quad_obj_exprs), the quadratic approximation of the non-quadratic
        terms (_approx_obj_exprs) and the penalty terms (_penalty_exprs).
        Note that this approximate value is computed with respect to when the
        last convexification was performed.

        if vectorize=True, then this returns a vector of constraint
        violations -- 1 per group id.
        """
        if vectorize:
            value = np.zeros(len(self._penalty_groups))
            for i, bexprs in enumerate(self._penalty_groups):
                x = np.array([np.sum(bexpr.eval()) for bexpr in bexprs])
                value[i] = np.sum(x.flatten())
            return value

        value = 0.0
        for bound_expr in self._quad_obj_exprs + self._approx_obj_exprs:
            value += np.sum(np.sum(bound_expr.eval()))
        for bound_expr in self._penalty_exprs:
            value += penalty_coeff * np.sum(bound_expr.eval())

        return value

    def _update_osqp_vars(self, var_to_osqp_indices_dict, solver_values):
        """
        Updates the variables values based on the OSQP solution
        """
        for osqp_var in var_to_osqp_indices_dict.keys():
            osqp_var.val = solver_values[var_to_osqp_indices_dict[osqp_var]]

    def _update_vars(self):
        for var in self._vars:
            var.update()

    def batch_add_lin_cnts(self, list_of_lin_cnts):
        self._osqp_lin_cnt_exprs.extend(list_of_lin_cnts)

    def save(self):
        """
        Saves the problem's current state by saving the values of all the
        variables.
        """
        for var in self._vars:
            var.save()

    def restore(self):
        """
        Restores the problem's state to the problem's saved state
        """
        for var in self._vars:
            var.restore()
