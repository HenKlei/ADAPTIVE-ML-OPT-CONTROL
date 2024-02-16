import numpy as np

from ml_control.systems import get_control_from_final_time_adjoint, solve_optimal_control_problem


class FullModel:
    def __init__(self, T, nt, N, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT, R_chol, M,
                 cg_params={}, title=""):
        self.T = T
        self.nt = nt
        self.N = N
        self.parametrized_A = parametrized_A
        self.parametrized_B = parametrized_B
        self.parametrized_x0 = parametrized_x0
        self.parametrized_xT = parametrized_xT
        self.R_chol = R_chol
        self.M = M
        self.cg_params = cg_params
        self.title = title

    def solve(self, mu, return_adjoint=True):
        A = self.parametrized_A(mu)
        B = self.parametrized_B(mu)
        x0 = self.parametrized_x0(mu)
        xT = self.parametrized_xT(mu)
        phiT_init = np.zeros(self.N)
        phi_opt = solve_optimal_control_problem(x0, xT, self.T, self.nt, A, B, self.R_chol, self.M, phiT_init,
                                                **self.cg_params)
        u_opt = get_control_from_final_time_adjoint(phi_opt, self.T, self.nt, A, B, self.R_chol)
        if return_adjoint:
            return u_opt, phi_opt
        else:
            return u_opt

    def _summary(self, func, postfix=""):
        func(f"Title: {self.title}" + postfix)
        func(f"Number of time steps: {self.nt}" + postfix)
        func(f"State dimension: {self.N}" + postfix)
