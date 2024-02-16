import numpy as np
import pathlib
import time

from ml_control.logger import getLogger
from ml_control.machine_learning_models.kernel_reduced_model import KernelReducedModel
from ml_control.problem_definitions.heat_equation import create_heat_equation_problem_complex
from ml_control.reduced_model import ReducedModel

from adaptive_ml_control.adaptive_model_hierarchy import AdaptiveModelHierarchy
from adaptive_ml_control.full_model import FullModel


T, nt, N, h, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT, \
    R_chol, M, parameter_space = create_heat_equation_problem_complex(200)

num_parameters = 100
training_parameters = np.array(np.meshgrid(np.linspace(*parameter_space[0], num_parameters),
                                           np.linspace(*parameter_space[1], num_parameters))).T.reshape(-1, 2)
np.random.shuffle(training_parameters)

fom = FullModel(T, nt, N, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT, R_chol, M,
                title="Complex heat equation problem")
rb_rom = ReducedModel(None, N, T, nt, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT,
                      R_chol, M, spatial_norm=lambda x: np.linalg.norm(h * x))
zero_padding = False
ml_rom = KernelReducedModel(rb_rom, [], T, nt, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT,
                            R_chol, M, spatial_norm=lambda x: np.linalg.norm(h * x), zero_padding=zero_padding)

tol = 1e-4
from vkoga.kernels import Gaussian
ml_training_parameters = {"kernel": Gaussian(0.5), "tol_p": 1e-10}
adaptive_model = AdaptiveModelHierarchy(fom, rb_rom, ml_rom, tol=tol, ml_training_parameters=ml_training_parameters)

logger = getLogger("HeatEquation", level="INFO")

logger.info(f"Solving for {len(training_parameters)} parameters ...")
for i, mu in enumerate(training_parameters):
    with logger.block(f"Parameter number {i} ..."):
        adaptive_model.solve(mu)

adaptive_model.print_summary()
timestr = time.strftime("%Y%m%d-%H%M%S")
filepath = f"results_heat_equation_{timestr}/"
pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)
adaptive_model.write_results_to_file(filepath)
plt = adaptive_model.plot_detailed_timings()
plt.show()
adaptive_model.write_detailed_timings_as_tex(filepath + "detailed_timings.tex")
plt.close()
plt = adaptive_model.plot_detailed_error_estimation()
plt.show()
adaptive_model.write_detailed_error_estimation_as_tex(filepath + "detailed_error_estimations.tex")
plt.close()
