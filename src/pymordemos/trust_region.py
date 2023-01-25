#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


import numpy as np
from typer import run

from pymor.parameters.functionals import MinThetaParameterFunctional
from pymor.reductors.coercive import CoerciveRBReductor
from pymor.algorithms.tr import trust_region
from pymor.analyticalproblems.thermalblock import thermal_block_problem
from pymor.analyticalproblems.functions import ExpressionFunction
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.discretizers.builtin import discretize_stationary_cg


def main():
    fom, parameter_space, mu_bar = create_fom()
    
    initial_guess = parameter_space.sample_randomly(1)[0]
    print(initial_guess)

    coercivity_estimator = MinThetaParameterFunctional(fom.operator.coefficients, mu_bar)
    reductor = CoerciveRBReductor(fom, product=fom.energy_product, coercivity_estimator=coercivity_estimator)
    
    mu, data = trust_region(parameter_space, reductor, initial_guess=initial_guess)
    
    print(f'Optimized mu: {mu}', 'iterations:', data['iterations'], f'output: {fom.output(mu)}')
    NUM_SAMPLES = 5
    sols = []
    rsols = []
    mus = parameter_space.sample_randomly(NUM_SAMPLES)
    for mu in mus:
        sols.append(fom.output(mu))
        rsols.append(reductor._last_rom.solve(mu))
    for i in range(NUM_SAMPLES):
        print(f'Comparison output for {mus[i]}: {sols[i]} - {rsols[i]}')
    
    
def create_fom():
    helper_problem = thermal_block_problem()
    central = ExpressionFunction(
        '(x[0] <= .7) * (.2 <= x[0]) * (x[1] <= .7) * (.2 <= x[1]) * 1.', dim_domain=2)
    domain_of_interest = central
    theta_J = ExpressionParameterFunctional('1 + 1/5 * sum(diffusion) - .4 * diffusion[2]', helper_problem.parameters,
                                            derivative_expressions={'diffusion': ['1/5', '1/5', '1/5', '1/5', '1/5', '1/5', '1/5', '1/5', '1/5']})
    helper_problem = helper_problem.with_(outputs=[('quadratic', theta_J * domain_of_interest)])

    parameter_space = helper_problem.parameter_space
    mu_bar = {}
    mu_bar_helper = parameter_space.sample_uniformly(1)[0]
    for key in helper_problem.parameters:
        range_ = parameter_space.ranges[key]
        if range_[0] == 0:
            value = 10**(np.log10(range_[1]) / 2)
        else:
            value = 10**((np.log10(range_[0]) + np.log10(range_[1])) / 2)
        mu_bar[key] = [value for i in range(len(mu_bar_helper[key]))]
    mu_bar = helper_problem.parameters.parse(mu_bar)

    pde_opt_fom, _ = discretize_stationary_cg(helper_problem, mu_energy_product=mu_bar)

    return pde_opt_fom, parameter_space, mu_bar


if __name__ == '__main__':
    run(main)
