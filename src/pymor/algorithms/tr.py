# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import warnings
import numpy as np
from copy import deepcopy

from pymor.parameters.base import Mu
from pymor.algorithms.bfgs import bfgs
from pymor.core.logger import getLogger
from pymor.core.defaults import defaults
from pymor.core.exceptions import TRError


@defaults('beta', 'radius', 'shrink_factor', 'miniter', 'maxiter', 'miniter_subproblem', 'maxiter_subproblem',
          'tol', 'radius_tol', 'atol', 'tol_sub', 'stagnation_window', 'stagnation_threshold')
def trust_region(parameter_space, reductor, initial_guess=None, beta=.95, radius=1.,
                 shrink_factor=.5, miniter=0, maxiter=100, miniter_subproblem=0, maxiter_subproblem=100, tol=1e-6,
                 radius_tol=.75, atol=1e-6, tol_sub=1e-6, line_search_params=None, stagnation_window=3,
                 stagnation_threshold=np.inf, return_stages=False, return_subproblem_stages=False):
    """BFGS algorithm.

    This method solves the optimization problem ::

        min J(mu), mu in C

    for an output functional depending on a box-constrained `mu` using an
    adaptive trust region method.

    The main idea for the algorithm can be found in :cite:`YM13`, and an application to
    box-constrained parameters with possible enlarging of the trust radius in :cite:`K21`.

    Parameters
    ----------
    parameter_space
        The |ParameterSpace| for enforcing the box constraints on the parameter `mu`.
    reductor
        The `reductor` used to generate the reduced order models and estimate the output error.
    initial_guess
        If not `None`, a |Mu| instance of length 1 containing an initial guess for
        the solution `mu`.
    beta
        The factor to check if the current parameter is close to the trust region boundary.
    radius
        The radius of the initial trust region.
    shrink_factor
        The factor by which the trust region is shrunk. If the trust region radius is increased,
        it is increased by `1. / shrink_factor`.
    miniter
        Minimum amount of iterations to perform.
    maxiter
        Fail if the iteration count reaches this value without converging.
    miniter_subproblem
        Minimum amount of iterations to perform in the BFGS subproblem.
    maxiter_subproblem
        Fail the BFGS subproblem if the iteration count reaches this value without converging.
    tol
        Finish when the clipped parameter error measure is below this threshold.
    radius_tol
        Threshold for increasing the trust region radius upon extending the reduced order model.
    atol
        Finish the subproblem when the absolute error measure is below this threshold.
    tol_sub
        Finish when the subproblem clipped parameter error measure is below this threshold.
    line_search_params
        Dictionary of additional parameters passed to the line search method.
    stagnation_window
        Finish when the parameter update has been stagnating within a tolerance of
        `stagnation_threshold` during the last `stagnation_window` iterations.
    stagnation_threshold
        See `stagnation_window`.
    return_stages
        If `True`, return a `list` of the intermediate parameter values of `mu` after each
        trust region iteration.
    return_subproblem_stages
        If `True`, return a `list` of the intermediate parameter values of `mu` after each
        BFGS subproblem iteration.

    Returns
    -------
    mu
        |Numpy array| containing the computed parameter.
    data
        Dict containing the following fields:

            :mus:             `list` of parameters after each iteration.
            :mu_norms:        |NumPy array| of the solution norms after each iteration.
            :subproblem_data: `list` of data generated by the individual subproblems.
            :stages:          See `return_stages`.

    Raises
    ------
    TRError
        Raised if the BFGS algorithm failed to converge.
    """
    logger = getLogger('pymor.algorithms.tr')

    data = {}

    if initial_guess is None:
        initial_guess = parameter_space.sample_randomly(1)[0]
        mu = initial_guess.to_numpy()
    else:
        mu = initial_guess.to_numpy() if isinstance(initial_guess, Mu) else initial_guess

    if reductor._last_rom is None:
        initial_basis = reductor.fom.solve(initial_guess)
        reductor.extend_basis(initial_basis)
        reductor.reduce()

    model = reductor._last_rom

    assert shrink_factor != 0.
    assert model.output_functional is not None, 'Please provide an output functional with your model!'

    output = lambda m: model.output(m)[0, 0]
    fom = reductor.fom

    data['subproblem_data'] = []
    if return_stages:
        stages = []

    # compute norms
    mu_norm = np.linalg.norm(mu)
    mu_norms = [mu_norm]
    data['mus'] = [mu.copy()]

    old_rom_output = output(mu)
    old_fom_output = fom.output(mu)[0, 0]

    mu_clip_norm = 1e6
    iteration = 0
    while True:
        with logger.block(f'Running adaptive TR algorithm iteration {iteration} with radius {radius}...'):
            rejected = False

            if iteration >= miniter:
                if mu_clip_norm < tol:
                    logger.info(f'Absolute tolerance of {tol} for parameter clipping error reached. Converged.')
                    break
                if iteration >= maxiter:
                    raise TRError(f'Failed to converge after {iteration} iterations.')

            iteration += 1

            # solve the subproblem using bfgs
            old_mu = mu.copy()
            compare_output = output(mu)
            mu, sub_data = bfgs(
                model, parameter_space, initial_guess=mu, miniter=miniter_subproblem,
                maxiter=maxiter_subproblem, atol=atol, tol_sub=tol_sub,
                line_search_params=line_search_params, stagnation_window=stagnation_window,
                stagnation_threshold=stagnation_threshold, error_aware=True, beta=beta,
                radius=radius, return_stages=return_subproblem_stages)

            estimate_output = model.estimate_output_error(mu)
            current_output = output(mu)

            # check output error conditions
            if current_output + estimate_output < compare_output:
                U_h_mu = fom.solve(mu)
                try:
                    reductor.extend_basis(U_h_mu)
                except Exception:
                    pass
                current_fom_output = fom.output(mu)[0, 0]
                fom_output_diff = old_fom_output - current_fom_output
                rom_output_diff = old_rom_output - current_output
                if fom_output_diff >= radius_tol * rom_output_diff:
                    # increase the radius if the model confidence is high enough
                    radius /= shrink_factor
                model = reductor.reduce()
                old_rom_output = current_output
            elif current_output - estimate_output > compare_output:
                # reject new mu
                rejected = True
                # shrink the radius
                radius *= shrink_factor
            else:
                U_h_mu = fom.solve(mu)
                new_reductor = deepcopy(reductor)
                try:
                    new_reductor.extend_basis(U_h_mu)
                except Exception:
                    pass
                new_rom = new_reductor.reduce()
                current_output = new_rom.output(mu)[0, 0]
                if current_output <= compare_output:
                    current_fom_output = fom.output(mu)[0, 0]
                    fom_output_diff = old_fom_output - current_fom_output
                    rom_output_diff = old_rom_output - current_output
                    if fom_output_diff >= radius_tol * rom_output_diff:
                        # increase the radius if the model confidence is high enough
                        radius /= shrink_factor
                    model = new_rom
                    old_rom_output = current_output
                else:
                    # reject new mu
                    rejected = True
                    # shrink the radius
                    radius *= shrink_factor

            # handle parameter rejection
            if not rejected:
                data['mus'].append(mu.copy())
                mu_norm = np.linalg.norm(mu)
                mu_norms.append(mu_norm)

                data['subproblem_data'].append(sub_data)

                gradient = model.parameters.parse(fom.output_d_mu(mu)).to_numpy()
                mu_clip_norm = np.linalg.norm(mu - parameter_space.clip(mu - gradient).to_numpy())
            else:
                mu = old_mu

            with warnings.catch_warnings():
                # ignore division-by-zero warnings when solution_norm or output is zero
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                logger.info(f'it:{iteration} '
                            f'norm:{mu_norm:.3e} '
                            f'radius:{radius:.3e}')

            if not np.isfinite(mu_norm):
                raise TRError('Failed to converge.')

        logger.info('')

    logger.info('')

    data['mu_norms'] = np.array(mu_norms)
    if return_stages:
        data['stages'] = np.array(stages)

    return mu, data
