#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from typer import Argument, run
import time

from pymor.tools.io import ShiftedVisualizer
from pymor.tools.typer import Choices


def main(
    dim: int = Argument(..., help='Spatial dimension of the problem.'),
    n: int = Argument(..., help='Number of mesh intervals per spatial dimension.'),
    order: int = Argument(..., help='Finite element order.'),
    model: Choices('fenics ngsolve') = Argument(..., help='High-dimensional model.'),
):
    t_all = time.perf_counter()
    """Reduces a FEniCS/NgSolve-based nonlinear diffusion problem using POD/DEIM."""
    if model == 'fenics':
        from pymor.tools import mpi

        if mpi.parallel:
            from pymor.models.mpi import mpi_wrap_model
            local_models = mpi.call(mpi.function_call_manage, discretize_fenics, dim, n, order)
            fom = mpi_wrap_model(local_models, use_with=True, pickle_local_spaces=False)
        else:
            fom = discretize_fenics(dim, n, order)
    elif model == 'ngsolve':
        fom = discretize_ngsolve(dim, n, order)
    else:
        raise NotImplementedError()

    parameter_space = fom.parameters.space((0, 1000.))

    # ### ROM generation (POD/DEIM)
    import numpy as np

    from pymor.algorithms.ei import ei_greedy
    from pymor.algorithms.newton import newton
    from pymor.algorithms.pod import pod
    from pymor.operators.ei import EmpiricalInterpolatedOperator
    from pymor.reductors.basic import StationaryRBReductor

    U = fom.solution_space.empty()
    residuals = fom.solution_space.empty()
    for ii, mu in enumerate(parameter_space.sample_uniformly(10)):
        UU, data = newton(fom.operator, fom.rhs.as_vector(), mu=mu, rtol=1e-6, return_residuals=True)
        U.append(UU)
        fom.visualize(UU, filename=f'{model}_full_mu={mu["c"][0]}.pvd')
        uu_res = fom.operator.apply(UU, mu)
        fom.visualize(uu_res, filename=f'{model}_res_full_mu={mu["c"][0]}.pvd')
        assert uu_res.norm() < 1e-7
        residuals.append(data['residuals'])

    dofs, cb, _ = ei_greedy(residuals, rtol=1e-7)
    ei_op = EmpiricalInterpolatedOperator(fom.operator, collateral_basis=cb, interpolation_dofs=dofs, triangular=True)
    rb, svals = pod(U, rtol=1e-7)
    fom_ei = fom.with_(operator=ei_op)
    reductor = StationaryRBReductor(fom_ei, rb)
    rom = reductor.reduce()
    # the reductor currently removes all solver_options so we need to add them again
    rom = rom.with_(operator=rom.operator.with_(solver_options=fom.operator.solver_options))

    # ### ROM validation
    # ensure that FFC is not called during runtime measurements
    rom.solve(1)
    t_after_ffc = time.perf_counter()

    errs = []
    speedups = []
    for mu in parameter_space.sample_randomly(10):
        tic = time.perf_counter()
        U = fom.solve(mu)
        t_fom = time.perf_counter() - tic
        tic = time.perf_counter()
        u_red = rom.solve(mu)
        t_rom = time.perf_counter() - tic
        U_red = reductor.reconstruct(u_red)
        abs_err = (U - U_red).norm()
        errs.append((abs_err / U.norm())[0])
        speedups.append(t_fom / t_rom)
        fom.visualize(U_red, filename=f'{model}_rom_mu={mu["c"][0]}.pvd')
        fom.visualize(U, filename=f'{model}_full_mu={mu["c"][0]}.pvd')
        u_res = rom.operator.apply(u_red, mu)
        fom.visualize(reductor.reconstruct(u_res), filename=f'{model}_res_rom_mu={mu["c"][0]}.pvd')
        assert u_res.norm() < 1e-7
    t_all = time.perf_counter() - t_all
    t_after_ffc = time.perf_counter() - t_after_ffc
    print(f'Maximum relative ROM error ({model}): {max(errs):e}')
    print(f'Median relative ROM error ({model}):  {np.median(errs):e}')
    print(f'Maximum of ROM speedup ({model}):     {max(speedups):e}')
    print(f'Median of ROM speedup ({model}):      {np.median(speedups):e}')
    print(f'Overall time ({model}):               {t_all:e}')
    print(f'Time after first solve ({model}):     {t_after_ffc:e}')
    print(f'POD mode count ({model}): {len(rb)}')
    print(f'IDOF count ({model}):     {len(dofs)}')
    print(f'IDOFs({model}): {dofs}')

    fom.visualize(U_red, filename=f'{model}_reconstructed_mu={mu["c"][0]}.pvd')
    fom.visualize(U, filename=f'{model}_full_mu={mu["c"][0]}.pvd')


def discretize_fenics(dim, n, order):
    # ### problem definition
    import dolfin as df

    if dim == 2:
        mesh = df.UnitSquareMesh(n, n)
    elif dim == 3:
        mesh = df.UnitCubeMesh(n, n, n)
    else:
        raise NotImplementedError

    V = df.FunctionSpace(mesh, "CG", order)

    g = df.Constant(1.0)
    c = df.Constant(1.)

    class DirichletBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return abs(x[0] - 1.0) < df.DOLFIN_EPS and on_boundary
    db = DirichletBoundary()
    bc = df.DirichletBC(V, g, db)

    u = df.Function(V)
    v = df.TestFunction(V)
    f = df.Expression("x[0]*sin(x[1])", degree=2)
    F = df.inner((1 + (c*u)**2)*df.grad(u), df.grad(v))*df.dx - f*v*df.dx

    df.solve(F == 0, u, bc,
             solver_parameters={"newton_solver": {"relative_tolerance": 1e-6}})

    # ### pyMOR wrapping
    from pymor.bindings.fenics import FenicsVectorSpace, FenicsOperator, FenicsVisualizer
    from pymor.models.basic import StationaryModel
    from pymor.operators.constructions import VectorOperator

    space = FenicsVectorSpace(V)
    op = FenicsOperator(F, space, space, u, (bc,),
                        parameter_setter=lambda mu: c.assign(mu['c'].item()),
                        parameters={'c': 1},
                        solver_options={'inverse': {'type': 'newton', 'rtol': 1e-6}})
    rhs = VectorOperator(op.range.zeros())

    fom = StationaryModel(op, rhs,
                          visualizer=FenicsVisualizer(space))

    return fom


def discretize_ngsolve(dim, n, order):
    # ### problem definition
    from ngsolve import (GridFunction, BND, Mesh, H1, CoefficientFunction, LinearForm,SymbolicBFI,
                         VTKOutput,
                         BilinearForm, Preconditioner, grad, solvers, sin, InnerProduct, dx, ds, Parameter)
    from ngsolve import x as x_expr, y as y_expr
    from netgen.csg import unit_cube
    from netgen.geom2d import unit_square

    if dim == 2:
        mesh = Mesh(unit_square.GenerateMesh(maxh=1/n))
    elif dim == 3:
        mesh = Mesh(unit_cube.GenerateMesh(maxh=1/n))
    else:
        raise NotImplementedError

    # V = H1(mesh, order=order, dirichlet='right|left|top|bottom')
    V = H1(mesh, order=order, dirichlet='right')

    g = CoefficientFunction(1.0)
    c = Parameter(1.)
    bc = GridFunction(V)
    bc.Set(g, definedon=mesh.Boundaries("right"))

    v = V.TestFunction()
    u = V.TrialFunction()
    f = x_expr*sin(y_expr)
    F = BilinearForm(V, symmetric=False)
    # us = bc-u
    us = u+bc
    F += SymbolicBFI(InnerProduct((1 + c*us*us)*(grad(u)+grad(bc)), grad(v)) - f*v)
    #penalty = n
    #F += penalty*(u-g)*v*ds("right")

    # ### pyMOR wrapping
    from pymor.bindings.ngsolve import NGSolveVectorSpace, NGSolveOperator, NGSolveVisualizer
    from pymor.models.basic import StationaryModel
    from pymor.operators.constructions import VectorOperator

    space = NGSolveVectorSpace(V)
    op = NGSolveOperator(F, space, space, dirichlet_bc=bc,
                        parameter_setter=lambda mu: c.Set(mu['c'].item()),
                        parameters={'c': 1},
                        solver_options={'inverse': {'type': 'newton', 'rtol': 1e-6}})
    rhs = VectorOperator(op.range.zeros())

    shift = space.make_array((bc,))
    # shift = space.zeros()
    # F.Apply(bc.vec, shift._list[0].real_part.impl.vec)

    og = NGSolveVisualizer(mesh, V)
    vis = ShiftedVisualizer(og, -shift)
    og.visualize(shift, m=None, filename='shift.vtk')
    fom = StationaryModel(op, rhs,
                          visualizer=vis)
    return fom


if __name__ == '__main__':
    run(main)
