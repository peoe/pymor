# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
from pathlib import Path
from copy import copy
from scipy.sparse import csr_matrix

from pymor.core.config import config
from pymor.core.defaults import defaults
from pymor.tools.io import change_to_directory
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator

if config.HAVE_NGSOLVE:
    import ngsolve as ngs
    import numpy as np

    from pymor.core.base import ImmutableObject
    from pymor.operators.list import LinearComplexifiedListVectorArrayOperatorBase
    from pymor.vectorarrays.interface import VectorArray
    from pymor.vectorarrays.numpy import NumpyVectorSpace
    from pymor.vectorarrays.list import CopyOnWriteVector, ComplexifiedVector, ComplexifiedListVectorSpace

    class NGSolveVectorCommon:
        def amax(self):
            A = np.abs(self.to_numpy())
            max_ind = np.argmax(A)
            max_val = A[max_ind]
            return max_ind, max_val

        def dofs(self, dof_indices):
            return self.to_numpy()[dof_indices]

    class NGSolveVector(NGSolveVectorCommon, CopyOnWriteVector):
        """Wraps a NGSolve BaseVector to make it usable with ListVectorArray."""

        def __init__(self, impl):
            self.impl = impl

        @classmethod
        def from_instance(cls, instance):
            return cls(instance.impl)

        def _copy_data(self):
            new_impl = ngs.GridFunction(self.impl.space)
            new_impl.vec.data = self.impl.vec
            self.impl = new_impl

        def to_numpy(self, ensure_copy=False):
            if ensure_copy:
                return self.impl.vec.FV().NumPy().copy()
            self._copy_data_if_needed()
            return self.impl.vec.FV().NumPy()

        def _scal(self, alpha):
            self.impl.vec.data = float(alpha) * self.impl.vec

        def _axpy(self, alpha, x):
            self.impl.vec.data = self.impl.vec + float(alpha) * x.impl.vec

        def inner(self, other):
            return self.impl.vec.InnerProduct(other.impl.vec)

        def norm(self):
            return self.impl.vec.Norm()

        def norm2(self):
            return self.impl.vec.Norm() ** 2

    class ComplexifiedNGSolveVector(NGSolveVectorCommon, ComplexifiedVector):
        pass

    class NGSolveVectorSpace(ComplexifiedListVectorSpace):

        complexified_vector_type = ComplexifiedNGSolveVector

        def __init__(self, V, id='STATE'):
            self.__auto_init(locals())

        def __eq__(self, other):
            return type(other) is NGSolveVectorSpace and self.V == other.V and self.id == other.id

        def __hash__(self):
            return hash(self.V) + hash(self.id)

        @property
        def value_dim(self):
            u = self.V.TrialFunction()
            if isinstance(u, list):
                return u[0].dim
            else:
                return u.dim

        @property
        def dim(self):
            return self.V.ndofglobal * self.value_dim

        @classmethod
        def space_from_vector_obj(cls, vec, id):
            return cls(vec.space, id)

        def real_zero_vector(self):
            impl = ngs.GridFunction(self.V)
            return NGSolveVector(impl)

        def real_make_vector(self, obj):
            return NGSolveVector(obj)

        def real_vector_from_numpy(self, data, ensure_copy=False):
            v = self.real_zero_vector()
            v.to_numpy()[:] = data
            return v

    class NGSolveMatrixOperator(LinearComplexifiedListVectorArrayOperatorBase):
        """Wraps a NGSolve matrix as an |Operator|."""

        def __init__(self, matrix, range, source, solver_options=None, name=None):
            self.__auto_init(locals())

        @defaults('default_solver')
        def _prepare_apply(self, U, mu, kind, least_squares=False, default_solver=''):
            if kind == 'apply_inverse':
                if least_squares:
                    raise NotImplementedError
                solver = self.solver_options.get('inverse', default_solver) if self.solver_options else default_solver
                inv = self.matrix.Inverse(self.source.V.FreeDofs(), inverse=solver)
                return inv

        def _real_apply_one_vector(self, u, mu=None, prepare_data=None):
            r = self.range.real_zero_vector()
            self.matrix.Mult(u.impl.vec, r.impl.vec)
            return r

        def _real_apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
            u = self.source.real_zero_vector()
            try:
                mat = self.matrix.Transpose()
            except AttributeError:
                mat = self.matrix.T
            mat.Mult(v.impl.vec, u.impl.vec)
            return u

        def _real_apply_inverse_one_vector(self, v, mu=None, initial_guess=None,
                                           least_squares=False, prepare_data=None):
            inv = prepare_data
            r = self.source.real_zero_vector()
            r.impl.vec.data = inv * v.impl.vec
            return r

        def _assemble_lincomb(self, operators, coefficients, identity_shift=0., solver_options=None, name=None):
            if not all(isinstance(op, NGSolveMatrixOperator) for op in operators):
                return None
            if identity_shift != 0:
                return None

            matrix = operators[0].matrix.CreateMatrix()
            matrix.AsVector().data = float(coefficients[0]) * matrix.AsVector()
            for op, c in zip(operators[1:], coefficients[1:]):
                matrix.AsVector().data += float(c) * op.matrix.AsVector()
            return NGSolveMatrixOperator(matrix, self.range, self.source, solver_options=solver_options, name=name)

        def as_vector(self, copy=True):
            vec = self.matrix.AsVector().FV().NumPy()
            return NumpyVectorSpace.make_array(vec.copy() if copy else vec)

    class NGSolveVisualizer(ImmutableObject):
        """Visualize an NGSolve grid function."""

        def __init__(self, mesh, fespace):
            self.__auto_init(locals())
            self.space = NGSolveVectorSpace(fespace)

        def visualize(self, U, m, legend=None, separate_colorbars=True, filename=None, block=True):
            """Visualize the provided data."""
            if isinstance(U, VectorArray):
                U = (U,)
            assert all(u in self.space for u in U)
            if any(len(u) != 1 for u in U):
                raise NotImplementedError
            if any(u._list[0].imag_part is not None for u in U):
                raise NotImplementedError
            if legend is None:
                legend = [f'VectorArray{i}' for i in range(len(U))]
            if isinstance(legend, str):
                legend = [legend]
            assert len(legend) == len(U)
            legend = [l.replace(' ', '_') for l in legend]  # NGSolve GUI will fail otherwise

            if filename:
                # ngsolve unconditionnaly appends ".vtk"
                filename = Path(filename).resolve()
                if filename.suffix == '.vtk':
                    filename = filename.parent / filename.stem
                else:
                    self.logger.warning(f'NGSolve set VTKOutput filename to {filename}.vtk')
                coeffs = [u._list[0].real_part.impl for u in U]
                # ngsolve cannot handle full paths for filenames
                with change_to_directory(filename.parent):
                    vtk = ngs.VTKOutput(ma=self.mesh, coefs=coeffs, names=legend, filename=str(filename), subdivision=0)
                    vtk.Do()
            else:
                if not separate_colorbars:
                    raise NotImplementedError

                for u, name in zip(U, legend):
                    ngs.Draw(u._list[0].real_part.impl, self.mesh, name=name)

    class NGSolveOperator(Operator):
        """Wraps a general NGSolve Bilinear form as an |Operator|."""

        linear = False
        parametric = True

        def __init__(self, form, source_space, range_space, dirichlet_bc, boundary_penalty=None,
                     parameter_setter=None, parameters={}, solver_options=None, name=None):
            self.__auto_init(locals())
            self.source = source_space
            self.range = range_space
            self.parameters_own = parameters
            self.unfree = np.array([not b for b in self.range.V.FreeDofs()])

        def set_dirichlet_boundary_values(self, vecarray):
            for u in vecarray._list:
                np.putmask(u.real_part.impl.vec.FV().NumPy(), self.unfree, self.dirichlet_bc.vec)

        def _set_mu(self, mu=None):
            assert self.parameters.assert_compatible(mu)
            if self.parameter_setter:
                self.parameter_setter(mu)

        def apply(self, U, mu=None):
            assert U in self.source
            self._set_mu(mu)
            R = []

            for u in U._list:
                if u.imag_part is not None:
                    raise NotImplementedError
                r = self.range.zero_vector()
                self.form.Apply(u.real_part.impl.vec, r.real_part.impl.vec)
                np.putmask(r.real_part.impl.vec.FV().NumPy(), self.unfree, 0)
                R.append(r)
            return self.range.make_array(R)

        def jacobian(self, U, mu=None):
            assert U in self.source and len(U) == 1
            if U._list[0].imag_part is not None:
                raise NotImplementedError
            self._set_mu(mu)

            self.form.AssembleLinearization(U._list[0].real_part.impl.vec)
            matrix = self.form.mat
            copy = matrix.CreateMatrix()
            copy.AsVector().data = matrix.AsVector()
            return NGSolveMatrixOperator(copy, self.range, self.source)

        def restricted(self, restrict_to_dofs):
            re_elements = set()
            affected_element_nodes = set()
            for element in self.range.V.Elements():
                for element_dof in element.dofs:
                    if element_dof in restrict_to_dofs:
                        re_elements.add(element)
                        affected_element_nodes.add(element.elementnode)
            source_dofs = set()
            for element in self.source.V.Elements():
                if element.elementnode in affected_element_nodes:
                    source_dofs.update(element.dofs)
            source_dofs = np.array(sorted(source_dofs), dtype=np.intc)
            return RestrictedNGSolveOperator(copy(self), source_dofs, restrict_to_dofs,
                                             re_elements), source_dofs

        # pyMOR's own newton performs much better in its default settings than this
        # def apply_inverse(self, V, mu=None, initial_guess=None, least_squares=False):
        #     if least_squares or len(V) > 1:
        #         raise NotImplementedError
        #     self._set_mu(mu)
        #
        #     result = ngs.GridFunction(self.range.V)
        #     if initial_guess:
        #         result.vec.data = initial_guess._list[0].real_part.impl.vec
        #         assert(False)
        #     else:
        #         result.vec.data = self.dirichlet_bc.vec
        #
        #     ngs.solvers.Newton(a=self.form, u=result, dirichletvalues=self.dirichlet_bc.vec)
        #     rr = self.range.zeros(len(V))
        #
        #     rr._list[0].real_part.impl.vec.data = result.vec
        #     self.set_dirichlet_boundary_values(rr)
        #     return rr


    class RestrictedNGSolveOperator(Operator):

        linear = False

        def __init__(self, unrestricted_op, source_dofs, restricted_range_dofs, restricted_elements):
            self.__auto_init(locals())
            self.source = NumpyVectorSpace(len(source_dofs))
            self.range = NumpyVectorSpace(len(restricted_range_dofs))
            self.source_dof_inv_map, c = [-1]*unrestricted_op.source.dim, 0
            for ii in range(self.unrestricted_op.source.dim):
                if ii in source_dofs:
                    self.source_dof_inv_map[ii] = c
                    c += 1
            self.range_dof_inv_map, c = [-1]*unrestricted_op.range.dim, 0
            for ii in range(self.unrestricted_op.range.dim):
                if ii in restricted_range_dofs:
                    self.range_dof_inv_map[ii] = c
                    c += 1

        def apply(self, U, mu=None):
            assert U in self.source
            self.unrestricted_op._set_mu(mu)

            UU = self.unrestricted_op.source.zeros(len(U))
            for uu, u in zip(UU._list, U.to_numpy()):
                uu.real_part.to_numpy()[self.source_dofs] = np.ascontiguousarray(u)

            VV = self.unrestricted_op.range.zeros(len(U))

            for element in self.restricted_elements:
                local_dofs = element.dofs
                finite_elment = self.unrestricted_op.source.V.GetFE(element)

                trafo = element.GetTrafo()
                for uu, vv in zip(UU._list,VV._list):
                    uu_vec = uu.real_part.impl.vec
                    uu_loc = ngs.Vector([uu_vec[dof] for dof in local_dofs])
                    for integrator in self.unrestricted_op.form.integrators:
                        vv.real_part.to_numpy()[local_dofs] += integrator.ApplyElementMatrix(finite_elment, uu_loc, trafo)

            V = self.range.zeros(len(U))
            for v, vv in zip(V.to_numpy(), VV._list):
                vv_np = vv.real_part.to_numpy()
                v[:] = vv_np[self.restricted_range_dofs]
            return V

        def jacobian(self, U, mu=None):
            assert U in self.source and len(U) == 1
            self.unrestricted_op._set_mu(mu)
            UU = self.unrestricted_op.source.zeros()
            UU._list[0].real_part.to_numpy()[self.source_dofs] = np.ascontiguousarray(U.to_numpy()[0])
            full_shape = (self.unrestricted_op.source.dim, self.unrestricted_op.range.dim)

            values, rows, cols = [], [], []
            for element in self.restricted_elements:
                local_dofs = element.dofs
                finite_elment = self.unrestricted_op.source.V.GetFE(element)

                trafo = element.GetTrafo()

                uu_vec = UU._list[0].real_part.impl.vec
                uu_loc = ngs.Vector([uu_vec[dof] for dof in local_dofs])
                element_matrix = ngs.Matrix(len(local_dofs), len(local_dofs))
                element_matrix *= 0
                for integrator in self.unrestricted_op.form.integrators:
                    element_matrix += integrator.CalcLinearizedElementMatrix(finite_elment, uu_loc, trafo)

                for ii, row in enumerate(element_matrix.NumPy()):
                    for jj, val in enumerate(row):
                        rows.append(local_dofs[ii])
                        cols.append(local_dofs[jj])
                        values.append(val)
            matrix = csr_matrix((values, (rows, cols)), shape=full_shape)
            re_mat = matrix[:, self.source_dofs][self.restricted_range_dofs, :]
            return NumpyMatrixOperator(re_mat)
