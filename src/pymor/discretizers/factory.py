# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from functools import partial

from pymor.core.base import ImmutableObject
from pymor.analyticalproblems.functions import LincombFunction
from pymor.operators.constructions import LincombOperator, NumpyConversionOperator
from pymor.operators.block import BlockColumnOperator


class GenericOutputFactory(ImmutableObject):

    ALLOWED_KEYS = ['l2', 'l2_boundary']

    def __init__(self, output_list):
        self.__auto_init(locals())
        for o in output_list:
            assert o[0] in self.ALLOWED_KEYS
        
        self.parsers = self._construct_parsers()

    def _parser(self, output):
        raise NotImplementedError

    def _construct_parsers(self):
        return [self._parser] * len(self.ALLOWED_KEYS)

    def _parse_output(self, output):
        key_index = self.ALLOWED_KEYS.index(output[0])
        return self.parsers[key_index](output[1])

    def _construct_outputs(self, output_list):
        outputs = []
        for o in output_list:
            op = self._parse_output(o)
            outputs.append(op)
        return outputs

    def construct_output_functional(self):
        outputs = self._construct_outputs(self.output_list)
        if len(outputs) > 1:
            output_functional = BlockColumnOperator(outputs)
            output_functional = NumpyConversionOperator(output_functional.range) @ output_functional
        else:
            output_functional = outputs[0]
        return output_functional


class OutputFactory(GenericOutputFactory):

    def __init__(self, output_list, grid, functional_list):
        assert len(functional_list) == len(self.ALLOWED_KEYS)
        self.functional_list = functional_list
        super().__init__(output_list)
        self.__auto_init(locals())
    
    def _parser(self, output, functional):
        if isinstance(output, LincombFunction):
            ops = [functional(self.grid, vv, dirichlet_clear_dofs=False).H
                    for vv in output.functions]
            op = LincombOperator(ops, output.coefficients)
        else:
            op = functional(self.grid, output, dirichlet_clear_dofs=False).H
        return op

    def _construct_parsers(self):
        parsers = []
        for i in range(len(self.functional_list)):
            parsers.append(partial(self._parser, functional=self.functional_list[i]))
        return parsers


class FenicsOutputFactory(GenericOutputFactory):

    def __init__(self, output_list, mesh, form_expressions):
        super().__init__(output_list)
        self.__auto_init(locals())

    def _construct_parsers(self):
        from pymor.discretizers.fenics.cg import _assemble_operator
        parsers = []
        for i in range(len(self.form_expressions)):
            parsers.append(partial(_assemble_operator, factory=self.form_expressions[i], mesh=self.mesh, name=self.ALLOWED_KEYS[i]))
        return parsers


class SKFemOutputFactory(GenericOutputFactory):

    def __init__(self, output_list, mesh, form_expressions):
        super().__init__(output_list)
        self.__auto_init(locals())

    def _parser(self, output, form_expression, name):
        from pymor.discretizers.skfem.cg import _assemble_operator
        return _assemble_operator(output[1], name+'_output', form_expression)

    def _construct_parsers(self):
        parsers = []
        for i in range(len(self.form_expressions)):
            parsers.append(partial(self._parse, form_expression=self.form_expressions[i], name=self.ALLOWED_KEYS[i]))
        return parsers
