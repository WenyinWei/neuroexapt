
"""
\defgroup group_evolvable_model Evolvable Model
\ingroup core
Evolvable Model module for NeuroExapt framework.
"""


import torch
import torch.nn as nn
from .operations import OPS, FactorizedReduce, ReLUConvBN
from .genotypes import Genotype, PRIMITIVES

class EvolvableCell(nn.Module):
    """
    A single cell that represents a discrete architecture based on a genotype.
    Unlike the original Cell that used a weighted sum of operations, this cell
    builds a specific, fixed graph based on the provided genotype.
    """

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(EvolvableCell, self).__init__()
        print(f"Initializing EvolvableCell: C_prev_prev={C_prev_prev}, C_prev={C_prev}, C={C}, reduction={reduction}")

        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]

        return torch.cat([states[i] for i in self._concat], dim=1)


class EvolvableNetwork(nn.Module):
    """
    A network that is built from a discrete architecture specified by a genotype.
    This network can be mutated by providing a new genotype.
    """
    def __init__(self, C, num_classes, layers, genotype, stem_multiplier=3):
        super(EvolvableNetwork, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self.genotype = genotype
        
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            # Determine if current cell is a reduction cell
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
                
            # Select appropriate genotype based on reduction type
            if reduction:
                op_names, indices = zip(*genotype.reduce)
                if len(op_names) != len(indices):
                    raise ValueError("Genotype reduce is not valid")
            else:
                op_names, indices = zip(*genotype.normal)
                if len(op_names) != len(indices):
                    raise ValueError("Genotype normal is not valid")
            
            cell = EvolvableCell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits 