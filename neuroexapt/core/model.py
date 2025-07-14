
import torch
import torch.nn as nn
from .operations import OPS, FactorizedReduce, ReLUConvBN, MixedOp
from .genotypes import PRIMITIVES, Genotype

class Cell(nn.Module):
    """
    A single cell in the network, represented as a DAG.
    Each cell consists of a set of nodes, where each node computes a feature map
    by applying mixed operations to the feature maps of its predecessors.
    """

    def __init__(self, steps, block_multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        self._C_out = C * block_multiplier

        # In a reduction cell, the previous cell's output is down-sampled
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._block_multiplier = block_multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        node_C_in = C # Input channels for all nodes in this cell
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                # The output of each MixedOp is C, ready for the next node
                op = MixedOp(node_C_in, C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._block_multiplier:], dim=1)


class Network(nn.Module):
    """
    The full neural network model, composed of a stack of cells.
    This class also initializes and stores the architecture parameters (alphas).
    """

    def __init__(self, C, num_classes, layers, potential_layers=4, steps=4, block_multiplier=4):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._potential_layers = potential_layers
        self._steps = steps
        self._block_multiplier = block_multiplier

        C_curr = self._block_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        total_layers = layers + potential_layers
        for i in range(total_layers):
            # Reduction cells are placed based on the initial layer count
            if i < layers and i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            
            cell = Cell(steps, block_multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            
            # Wrap potential layers in a GatedCell
            if i >= layers:
                cell = GatedCell(cell, C_prev_prev, C_curr)

            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, self._block_multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        """Create a new model with the same architecture but uninitialized weights."""
        model_new = Network(self._C, self._num_classes, self._layers, self._potential_layers).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, GatedCell):
                # For gated cells, use the same weights as normal cells for now
                # A more advanced strategy could have separate weights for potential layers
                if cell.cell.reduction:
                    weights = torch.softmax(self.alphas_reduce, dim=-1)
                else:
                    weights = torch.softmax(self.alphas_normal, dim=-1)
            elif cell.reduction:
                weights = torch.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = torch.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _initialize_alphas(self):
        """Initialize the architecture parameters alpha."""
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        
        # Calculate the number of operations in a MixedOp
        # This is now more complex due to channel variations
        C_in_for_ops = self._C # An example channel count
        channel_options_len = 3 # half, same, double
        num_conv_ops = len([p for p in OPS if 'conv' in p])
        num_non_conv_ops = len(OPS) - num_conv_ops
        num_ops_per_mixedop = num_conv_ops * channel_options_len + num_non_conv_ops

        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops_per_mixedop))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops_per_mixedop))
        
        # Initialize gates for potential cells
        self.alphas_gates = nn.ParameterList(
            [cell.gate for cell in self.cells if isinstance(cell, GatedCell)]
        )

        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]
        # Register the gate parameters with the architect
        self._arch_parameters.extend(self.alphas_gates)

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        """
        Decodes the learned architecture parameters (alphas) into a discrete genotype.
        This method determines the final architecture of the network after search.
        """

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                
                # For each node, find the two strongest predecessor connections
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')) if any(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')) else -1.0)[:2]
                
                # For each of the two selected edges, find the best operation
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    # Default to a safe choice if all options are zero
                    if k_best is None:
                        k_best = 1 # 'max_pool_3x3' as a fallback
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        # Normalize the alpha values using softmax
        gene_normal = _parse(torch.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(torch.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self._steps - self._block_multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

class Resizing(nn.Module):
    """
    A utility module to resize tensors to a target channel count.
    This is used to match channel dimensions when operations with different
    channel counts are mixed.
    """
    def __init__(self, C_in, C_out, affine=True):
        super(Resizing, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    
    def forward(self, x):
        if self.C_in == self.C_out:
            return x
        return self.op(x)

class GatedCell(nn.Module):
    """
    A wrapper for a cell that includes a learnable gate to control its contribution.
    This version implements a function-preserving residual connection, ensuring that
    when the gate is closed, it acts as an identity connection.
    """
    def __init__(self, cell, C_in, C_out):
        super(GatedCell, self).__init__()
        self.cell = cell
        self.gate = nn.Parameter(torch.randn(1) * 1e-3)
        # Resizer for the identity path to match the cell's output dimensions
        self.identity_resizer = Resizing(C_in, C_out, affine=False)

    def forward(self, s0, s1, weights):
        """
        Computes: gate * cell(s0, s1) + (1 - gate) * identity(s1)
        """
        gate_val = torch.sigmoid(self.gate)
        cell_out = self.cell(s0, s1, weights)
        # The identity path is based on the s1 state
        identity_out = self.identity_resizer(s1)
        
        return gate_val * cell_out + (1 - gate_val) * identity_out 