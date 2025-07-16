
from collections import namedtuple

# Define a standard set of primitive operations for genotype representation
# Must match the keys in operations.OPS exactly
PRIMITIVES = [
    'none',
    'avg_pool_3x3',
    'max_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'conv_7x1_1x7'
]

# Define the Genotype structure using a namedtuple for clarity and immutability
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat') 