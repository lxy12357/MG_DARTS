from collections import namedtuple

Genotype = namedtuple('Genotype', 'gene concat')

PRIMITIVES = [
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'avg_pool_3x3',
    'max_pool_3x3',
]


