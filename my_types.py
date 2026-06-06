r"""
Type aliases and generics for clarity of Markov Model related \
implementations
---

Aliases defined:

"""

import numpy as np
from typing import *
from collections.abc import Sequence

V = TypeVar("V")
StateT = TypeVar("StateT")  # states' names' type
ValueT = TypeVar("ValueT")  # values' names' type
LabelT = TypeVar("LabelT")  # emission labels' type

Time     = int   # T: sequence length
NStates  = int   # N: no. of states
NValues  = int   # M: no. of possible values for the observations

                                        # matrix dimensions    # data type (almost all store probebilities - floats)
type Vector1D[V]    = np.ndarray[tuple[NStates                  ], V]
type Matrix2D[V]    = np.ndarray[tuple[NStates, NStates         ], V]
type Emission2D[V]  = np.ndarray[tuple[NStates, NValues         ], V]

type AlgArray2D[V]  = np.ndarray[tuple[Time   , NStates         ], V] # psi matrix stores state indices - ints
type AlgArray3D[V]  = np.ndarray[tuple[Time   , NStates, NStates], V]

type SeqInput[V] = Sequence[V] | Sequence[int]      # sequences can be referenced by the names or their indices
