r"""
Type aliases and generics for clarity of Markov Model related implementations.

Attributes
----------
StateT : TypeVar
    Generic type variable representing state identifiers.
    (typically `str` or `int`).

ValueT : TypeVar 
    Generic type variable representing emission/observation values.
    (typically `str` or `int`).

LabelT : TypeVar
    Generic type variable representing labels applied during hidden state tracebacks.

SeqInput 
    Generic type alias factory for sequence arrays. 
    Accepts a sequence of user-defined items or integer indexes.
    Defined as: `Union[Sequence[V], Sequence[int]]`


Time    : TypeAlias 
    Semantic alias for an integer (`int`) representing sequence length.

NStates : TypeAlias 
    Semantic alias for an integer (`int`) representing number of states.

NValues : TypeAlias 
    Semantic alias for an integer (`int`) representing number of possible values for the observations.


Vector1D : TypeAlias
    1-dimensional NumPy array representing probability vectors such as
    initial state distributions or stationary state distributions. 
    Shape: `(Nstates,)`

Matrix2D : TypeAlias
    2-dimensional NumPy array representing transition probability matrices.
    Shape: `(NStates, NStates)`

Emission2d : TypeAlias
    2-dimensional NumPy array representing emission probability matrices. 
    Shape `(NStates, NValues)`

AlgArray2D : TypeAlias
    2-dimensional NumPy array used in the dynamic programming algorithms.
    Shape: `(Time, NStates)`

AlgArray3D : TypeAlias
    3-dimensional NumPy array used in the dynamic programming algorithms
    (specifically, the xi matrix in Baum-Welch steps).
    Shape: `(Time, NStates, NStates)`

"""

import numpy as np
from typing import *
from collections.abc import Sequence

StateT = TypeVar("StateT")  # states' names' type
ValueT = TypeVar("ValueT")  # values' names' type
LabelT = TypeVar("LabelT")  # emission labels' type

V = TypeVar("V")
type SeqInput[V] = Sequence[V] | Sequence[int]      # sequences can be referenced by the names or their indices


type Time     = int   # T: sequence length
type NStates  = int   # N: no. of states
type NValues  = int   # M: no. of possible values for the observations

                                        # matrix dimensions    # data type (almost all store probebilities - floats)
type Vector1D[V]    = np.ndarray[tuple[NStates                  ], V]
type Matrix2D[V]    = np.ndarray[tuple[NStates, NStates         ], V]
type Emission2D[V]  = np.ndarray[tuple[NStates, NValues         ], V]

type AlgArray2D[V]  = np.ndarray[tuple[Time   , NStates         ], V] # psi matrix stores state indices - ints
type AlgArray3D[V]  = np.ndarray[tuple[Time   , NStates, NStates], V]
