import numpy as np
from typing import *
from collections.abc import Sequence

V = TypeVar("V")
StateT = TypeVar("StateT")  # tipo dos nomes dos estados
ValueT = TypeVar("ValueT")  # tipo dos nomes dos valores
LabelT = TypeVar("LabelT")  # tipo das labels de emissão

Time     = int   # T: comprimento da cadeia
NStates  = int   # N: nº de estados
NValues  = int   # M: nº de valores possíveis para as observações

                            # dimensões das matrizes    # tipo de dados (quase todas guardam probabilidades - floats)
type Vector1D[V]    = np.ndarray[tuple[NStates                  ], V]
type Matrix2D[V]    = np.ndarray[tuple[NStates, NStates         ], V]
type Emission2D[V]  = np.ndarray[tuple[NStates, NValues         ], V]

type AlgArray2D[V]  = np.ndarray[tuple[Time   , NStates         ], V] # matriz psi guarda índices de estados - ints
type AlgArray3D[V]  = np.ndarray[tuple[Time   , NStates, NStates], V]

type SeqInput[V] = Sequence[V] | Sequence[int]