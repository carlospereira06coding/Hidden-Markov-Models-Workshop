r"""
Auxiliary mathematical and data transformation utilities for the HMM implementations.

Functions
---------
* **Random Matrix Generation**:
    `random_sum(n, s=1)` 
        Creates an array of `n` random numbers that sum to `s`.
    `random_sum_mat(n, m=None, s=1)`
        Creates an `n`x`m` matrix where each row sums to `s`.

* **Log-Space Math**:
    `logaddexp10(x1, x2)`
        Computes a base-10 log-sum-exp between `x1` and `x2`.
    `logaddexp10_reduce(arr, axis=None)`
        Computes a base-10 log-sum-exp along `arr`.

* **State Translation**:
    `names_to_indexes(sequence, mapping)`
        Translates a `sequence` of state/value names to an array of indexes.
    `indexes_to_names(sequence, mapping)`
        Translates a `sequence` of state/value indexes to an array of their user-defined names.

* **Others**:
    `seqobs_pretty_print(seq, obs)`
        Prints a readable alignment between a state sequence and an observation sequence.
    `parse_phobius_model(filename)`
        Translates the phobius model to the proper variables.
"""

from __future__ import annotations

import textwrap
import numpy as np
from numpy.typing import NDArray
from my_types import *


def random_sum(n, s=1):
    """Generates an array of `n` random numbers that add to `s`.

    
    Args
    ----
    n : int
        Desired length of the array.

    s : int, optional, default 1
        Total sum of the array.

        
    Returns
    -------
    array : array_like
        The created array.
    """

    array = []
    lim = s
    for _ in range(n-1):
        v = np.random.random()*lim
        array.append(v)
        lim -= v
    array.append(s-np.sum(array))
    return array

def random_sum_mat(n, m=None, s=1, *, seed=None):
    """Generates a 2d matrix of dimensions `n`x`m` if
    `m` is provided, otherwise, a square `n`x`n` matrix 
    where each row sums to `s`.

    
    Args
    ----
    n : int
        Number of rows.

    m : int, optional
        Number of columns. Defaults to `n`.

    s : int, optional, default 1
        Total that each row sums to.

    seed : int, optional 
        Seed integer passed to `np.random.seed` for reproducible RNG.

        
    Returns
    -------
    mat : array_like
        The created matrix.
    """

    np.random.seed(seed)
    m = m or n
    mat=[]
    for _ in range(n):
        mat.append(random_sum(m, s))
    return mat


def names_to_indexes(name_sequence: SeqInput[V], index_map: list[V]) -> NDArray[np.int64]:
    """Transforms `name_sequence` into their respective indices in `index_map`.

    
    Args
    ----
    name_sequence : SeqInput of V
        The sequence of names to translate.

    index_map : list of V
        The name list to use as reference.

        
    Returns
    -------
    indices : ndarray of int
        The correspondent index array.
    """

    if isinstance(name_sequence[0], (int, np.int64)): return np.array(name_sequence)
    return np.array(list(map(lambda val: index_map.index(val), name_sequence)))

def indexes_to_names(index_sequence: Sequence[int], name_map: list[V]) -> list[V]:
    """Transforms `index_sequence` into their respective names in `index_map`.

    
    Args
    ----
    index_sequence : Sequence of int
        The sequence of indices to translate.

    name_map : list of V
        The name list to use as reference.

        
    Returns:
        names (list of V): The correspondent name array.
    """
    return list(map(lambda val: name_map[val], index_sequence))


def seqobs_pretty_print(seq: Sequence, obs: Sequence) -> str:
    """Generates a readable alignment between `seq` and `obs`.

    If all elements of `seq` and `obs` are single characters, 
    there is no space between characters and the state sequence
    is directly below the observations, otherwise, consecutive
    states are spaced out and separated by arrows, and observations
    are presented below the states.


    Args
    ----
    seq : Sequence
        Sequence of states.
    
    obs : Sequence
        Sequence of observations.


    Raises
    ------
    ValueError
        If `seq` and `obs` are not the same length.

        
    Returns
    -------
    message : str
        The created alignment.
    """

    if len(seq) != len(obs):
        raise ValueError("sequences must be the same length")
    
    idx_lim = int(np.log10(len(seq))) + 1
    seq = [str(v) for v in seq]
    obs = [str(v) for v in obs]

    single = np.all(np.array(list(map(len, seq)) + list(map(len, obs)))==1)
    if single:
        seq = "".join(seq)
        obs = "".join(obs)

        seq_slices = textwrap.wrap(seq, 60)
        obs_slices = textwrap.wrap(obs, 60)

        msg = ""
        i = 1
        for line_i in range(len(seq_slices)):
            str_i = str(i); decimals = len(str_i)
            msg += " "*(idx_lim-decimals) + str_i + " " + obs_slices[line_i] + "\n"
            msg += " "*(idx_lim-decimals) + str_i + " " + seq_slices[line_i] + "\n\n"
            i += 60

        return msg

    lines = []
    w_msg = " "*(idx_lim-1) + str(1) + "  "
    o_msg = " "*(idx_lim-1) + str(1) + "  "
    length = 0
    for i in range(len(seq)):
        state = seq[i]; l_stt = len(state)
        value = obs[i]; l_val = len(value)

        max_len = max(l_stt, l_val)
        length += max_len +2

        if length > 60:
            lines.append(w_msg + "\n" + o_msg + "\n")
            length = max_len +2
            
            str_i = str(i); decimals = len(str_i)
            w_msg = " "*(idx_lim-decimals) + str_i + "  "
            o_msg = " "*(idx_lim-decimals) + str_i + "  "

        w_msg += state + " "*(max_len-l_stt) + " \u2192 "
        o_msg += "\u21B3" + value + (max_len-l_val+2)*" "

    lines.append(w_msg.removesuffix(" \u2192 ") + "\n" + o_msg)
    return "\n".join(lines)



def logaddexp10(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Function equivalent to `np.logaddexp` in base 10.

    
    Args
    ----
    x1 : array_like
        The first array.
    
    x2 : array_like
        The second array.


    Returns
    -------
    result : ndarray
        The array of summed values in log-space.
    """

    max_val = np.maximum(x1, x2)
    
    max_val[np.isinf(max_val) & (max_val < 0)] = 0.0 
    return max_val + np.log10(10**(x1 - max_val) + 10**(x2 - max_val))

def logaddexp10_reduce(arr: np.ndarray, axis: int = None) -> np.ndarray | float:
    """Function equivalent to `np.logaddexp.reduce` in base 10.
    
    Trick to sum values in logarithmic space:

    log_10(a+b) = log_10(10^a + 10^b)

    To make every value rest between 0 and 1, 
    the biggest value is pulled out so that all
    exponents become negative:

    Let a > b

    log_10(10^a + 10^b) = a + log_10( 1 + 10^(b-a) )

    
    Args
    ----
    arr : array_like
        The array to perform the calculations in.

    axis : int, optional
        The axis along which to perform the calculations.

        
    Returns
    -------
    result : ndarray or float
        An array with the same shape as `arr`, with the specified axis removed. 
        If `arr` is a 0-d array, or if `axis` is None, a scalar is returned.
    """
    
    max_val = np.max(arr, axis=axis, keepdims=True) if axis is not None else np.max(arr)
    
    # se todas as probabilidades forem 0, max_val = -inf
    # o cálculo dos expoentes vai tentar fazer -inf - (-inf) = -inf + inf = NaN
    # neste caso como o cálculo em espaço linear é 0 - 0, 
    # sabemos que a indeterminação é igual a -inf 
    # então podemos simplesmente retornar um array com esse valor
    if np.all(np.isinf(max_val) & (max_val < 0)):
        if axis is None:
            return -np.inf
        else:
            return np.squeeze(np.full_like(max_val, -np.inf), axis=axis)
    
    if axis is not None:
        summed = np.sum(10 ** (arr - max_val), axis=axis)
        return np.squeeze(max_val, axis=axis) + np.log10(summed)
    else:
        return max_val + np.log10(np.sum(10 ** (arr - max_val)))
    

def parse_phobius_model(file = "phobius.model"):
    """Parses the phobius model into usable variables.

    
    Args
    ----
    file : str, optional
        File name where the model is. Defaults to "phobius.model".

        
    Returns
    -------
    out : tuple of 
        * **transition_matrix** : *Matrix2D* 
        * **emission_matrix** : *Emission2D*
        * **states** : *list of str*
        * **values** : *list of str*
        * **initial_distribution** : *Vector1D*
        * **state_labels** : *dict of {str: str}* 
        
        variables to be used in the definition of the `HiddenMarkovModel` and Viterbi decoding
    """
    
    states: dict[str, int] = {}
    values: dict[str, int] = {}

    temp_pi:   dict[str, str] = {}

    temp_tmat: dict[str, dict[str, str]] = {}
    temp_emat: dict[str, dict[str, str]] = {}

    state_labels: dict[str,str] = {}
    
    inside_header = False
    inside_pi = False
    inside_st = False
    inside_trans = False
    inside_em = False
    i_st = 0
    i_v = 0
    with open(file) as f:
        for line in f.readlines():

            line = line.strip()
            # ignorar comentários (só aparecem no início)
            if not line or line.startswith("#"):
                continue
            
            # ponto e vírgula ; indica separação entre propriedades dentro das chavetas {}
            # e algumas, especialmente as matrizes, ocupam mais de uma linha
            has_semicolon = line.endswith(";")
            line = line.replace(";","").replace("{","")

            tokens = line.split()
            # linhas vazias
            if not tokens:
                continue
            
            # header contem o alfabeto de aminoácidos e wildcards desse mesmo alfabeto
            # as wildcards não são usadas mas não custa incluí-las
            if tokens[0] == "header":
                inside_header = True
                continue

            if inside_header:
                for val in list(tokens[1]):
                    values[val] = i_v
                    i_v += 1
                    
                if tokens[0] == "wildcards":    # header acaba
                    inside_header = False
                    
            # Distribuição Inicial
            if tokens[0] == "begin":
                inside_pi = True
                continue

            if inside_pi:
                start_i = 1 if tokens[0] == "trans" else 0
                parts = "".join(tokens[start_i:]).split(":")
                temp_pi[parts[0]] = parts[1]
                
                if has_semicolon:
                    inside_pi = False
                continue

            # Estados Ocultos
            if len(tokens)==1 and tokens[0] not in ["begin", "header", "}"] and not (inside_st or inside_pi):
                cur_state = tokens[0]
                states[cur_state] = i_st
                i_st += 1
                inside_st = True
                temp_tmat[cur_state] = {}
                temp_emat[cur_state] = {}
                continue
            
            if inside_st:
                # identificação da localização (inside, membrane, outside, etc.)
                if tokens[0] == "label":
                    state_labels[cur_state] = tokens[1]
                    continue
                
                # probabilidades de ir para os outros estados, i.e.
                # linha da matriz de transição relativa ao estado
                if tokens[0] == "trans" or inside_trans:
                    inside_trans = True
                    start_i = 1 if tokens[0] == "trans" else 0
                    parts = "".join(tokens[start_i:]).split(":")

                    temp_tmat[cur_state][parts[0]] = parts[1]

                    if has_semicolon:
                        inside_trans = False
                        continue

                # probabilidades de emissão de cada aminoácido, i.e.
                # linha da matriz de emissão relativa ao estado
                if tokens[0] == "only" or inside_em:
                    inside_em = True
                    start_i = 1 if tokens[0] == "only" else 0
                    ems = ["".join(token).split(":") for token in tokens[start_i:]]
                    for em in ems:
                        temp_emat[cur_state][em[0]] = em[1]
                    
                    if has_semicolon:
                        inside_em = False
                        continue
                
                # significa que tem a mesma matriz de emissão que o estado indicado
                if tokens[0] == "tied_letter":
                    temp_emat[cur_state] = tokens[1]
                
                # fim do estado
                if tokens[0]== "}":
                    inside_st = False

    
    pi:   Vector1D   = np.zeros(len(states))
    tmat: Matrix2D   = np.zeros((len(states), len(states)))
    emat: Emission2D = np.zeros((len(states), len(values)))

    # numpy converte as strings automaticamente para float64
    for st, prob in temp_pi.items():
        pi[states[st]] = prob
    
    for st1, trans in temp_tmat.items():
        for st2, prob in trans.items():
            tmat[states[st1], states[st2]] = prob

    for st, emi in temp_emat.items():
        if isinstance(emi, str):
            emi = temp_emat[emi]

        for aa, prob in emi.items():
            emat[states[st], values[aa]] = prob
    return tmat, emat, list(states), list(values), pi, state_labels
