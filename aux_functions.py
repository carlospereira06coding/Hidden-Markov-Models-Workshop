from __future__ import annotations

import textwrap
import numpy as np
from numpy.typing import NDArray
from my_types import *

def random_sum(n, s=1):
    array = []
    lim = s
    for _ in range(n-1):
        v = np.random.random()*lim
        array.append(v)
        lim -= v
    array.append(s-np.sum(array))
    return array

def random_sum_mat(n, m=None, seed=None):
    np.random.seed(seed)
    m = m or n
    mat=[]
    for _ in range(n):
        mat.append(random_sum(m))
    return mat


def names_to_indexes(name_sequence: SeqInput[V], index_map: list[V]) -> NDArray[np.int64]:
    if isinstance(name_sequence[0], (int, np.int64)): return np.array(name_sequence)
    return np.array(list(map(lambda val: index_map.index(val), name_sequence)))

def indexes_to_names(index_sequence: Sequence[int], name_map: list[V]) -> list[V]:
    return list(map(lambda val: name_map[val], index_sequence))


def seqobs_pretty_print(seq: Sequence, obs: Sequence) -> str:
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
    """função equivalente a `np.logaddexp` em base 10"""
    max_val = np.maximum(x1, x2)
    
    max_val[np.isinf(max_val) & (max_val < 0)] = 0.0 
    return max_val + np.log10(10**(x1 - max_val) + 10**(x2 - max_val))

def logaddexp10_reduce(arr: np.ndarray, axis: int = None) -> np.ndarray | float:
    """
    **função equivalente a `np.logaddexp.reduce` em base 10**
    
    truque para somar valores em espaço logarítmico:

    log_10(a+b) = log_10(10^a + 10^b)

    para que todos os valores fiquem entre 0 e 1, 
    passa-se o maior valor para fora de modo a que 
    os expoentes fiquem todos negativos:

    Seja a > b

    log_10(10^a + 10^b) = a + log_10( 1 + 10^(b-a) )
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