from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.linalg as scyla # (sc)ip(y).(l)in(a)lg :D

from numpy.typing import ArrayLike
from my_types import *
from aux_functions import *

class MarkovChain[StateT=int]:

    tmat:               Matrix2D[np.float64]
    states:             list[StateT]
    nstates:            NStates
    
    stationary_state:   Vector1D[np.float64]
    initial_dist:       Vector1D[np.float64]

    def __init__(self, 
                 transition_matrix:     ArrayLike, 
                 states:                Optional[Sequence[StateT]]  = None, 
                 initial_distribution:  Optional[ArrayLike]         = None):

        self.tmat = np.array(transition_matrix)

        if self.tmat.shape[0] != self.tmat.shape[1]: 
            raise ValueError("transmission matrix is not square")
        
        if not (np.all(np.logical_or(self.tmat.sum(1).round(5) == 1, self.tmat.sum(1).round(5) == 0))):
            raise ValueError("outgoing probabilities do not add up to 1 or 0")
        
        self.nstates = self.tmat.shape[0]

        self.states = list(range(self.nstates)) if states is None else list(states)

        self.indexed_matrix = pd.DataFrame(self.tmat, self.states, self.states)

        self.stationary_state = self.calculate_pi("RepMatMul")
        self.indexed_stationary = pd.Series(self.stationary_state, self.states)

        self.initial_dist = self.stationary_state if initial_distribution is None else np.array(initial_distribution)
        self.indexed_init_dist = pd.Series(self.initial_dist, self.states)
            
    @overload
    def random_walk(self, 
                    steps: int, 
                    start: int | StateT | None = None,
                    *,
                    seed: int = None,
                    as_array: Literal[True]) -> NDArray: ...
    @overload
    def random_walk(self, 
                    steps: int, 
                    start: int | StateT | None = None,
                    *,
                    seed: int = None,
                    as_array: Literal[False] = False) -> None: ...
    
    def random_walk(self, 
                    steps: int, 
                    start: int | StateT | None = None,
                    *,
                    seed: int = None,
                    as_array: bool = False) -> None | NDArray:
        
        np.random.seed(seed)
        cur_state = start if start is not None else np.random.choice(self.states, p=self.initial_dist)
        if not isinstance(cur_state, int):
            cur_state = self.states.index(cur_state)

        walk=np.array([])
        for _ in range(steps):
            walk = np.append(walk, self.states[cur_state])
            cur_state = \
                np.random.choice(
                    np.arange(self.nstates), 
                    p=self.indexed_matrix.iloc[cur_state]
                )
        if as_array: 
            return walk
        
        print("\n\n".join(textwrap.wrap(" \u2192 ".join(walk))))
    

    def calculate_pi(self, 
                     method:        Literal["MonteCarlo", "RepMatMul", "LeftEigVec"], 
                     num_repeats:   int | None = None,
                     *,
                     seed: int = None) -> Vector1D[np.float64]:
        
        methods = ["MonteCarlo", "RepMatMul", "LeftEigVec"]
        if method not in methods: 
            raise NotImplementedError("method not implemented or invalid method name")
        
        if method == "MonteCarlo": #seria mais eficiente fazer durante o loop da random walk
            LARGE_NUMBER_TM = num_repeats if num_repeats is not None else 10**4
            big_walk = self.random_walk(LARGE_NUMBER_TM, seed=seed, as_array=True) # make array
            uniq_c = np.unique_counts(big_walk)
            counts_lookup = dict(zip(uniq_c.values, uniq_c.counts))
            ordered_counts = np.array([counts_lookup.get(state, 0) for state in self.states], dtype=np.float64)
            pi = ordered_counts / ordered_counts.sum()

        elif method == "RepMatMul":
            pi = self.tmat
            LARGE_NUMBER_TM = num_repeats if num_repeats is not None else 10
            for _ in range(LARGE_NUMBER_TM):
                pi = pi @ pi
            pi=pi[0]

        elif method == "LeftEigVec":
            # valores e vetores próprios (reais e complexos)
            val, left = scyla.eig(self.tmat, left=True, right=False)
    
            val_real = np.array(list(map(lambda n: round(n, 13), val.real)))
            val_imag = np.array(list(map(lambda n: round(n, 13), val.imag)))
            index = np.arange(len(val))[np.logical_and(val_real==1, val_imag==0)]
            pi = left[:,index].real.T[0]
            pi /= pi.sum()

        return pi
    

class HiddenMarkovModel[StateT=int, ValueT=int]:

    tmat:               Matrix2D[np.float64]
    states:             list[StateT]
    nstates:            NStates

    emat:               Emission2D[np.float64]
    values:             list[ValueT]
    nvalues:            NValues

    stationary_state:   Vector1D[np.float64]
    initial_dist:       Vector1D[np.float64]
    
    def __init__(self, 
                 chain_or_transition_matrix:    Optional[MarkovChain[StateT] | ArrayLike]   = None, 
                 emission_matrix:               Optional[Emission2D[np.float64]]            = None,
                 *,
                 states:                        Optional[Sequence[StateT]]                  = None, 
                 values:                        Optional[Sequence[ValueT]]                  = None, 
                 initial_distribution:          Optional[ArrayLike]                         = None,
                 seed = None):
        
        self.tmat = None
        self.emat = None
        random_params = False

        if chain_or_transition_matrix is None and emission_matrix is None:
            if not values is None and states is None:
                raise ValueError("É necessário fornecer no mínimo\nou as matrizes de probabilidade\nou os estados e valores")
            
            random_params = True
        
        if random_params:
            self.states = states
            self.values = values

            self.nstates = len(self.states)
            self.nvalues = len(self.values)

            self.tmat = random_sum_mat(self.nstates, self.nstates, seed)
            self.emat = random_sum_mat(self.nstates, self.nvalues, seed)
        
        if isinstance(chain_or_transition_matrix, MarkovChain):
            self.chain = chain_or_transition_matrix
        else:
            if not self.tmat: self.tmat = chain_or_transition_matrix
            self.chain = MarkovChain(self.tmat, states, initial_distribution)

        self.tmat = self.chain.tmat
        self.indexed_tmat = self.chain.indexed_matrix

        self.stationary_state = self.chain.stationary_state
        self.indexed_stationary = self.chain.indexed_stationary

        self.initial_dist = self.chain.initial_dist
        self.indexed_init_dist = self.chain.indexed_init_dist
        
        if not self.emat: self.emat = emission_matrix
        
        self.emat = np.array(self.emat)

        if not (np.all(np.logical_or(self.emat.sum(1).round(5) == 1, self.emat.sum(1).round(5) == 0))):
            raise ValueError("outgoing probabilities do not add up to 1 or 0")
        
        self.nstates = self.chain.nstates
        self.nvalues = self.emat.shape[1]

        self.states = self.chain.states
        self.values = list(values) or list(range(self.nvalues))

        self.indexed_emat = pd.DataFrame(self.emat, self.states, self.values)

        eps = 1e-300    # evitar log(0)
        self.log_pi:     Vector1D[np.float64]   = np.log10(self.initial_dist + eps)
        self.log_tmat:   Matrix2D[np.float64]   = np.log10(self.tmat + eps)
        self.log_emat:   Emission2D[np.float64] = np.log10(self.emat + eps)


    def calculate_probability(self, states: SeqInput[StateT], observed: SeqInput[ValueT], *, log = True) -> float:
        observed = names_to_indexes(observed, self.values)
        states = names_to_indexes(states, self.states)

        if log:
            p =       self.log_pi[states[0]] + self.log_emat[states[0],observed[0]]
        else:
            p = self.initial_dist[states[0]] *     self.emat[states[0],observed[0]]

        for t in range(1,len(states)):
            if log:
                p += self.log_tmat[states[t-1], states[t]] + self.log_emat[states[t], observed[t]]
            else:
                p *=     self.tmat[states[t-1], states[t]] *     self.emat[states[t], observed[t]]
        
        return p
    
    @overload
    def random_walk(self, 
                    steps: int, 
                    start: int | StateT | None = None,
                    *,
                    seed:       int = None,
                    emit:       Literal[False],
                    as_arrays:  Literal[True]           ) -> tuple[NDArray, NDArray]: ...
    @overload  
    def random_walk(self, 
                    steps: int, 
                    start: int | StateT | None = None,
                    *,
                    seed:       int = None,
                    emit:       Literal[True] = True,
                    as_arrays:  Literal[True]           ) -> NDArray: ...
    @overload  
    def random_walk(self, 
                    steps: int, 
                    start: int | StateT | None = None,
                    *,
                    seed:       int = None,
                    emit:       bool = True,
                    as_arrays:  Literal[False] = False  ) -> None: ...
    
    def random_walk(self, 
                    steps: int, 
                    start: int | StateT | None = None,
                    *,
                    seed:       int = None,
                    emit:       bool = True,
                    as_arrays:  bool = False            ) -> Union[tuple[NDArray, NDArray], NDArray, None]:
        
        if not emit:
            return self.chain.random_walk(steps, start, seed=seed, as_array=as_arrays)
        
        walk = self.chain.random_walk(steps, start, seed=seed, as_array=True)
        observations = self.random_emissions(walk, seed=seed)
        if as_arrays:
            return walk, observations
        
        print(seqobs_pretty_print(walk, observations))

    
    def random_emissions(self, hidden_sequence: Sequence[StateT], *, seed: int = None) -> NDArray:
        np.random.seed(seed)
        hidden_sequence = names_to_indexes(hidden_sequence, self.states)
        observations = np.array([])
        for state in hidden_sequence:
            obs = \
                np.random.choice(
                    np.arange(self.nvalues), 
                    p=self.indexed_emat.iloc[state]
                )
            observations = np.append(observations, self.values[obs])

        return observations


    @overload
    def forward_algorithm(self, observed: SeqInput[ValueT], *, log = True, use_alfas: Literal[False] = False) -> np.float64:...
    @overload
    def forward_algorithm(self, observed: SeqInput[ValueT], *, log = True, use_alfas: Literal[True] = False) -> AlgArray2D[np.float64]:...

    def forward_algorithm(self, observed: SeqInput[ValueT], *, log = True, use_alfas = False) -> AlgArray2D[np.float64] | np.float64:
        T: Time = len(observed)
        #transformar valores em indices
        observed = names_to_indexes(observed, self.values)

        alfas: AlgArray2D[np.float64] = np.zeros((T, self.nstates))

        if log:
            alfas[0] =       self.log_pi + self.log_emat[:, observed[0]]
        else:
            alfas[0] = self.initial_dist *     self.emat[:, observed[0]]
        
        for t in range(T-1):
            for j in range(self.nstates):
                if log:
                    alfas[t+1,j] = logaddexp10_reduce(alfas[t] + self.log_tmat[:, j]) + self.log_emat[j, observed[t+1]]
                else:
                    alfas[t+1,j] =             np.sum(alfas[t] *     self.tmat[:, j]) *     self.emat[j, observed[t+1]]


        if use_alfas: return alfas 
        
        else:
            if log: total_probability = logaddexp10_reduce(alfas[-1])
            else:   total_probability =             np.sum(alfas[-1])
        
        return total_probability
    
    @overload
    def backward_algorithm(self, observed: SeqInput[ValueT], *, log = True,  use_betas: Literal[False] = False) -> np.float64:...
    @overload
    def backward_algorithm(self, observed: SeqInput[ValueT], *, log = True,  use_betas: Literal[True] = False) -> AlgArray2D[np.float64]:...

    def backward_algorithm(self, observed: SeqInput[ValueT], *, log = True,  use_betas = False) -> AlgArray2D[np.float64] | np.float64:
        T: Time = len(observed)
        
        observed = names_to_indexes(observed, self.values)

        betas: AlgArray2D[np.float64] = np.zeros((T, self.nstates))
        if not log:
            betas[T-1, :] = 1.0
        
        for t in range(T-2, -1, -1):
            for i in range(self.nstates):
                if log:
                    betas[t, i] = logaddexp10_reduce(self.log_tmat[i, :] + self.log_emat[:, observed[t+1]] + betas[t+1, :])
                else:
                    betas[t, i] =             np.sum(    self.tmat[i, :] *     self.emat[:, observed[t+1]] * betas[t+1, :])
        
        if use_betas: return betas
        else:
            if log: total_probability = logaddexp10_reduce(      self.log_pi + self.log_emat[:, observed[0]] + betas[0, :])
            else:   total_probability =             np.sum(self.initial_dist *     self.emat[:, observed[0]] * betas[0, :])

        return total_probability
    

    # matrizes e resultados
    @overload   # matrizes e resultados com labels; retornar os estados
    def viterbi_algorithm(self, 
                          observed:         SeqInput[ValueT],
                          state_labels:     dict[StateT, LabelT], 
                          *, 
                          log:              bool = True,
                          use_results:      Literal[True],
                          return_original:  Literal[True],
                          use_matrices:     Literal[True]               ) -> tuple[np.float64, list[StateT], AlgArray2D[np.float64], AlgArray2D[int]]: ...
    
    @overload   # matrizes e resultados com labels
    def viterbi_algorithm(self, 
                          observed:         SeqInput[ValueT],
                          state_labels:     dict[StateT, LabelT], 
                          *, 
                          log:              bool = True,
                          use_results:      Literal[True],
                          return_original:  Literal[False] = False,
                          use_matrices:     Literal[True]               ) -> tuple[np.float64, list[LabelT], AlgArray2D[np.float64], AlgArray2D[int]]: ...
    
    @overload    # matrizes e resultados sem labels
    def viterbi_algorithm(self, 
                          observed:         SeqInput[ValueT],
                          state_labels:     None = None, 
                          *, 
                          log:              bool = True,
                          use_results:      Literal[True],
                          return_original:  bool = False,
                          use_matrices:     Literal[True]               ) -> tuple[np.float64, list[StateT], AlgArray2D[np.float64], AlgArray2D[int]]: ...
    
    # só resultados
    @overload   # resultados com labels; retornar os estados
    def viterbi_algorithm(self, 
                          observed:         SeqInput[ValueT],
                          state_labels:     dict[StateT, LabelT], 
                          *, 
                          log:              bool = True,
                          use_results:      Literal[True],
                          return_original:  Literal[True],
                          use_matrices:     Literal[False] = False      ) -> tuple[np.float64, list[StateT]]: ...
    
    @overload   # resultados com labels
    def viterbi_algorithm(self, 
                          observed:         SeqInput[ValueT],
                          state_labels:     dict[StateT, LabelT], 
                          *, 
                          log:              bool = True,
                          use_results:      Literal[True],
                          return_original:  Literal[False] = False,
                          use_matrices:     Literal[False] = False      ) -> tuple[np.float64, list[LabelT]]: ...
    
    @overload   # resultados sem labels
    def viterbi_algorithm(self, 
                          observed:         SeqInput[ValueT],
                          state_labels:     None = None, 
                          *, 
                          log:              bool = True,
                          use_results:      Literal[True],
                          return_original:  bool = False,
                          use_matrices:     Literal[False] = False      ) -> tuple[np.float64, list[StateT]]: ...
    
    # só matrizes
    @overload
    def viterbi_algorithm(self, 
                          observed:         SeqInput[ValueT],
                          state_labels:     Optional[dict[StateT, LabelT]] = None, 
                          *, 
                          log:              bool = True,
                          use_results:      Literal[False] = False,
                          return_original:  bool = False,
                          use_matrices:     Literal[True]               ) -> tuple[AlgArray2D[np.float64], AlgArray2D[int]]: ...
    
    # string dos resultados
    @overload   
    def viterbi_algorithm(self, 
                          observed:         SeqInput[ValueT],
                          state_labels:     Optional[dict[StateT, LabelT]] = None, 
                          *, 
                          log:              bool = True,
                          use_results:      Literal[False] = False,
                          return_original:  bool = False,
                          use_matrices:     Literal[False] = False,     ) -> None: ...
    
    
    def viterbi_algorithm(self, 
                          observed:         SeqInput[ValueT], 
                          state_labels:     Optional[dict[StateT, LabelT]] = None, 
                          *, 
                          log:              bool = True,
                          use_results:      bool = False,
                          return_original:  bool = False,
                          use_matrices:     bool = False
                          ) -> Union[
                                None,
                                tuple[AlgArray2D[np.float64], AlgArray2D[int]],
                                tuple[np.float64, list[LabelT] | list[StateT]],
                                tuple[np.float64, list[LabelT] | list[StateT], AlgArray2D[np.float64], AlgArray2D[int]]
                                ]:
        
        T = len(observed)
        observed = names_to_indexes(observed, self.values)

        deltas: AlgArray2D[np.float64]  = np.zeros((T, self.nstates))
        psis:   AlgArray2D[int]         = np.zeros((T, self.nstates), dtype=int)

        if log:
            deltas[0, :] = self.log_pi + self.log_emat[:, observed[0]]
        else:
            deltas[0, :] = self.initial_dist * self.emat[:, observed[0]]

        for t in range(1, T):
            for i in range(self.nstates):
                if log:
                    probs = deltas[t-1] + self.log_tmat[:,i] + self.log_emat[i, observed[t]]
                else:
                    probs = deltas[t-1] *     self.tmat[:,i] *     self.emat[i, observed[t]]

                psi = np.argmax(probs)
                delta = probs[psi]

                psis[t,i] = psi
                deltas[t,i] = delta

        retvals = []

        if use_matrices:
            if not use_results:
                return deltas, psis
            retvals.extend([deltas, psis])


        # Traceback
        x_star_T = np.argmax(deltas[-1,:])
        p_star = deltas[-1,x_star_T]

        X_star = np.array([-1 for _ in range(T)])
        X_star[-1] = x_star_T
        
        for t in range(T-1,0,-1):
            X_star[t-1] = psis[t,X_star[t]]

        X_star = indexes_to_names(X_star, self.states)


        if state_labels is not None and not return_original:
            labeled_X_star = list(map(lambda st: state_labels[st], X_star))
            return_sequence = labeled_X_star
        else:
            return_sequence = X_star
        
        if use_results:
            retvals = [p_star, return_sequence] + retvals
        
        if use_matrices or use_results:
            return tuple(retvals)


        logP = "Logaritmo da p" if log else "P"
        msg = f"{logP}robablilidade: {p_star}\n\n"

        msg += seqobs_pretty_print(return_sequence, indexes_to_names(observed, self.values))
        print(msg)


    def _baum_welch_helper(self, sequence: SeqInput[ValueT], *, log = True):
        T: Time = len(sequence)
        sequence = names_to_indexes(sequence, self.values)

        alfas = self.forward_algorithm(sequence, log=log, use_alfas=True)
        betas = self.backward_algorithm(sequence, log=log, use_betas=True)
        
        gamas: AlgArray2D[np.float64] = np.zeros((T, self.nstates))
        csis:  AlgArray3D = np.zeros((T, self.nstates, self.nstates))

        if log:
            P_seq_given_theta = logaddexp10_reduce(alfas[0] + betas[0])
        else:
            P_seq_given_theta =             np.sum(alfas[0] * betas[0])

        if P_seq_given_theta == 0 and not log:
            P_seq_given_theta = 1

        for t in range(T):
            if log:
                gamas[t] = (alfas[t] + betas[t]) - P_seq_given_theta
            else:
                gamas[t] = (alfas[t] * betas[t]) / P_seq_given_theta

            if t == T-1: continue
            for i in range(self.nstates):
                for j in range(self.nstates):
                    if log:
                        csis[t,i,j] = (alfas[t,i] + self.log_tmat[i,j] + betas[t+1,j] + self.log_emat[j,sequence[t+1]]) - P_seq_given_theta
                    else:
                        csis[t,i,j] = (alfas[t,i] *     self.tmat[i,j] * betas[t+1,j] *     self.emat[j,sequence[t+1]]) / P_seq_given_theta

        return gamas, csis
    
    def _baum_welch_algorithm_step(self, sequences: list[SeqInput[ValueT]], *, log = True, verbose = 0) -> tuple[Matrix2D[np.float64], Emission2D[np.float64], Vector1D[np.float64]]:
        all_gamas: list[AlgArray2D[np.float64] ] = []
        all_csis:  list[AlgArray3D] = []
        nseqs = len(sequences)
        sequences = list(sequences)

        for r, seq in enumerate(sequences):
            seq = names_to_indexes(seq, self.values)
            sequences[r] = seq
            seq_gamas, seq_csis = self._baum_welch_helper(seq, log=log)
            
            if verbose>1 and r%100 == 0: print(f"seq {r}: gamas e csis calculados")
            
            all_gamas.append(seq_gamas)
            all_csis.append(seq_csis)

        if verbose>1: print(f"todos os gamas e csis calculados")

        # update dos parâmetros
        if log:
            new_log_pi: Vector1D[np.float64] = np.full((self.nstates), -np.inf)

            new_log_tmat_num: Matrix2D[np.float64] = np.full((self.nstates, self.nstates), -np.inf)
            new_log_tmat_den: Matrix2D[np.float64] = np.full((self.nstates, self.nstates), -np.inf)

            new_log_emat_num: Emission2D[np.float64] = np.full((self.nstates, self.nvalues), -np.inf)
            new_log_emat_den: Emission2D[np.float64] = np.full((self.nstates, self.nvalues), -np.inf)

            for r, seq in enumerate(sequences):
                T = len(seq)

                new_log_pi = logaddexp10(new_log_pi, all_gamas[r][0,:])

                new_log_tmat_num = logaddexp10(new_log_tmat_num, logaddexp10_reduce(all_csis[r][:T-1], 0))
                new_log_tmat_den = logaddexp10(new_log_tmat_den, logaddexp10_reduce(all_gamas[r][:T-1], 0).reshape(-1,1))

                for j in range(self.nvalues):
                    mask = (seq==j)
                    if np.any(mask):
                        new_log_emat_num[:,j] = logaddexp10(new_log_emat_num[:,j], logaddexp10_reduce(all_gamas[r][mask], 0))
                    new_log_emat_den[:,j] = logaddexp10(new_log_emat_den[:,j], logaddexp10_reduce(all_gamas[r], 0))

            new_log_pi -= np.log10(nseqs)
            new_log_tmat = new_log_tmat_num - new_log_tmat_den
            new_log_emat = new_log_emat_num - new_log_emat_den

            return 10**new_log_tmat, 10**new_log_emat, 10**new_log_pi

        else:
            new_pi: Vector1D[np.float64] = np.zeros(self.nstates)

            new_tmat_num: Matrix2D[np.float64] = np.zeros((self.nstates, self.nstates))
            new_tmat_den: Matrix2D[np.float64] = np.zeros((self.nstates, self.nstates))

            new_emat_num: Emission2D[np.float64] = np.zeros((self.nstates, self.nvalues))
            new_emat_den: Emission2D[np.float64] = np.zeros((self.nstates, self.nvalues))

            for r, seq in enumerate(sequences):
                T = len(seq)

                new_pi += all_gamas[r][0,:]

                new_tmat_num += np.sum(all_csis[r][:T-1], 0)
                new_tmat_den += np.sum(all_gamas[r][:T-1], 0).reshape(-1,1)

                for j in range(self.nvalues):
                    new_emat_num[:,j] += np.sum(all_gamas[r][seq==j], 0)
                    new_emat_den[:,j] += (np.sum(all_gamas[r],0))

            new_pi /= nseqs
            new_tmat = new_tmat_num/new_tmat_den
            new_emat = new_emat_num/new_emat_den

            return new_tmat, new_emat, new_pi
    
    def baum_welch_algorithm(self, sequences: list[SeqInput[ValueT]], *, convergence_limit: float = 1e-10, log = True, verbose = 0):
        converged = False
        i=0
        while not converged:
            if verbose>0: print(f"loop {i}")
            new_tmat, new_emat, new_pi = self._baum_welch_algorithm_step(sequences, log=log, verbose=verbose)
            change_A = np.mean(np.abs(self.tmat-new_tmat))
            change_B = np.mean(np.abs(self.emat-new_emat))
            change_p = np.mean(np.abs(self.initial_dist-new_pi))
            converged = (change_A + change_B + change_p) < convergence_limit

            self.tmat = new_tmat
            self.emat = new_emat
            self.initial_dist = new_pi
            if i > 20:
                raise Exception("Os parâmetros não convergiram")
        
        # re-calcular distribuição estacionária
        self.__init__(self.tmat, self.emat, states=self.states, values=self.values, initial_distribution=self.initial_dist)
        return ("HMM treinado com sucesso, parâmetros finais:\n\n"
              f"Matriz de transmissão\n{self.indexed_tmat.round(5)}\n\n"
              f"Matriz de emissão\n{self.indexed_emat.round(5)}\n\n"
              f"estado inicial\n{self.indexed_init_dist.round(5)}\n\n"
              f"estado estacionário\n{self.indexed_stationary.round(5)}\n\n"
        )
