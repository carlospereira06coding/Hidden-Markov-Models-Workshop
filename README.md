# **Research Workshop: Hidden Markov Models for Drug Design (Prediction of Transmembrane Protein Topology)**
Welcome! This workshop was developed by 4 bioinformatics students from the 2nd year of the Bachelor's in Bioinformatics at FCUP, University of Porto, for the course Algorithms for Biological Sequence Analysis.

With this project, our objective was to explore the application of Hidden Markov Models in the prediction of the topology of transmembrane proteins, with the motivation of developing a new drug. In this scenario, we chose to explore how HIV infects our immune system by binding to the co-receptor CCR5 on T-cells, which is a transmembrane protein.

This idea was inspired by the articles in the folder "HIV e Maraviroc". We also explored other applications of HMMs in bioinformatics and outside of this field (an article about bees).

## Objective: Prediction of the extracellular domain of CCR5 HIV co-receptor

We want to develop a new drug that binds to the extracellular terminal of CCR5. By using HMMs and the Viterbi algorithm, we can predict and see which zone/part of the amino acid (a.a.) sequence is more likely to be outside the cell and bind to the HIV virus. This new drug would be an antagonist of CCR5 because it would bind to this co-receptor and block HIV from entering and infecting the cell.

## Methodology and Implementations
Our objective is to use HMMs to predict the topology of CCR5, with the hidden states being the localization of each amino acid in the cell: Inside the cell (I), Membrane (M), and Outside the cell (O). We used a simplified model to explore the concept of HMMs, but also implemented a more complex model that is more biologically accurate (Gabriel arrasa). These models and classes are in the file hands_on. With this implementation, we can predict the most likely sequence of hidden states of our protein.

To be more interactive, we produced a document with a brief explanation of the concepts and some exercises to calculate the hidden states for a small sequence of a.a., implementing the simplified HMM with the 3 states, as well as some general questions. We also made a version of this document with the solutions to the questions and the steps to calculate everything correctly.

Finally, we wanted to compare the results of our hands-on with the output of HMMTOP, which is a tool available online to predict the topology of transmembrane proteins.

With the results of this prediction, we identified correctly the binding site of HIV in this protein, and the next step is to develop the drug that will bind here and block the entrance of HIV into the immune system. These results can be found in the last slide of our presentation.

### To run the hands-on please install the environment through this command
conda env create -f HMM_ENV.yaml

We hope you enjoy learning more about this application of HMMs! :)
