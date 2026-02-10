# GNN-based Optimization of sum-SE in Cell-Free Massive MIMO Communications using maximum ratio (MR) combining 

This project implements a framework  based on Graph Neural Networks (GNNs) for optimizing the Access Points (APs) cooperation cluster 
formation in cell-free massive MIMO systems using maximum ratio (MR) combining.

See details in the paper:
- **Title**: A GNN-Based Approach to AP Cooperation Cluster Formation in Cell-Free Massive MIMO
- **Authors**: Dariel Pereira Ruisánchez, Michael Joham, Óscar Fresnedo, Darian Pérez Adán, Luis Castedo, and Wolfgang Utschick
- **Presented at**: IEEE 101st Vehicular Technology Conference: VTC2025-Spring
---

## Project Structure

### Main Files and Scripts

- **`APassignment_heuristics_{}.py, {CDF,K}`**
  This files generate the results for the AP cooperation cluster formation optimization using different setups. The placeholders 
`{CDF, K}` indicate different configurations.

- **`_AP_SampleGenerating.py`**  
  Generates training samples for the GNN models. It creates datasets with graph-related information for the AP cooperation
cluster formation task.

- **`_AP_training.py`**  
  This script implements the training of a Graph Neural Network (GNN) to learn how to form AP cooperation clusters.

- **`_functionsGraphHandling.py`**  
  This module contains classes and functions for handling graph-based data structures. It includes implementations of
Graph Neural Network (GNN) models.

- **`functionsSetup_heuristic.py`**  
  Generates the system setup, including AP and UE positions, channel realizations, and other parameters.

- **`functionsAllocation_heuristics.py`**  
  Implements heuristic methods for pilot allocation and AP on/off configuration.

- **`functionsComputeSE_uplink.py`**  
  Computes the uplink SE for different receive combining schemes.

- **`functionsChannelEstimates.py`**  
  Generates the channel realizations and the channel estimates for a cell-free massive MIMO system.

- **`functionsUtils.py`**  
  Provides utility functions for loading results, saving data, and other general-purpose tasks.

- **`README.md`**  
  This file provides an overview of the project, its structure, and usage instructions.

---

### Output Folders

- **`AP_TRAININGDATA/`**  
  Stores the generated training datasets.

- **`GRAPHS_heuristics/`**  
  Contains graphs for heuristic methods used in the VTC25 conference paper.

## Libraries Used

The project relies on the following Python libraries:

- **`torch`**: For building and training GNN models.
- **`torch_geometric`**: For handling graph data and implementing graph-based neural networks.
- **`numpy`**: For numerical computations.
- **`matplotlib`**: For plotting and visualizing results.
- **`tqdm`**: For progress bars during data generation and training.
- **`random`**: For generating random setups and seeds.
- **`scikit-learn`**: For clustering (e.g., k-means) and other machine learning utilities.

---

## Recommended use
-  Generate training samples using AP_SampleGenerating.py.
-  Train the GNN models using _AP_training.py.
-  Evaluate the models using as comparison the heuristic methods in APassignment_heuristics_{} scripts for 
different network configurations.

---

## Versions:
    # 20241108: 
        # Added:
            # Include the method main_CDF.py that compute the CDF that generates the values
              to generate the CDF graphs for Optimal, DCC and ALL baselines.   
    # 20241109: 
        # Added:
            # Include the Graph_SEs_CDF.py script to generate the graphs for CDF and histogram of 
              number of serving APs. It loads only the values for the Optimal, DCC and ALL baselines. 
    
    # 20241111:
        # Added:
            # Review the main_SampleGenerating.py script to generate the samples for training the GNNs.
            # Include the script functionsGraphHandling.py to handle the buffer of training data, and
                the functions to convert the communication matrices into the graph-structured data.

    # 20241113:
        # Added:
            # Include the P_MMSE combining in the functionsComputeSE_uplink.py script, and included in the
                flow of the main_CDF.py and functionsAPAllocation.py scripts.

    # 20241122:
        # Added:
            # Include the ToyModel_iCSI.py script to study the behavior of small cell-free
                networks regarding AP assignment. It include the benchmarks: 
                    -"allUEs": the AP assignment of all UEs is updated at each time
                    instant by considering the information regarding all the remaining 
                    UEs and all the APs.
                    -"allUEs_local": the same as "allUEs" but only the information regarding
                    the best serving APs and the UEs served by these APs is considered.
                    -"newUE": at each time instant, only the AP assignment of the new UEs 
                    is updated.
                    -"newUE_local": the same as "newUE" but only the information regarding
                    the best serving APs and the UEs served by these APs is considered.
            # Include the folder ToyModelsData to store the results of the ToyModel_iCSI.py 
                script and generate the graphs.

    # 20241125:
        # Added:
            # Include the AP_SampleGenerating.py script to generate the samples for training the GNNs: 
                    -It generates random setups with a fixed number of APs and a random number of UEs that follows
                    a uniform distribution within a specified range.
                    -The samples are composed of a target UE and the I UEs which are relevant for the M APs relevant
                    for the target UE. For every UE we get a vector of features comprising the channel gains to the 
                    M APs.

    # 20241127:
        # Added:
            # Include the functions generateSetup_UnbalancedUE() in functionsSetup.py to create setups where the 
            distribution of the UEs within the area is unbalanced.
            # Include new functions in the script functionsGraphHandling.py:
                    -Include MyGraphDataset() class to create a pytorch dataset from the sample buffers created during
                    sample generation.
                    -Include get_star_graph() function that convert a sample into star graph with a feature matrix and 
                    an edge list.
            # Include the script AP_SampleGenerating_unbalancedSetup.py to test setups with unbalanced UE distribution.

    # 20241129:
        # Added:
            # The scripts functionsGraphHandling.py and AP_training.py were updated to include some elements for the 
            training of the GNNs.
                - Training should be revised because the loss is not decreasing.

    # 20250108:
        # Added:
            # Include the scripts ..._heuristics that implement the heuristic methods for the AP cooperation cluster 
            formation (ALL, BCC, M-best, Graph).
            # The folder GRAPHS_heuristics contains the graphs for the VTC25 conference paper.

        # To do:
        # Revise all the scripts starting with _ to adjust to the changes introduced in the scripts ending with 
        _heuristics.
        # Implement the methods for creating and training the GNNs (To be revised).

        # Key issues:
        # The AP assignment problem only makes sense when considering MR combining. For the case of MMSE and P-RZF 
        combining, the best thing that can be done to improve sum-rate is to serve all UEs with all APs.

    # Note_Commit:
        - Check that the dimensions of edge lists are right. The right dimensions are (2, num_edges) and not (num_edges, 2).

 

#
