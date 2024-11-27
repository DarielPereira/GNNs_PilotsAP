# GNNs optimization of the AP and Pilot allocation in Cell-Free MIMO Systems

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

        # To do:
        # Work on the script for generating samples for training the GNNs (Done in 20241125).
        # Implement the methods for creating and training the GNNs (Incomplete in 20241127).

        # Key issues:
        # The AP assignment problem only makes sense when considering MR combining. For the case of MMSE and P-RZF 
        combining, the best thing that can be done to improve sum-rate is to serve all UEs with all APs.


 

## Getting Started

Download links:

SSH clone URL: ssh://git@git.jetbrains.space/gtec/drl-sch/Cell-Free.git

HTTPS clone URL: https://git.jetbrains.space/gtec/drl-sch/Cell-Free.git



These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites

What things you need to install the software and how to install them.

```
Examples
```

## Deployment

Add additional notes about how to deploy this on a production system.

## Resources

Add links to external resources for this project, such as CI server, bug tracker, etc.
