# PersonalizedMultitaskLearning
Code for performing 3 multitask machine learning methods: deep neural networks, Multitask Multi-kernel Learning (MTMKL), and a hierarchical Bayesian model (HBLR). 

## This code is still under construction!
Please note that this code is fresh from our research proejct. While all the functionality is here, we are still working cleaning up some aspects and providing proper documentation. We plan to complete this process before the publication of the paper mentioned above. In the mean time, if you have any questions please email us at jaquesn@mit.edu or sataylor@mit.edu. 

## Please cite our work!
Here's the paper: 

# Models in this code:

## Multitask Neural Network 

## Multitask Multi-kernel Learning (MTMKL)

## Hierarchical Bayesian Logistic Regression (HBLR)

# Structure

## Code structure
Wrappers are used to perform a grid search over hyperparameters

## Input data format
### .csv files
Assume csvs have columns for 'user_id', 'timestamp'

### 'Task dict list' 
For the multi-task algorithms, we use a special data structure saved to a pickle file to represent the data from multiple tasks. 

~~ Describe ~~

The code for generating files in this format give a .csv file is available in make_datasets.py


