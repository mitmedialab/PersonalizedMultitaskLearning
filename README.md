# PersonalizedMultitaskLearning
Code for performing 3 multitask machine learning methods: deep neural networks, Multitask Multi-kernel Learning (MTMKL), and a hierarchical Bayesian model (HBLR). 

## This code is still under construction!
Please note that this code is fresh from our research proejct. While all the functionality is here, we are still working cleaning up some aspects and providing proper documentation. We plan to complete this process before the publication of the associated paper. In the mean time, if you have any questions please email us at jaquesn@mit.edu or sataylor@mit.edu. 

## Please cite our work!
The journal paper associated with this code is currently under revision. For now, you can cite our earlier workshop paper: <br />

Jaques N.*, Taylor S.*, Nosakhare E., Sano A., Picard R.,<strong>"Multi-task Learning for Predicting Health, Stress, and Happiness", </strong> NIPS Workshop on Machine Learning for Healthcare, December 2016, Barcelona, Spain. <small>(*equal contribution) </small> <a href="http://affect.media.mit.edu/pdfs/16.Jaques-Taylor-et-al-PredictingHealthStressHappiness.pdf">PDF</a> <strong>*BEST PAPER AWARD*</strong><br/>


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


