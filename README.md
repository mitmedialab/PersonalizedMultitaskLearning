# Personalized Multitask Learning
Code for performing 3 multitask machine learning methods: deep neural networks, Multitask Multi-kernel Learning (MTMKL), and a hierarchical Bayesian model (HBLR). 

If you have any questions about this code or the associated papers, please email us at jaquesn@mit.edu or sataylor@mit.edu. 

## Please cite our work!

Taylor, S.\*, Jaques, N.\*, Nosakhare, E., Sano, A., Picard, R., <strong> "Personalized Multitask Learning for Predicting Tomorrow’s Mood, Stress, and Health"</strong>, IEEE Transactions on Affective Computing December 2017. <small>(\*equal contribution)</small> <a href="https://affect.media.mit.edu/pdfs/17.TaylorJaques-PredictingTomorrowsMoods.pdf">PDF</a>

Jaques N.\*, Taylor S.\*, Nosakhare E., Sano A., Picard R., <strong>"Multi-task Learning for Predicting Health, Stress, and Happiness", </strong> NIPS Workshop on Machine Learning for Healthcare, December 2016, Barcelona, Spain. <small>(\*equal contribution)</small> <a href="http://affect.media.mit.edu/pdfs/16.Jaques-Taylor-et-al-PredictingHealthStressHappiness.pdf">PDF</a> <strong>*BEST PAPER AWARD*</strong><br/>


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


