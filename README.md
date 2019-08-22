# Personalized Multitask Learning
This repo contains code for 3 multitask machine learning methods: deep neural networks, Multitask Multi-kernel Learning (MTMKL), and a hierarchical Bayesian model (HBLR). These methods can be used to personalize the prediction of outcomes like stress, happiness, etc. to each individual, by treating predicting the outcome of a single individual (or a cluster of related individuals) as a task. 

The code is related to two research papers which explain this approach in further detail: 

Taylor, S.\*, Jaques, N.\*, Nosakhare, E., Sano, A., Picard, R., <strong>"Personalized Multitask Learning for Predicting Tomorrow’s Mood, Stress, and Health"</strong>, IEEE Transactions on Affective Computing December 2017. <small>(\*equal contribution)</small> <a href="https://affect.media.mit.edu/pdfs/17.TaylorJaques-PredictingTomorrowsMoods.pdf">PDF</a>

Jaques, N.\*, Taylor, S.\*, Nosakhare, E., Sano, A., Picard, R., <strong>"Multi-task Learning for Predicting Health, Stress, and Happiness", </strong> NIPS Workshop on Machine Learning for Healthcare, December 2016, Barcelona, Spain. <small>(\*equal contribution)</small> <a href="http://affect.media.mit.edu/pdfs/16.Jaques-Taylor-et-al-PredictingHealthStressHappiness.pdf">PDF</a> <strong>*BEST PAPER AWARD*</strong><br/>

<strong>If you find this code useful, please cite our work!</strong>

If you have any questions about this code or the associated papers, please email us at jaquesn@mit.edu or sataylor@mit.edu. 

# Models in this code:

## Multitask Neural Network (MTL-NN)

![image](mtl_nn_clusters.png)

The intuition behind the multitask neural network design is that the shared layers will learn to extract information 
that is useful for summarizing relevant characteristics of any person’s day into an efficient, generalizable embedding. 
The final, task-specific layers are then expected to learn how to map this embedding to a prediction customized for each person or cluster of people.

For example, if the shared layers learn to condense all of the relevant smartphone app data about phone calls and 
texting into an aggregate measure of social support, the task-specific layers can then learn a unique weighting of this 
measure for each cluster of participants. Perhaps a cluster containing participants with high extroversion scores will 
be more strongly affected by a lack of social support than another cluster.

## Multitask Multi-kernel Learning (MTMKL)

MTMKL (originally developed by <a href="https://www.sciencedirect.com/science/article/pii/S0925231214005025">Kandemir 
et. al.</a>) is a modified version of Multi-Kernel Learning (MKL) in which tasks 
share information through kernel weights on the modalities.  MTMKL uses a least-squares support vector machine (LSSVM) 
for each task-specific model. Unlike the canonical SVM, the LSSVM uses a quadratic error on the “slack” variables 
instead of an L1 error. As a result, the LSSVM can be learned by solving a series of linear equations in contrast to 
using quadratic programing to learn a canonical SVM model.


## Hierarchical Bayesian Logistic Regression (HBLR)

In hierarchical Bayesian MTL approaches, the model for each task draws its parameters from a common prior distribution. 
As the model is trained, the common prior is updated, allowing information to be shared across tasks. The model we 
adopt, which was originally proposed by <a href="http://www.jmlr.org/papers/v8/xue07a.html">Xue et. al.</a>, draws logistic regression (LR) weights for each task 
from a shared Dirichlet Process (DP) prior; we call this model Hierarchical Bayesian Logistic Regression (HBLR).

In contrast with our prior approaches (MTL-NN and MTMKL), the HBLR model allows us to directly define each task as 
predicting a label (e.g. tomorrow's stress level) of a single user, since the model is able to implicitly learn its 
own (soft) clustering. This model clusters tasks that are most similar in terms of their relationship between the 
input features and their resulting outcome (i.e. decision boundaries) while simultaneously learning the prediction 
function.

## Single Task Learning models
Code to train a logistic regression model, an LSSVM, and a single-task neural network is include for comparison purposes.

# Structure

## Code structure
Wrappers are used to perform a grid search over hyperparameters. The file `run_jobs.py` can be used to launch the training of several models in sequence, and send emails after they complete. To see an example of how to run the training code for the models, see `jobs_to_run.txt`. 

## Input data format
### .csv files
Assume csvs have columns for 'user_id', 'timestamp', and columns for the outcome labels containing the string '_Label'.

### 'Task dict list' 
For the multi-task algorithms, we use a special data structure saved to a pickle file to represent the data from multiple tasks. The code for generating files in this format given a .csv file is available in make_datasets.py. To run it, use:

```python make_datasets.py --datafile='./example_data.csv' --task_type='users'```

#### File Format details
- Data for both labels-as-tasks and users-as-tasks are stored in pickled files as a list of dicts (each list item represents a task)
    - Labels-as-tasks
        - The .csv file will be partitioned such that predicting related outcomes is each task (e.g. predicting stress is one task and predicting happiness is another)
        - Normalization is done based on training data for entire group
	- Users-as-tasks:
        - The .csv file will be partioned such that predicting the outcome of each user is one task.
        - Need to specify which label to target (i.e., the label that you will be predicting)
        - Normalization is done per-person
        
- Each task is a dict containing 4 keys:
    - ‘Name’: gives the name of the task, eg. "Group_Happiness_Evening_Label" or a user ID
    - ‘X’: the data matrix. Rows are samples, columns are features. Does not contain unnecessary stuff like ‘user_id’ and ‘timestamp’, and has already been normalized and empty cells filled
    - ‘Y’: the classification labels for this task, in the same order as the rows of X
    - ‘ModalityDict’: used for MTMKL model. Maps modalities like “phys” or “location” to their start index in the feature list 

