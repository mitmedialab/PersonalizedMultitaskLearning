# Personalized Multitask Learning
Code for performing 3 multitask machine learning methods: deep neural networks, Multitask Multi-kernel Learning (MTMKL), and a hierarchical Bayesian model (HBLR). 

If you have any questions about this code or the associated papers, please email us at jaquesn@mit.edu or sataylor@mit.edu. 

## Please cite our work!

Taylor, S.\*, Jaques, N.\*, Nosakhare, E., Sano, A., Picard, R., <strong> "Personalized Multitask Learning for Predicting Tomorrow’s Mood, Stress, and Health"</strong>, IEEE Transactions on Affective Computing December 2017. <small>(\*equal contribution)</small> <a href="https://affect.media.mit.edu/pdfs/17.TaylorJaques-PredictingTomorrowsMoods.pdf">PDF</a>

Jaques, N.\*, Taylor S.\*, Nosakhare E., Sano A., Picard R., <strong>"Multi-task Learning for Predicting Health, Stress, and Happiness", </strong> NIPS Workshop on Machine Learning for Healthcare, December 2016, Barcelona, Spain. <small>(\*equal contribution)</small> <a href="http://affect.media.mit.edu/pdfs/16.Jaques-Taylor-et-al-PredictingHealthStressHappiness.pdf">PDF</a> <strong>*BEST PAPER AWARD*</strong><br/>


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
The code for generating files in this format given a .csv file is available in make_datasets.py


#### File Format details
- Data for both [users as tasks and labels as tasks] are stored in pickled files as a list of dicts
- Each list item represents a task
- To load a task_list_dict, do the following (use the desired data file name):
```task_dict_list = pickle.load(open(PATH_TO_DATASETS + "datasetTaskList-UsersAsTasks-Discard-Simple-Happiness_Train.p","rb"))```


- There are two types of task dicts:
	- Wellbeing measures as tasks
        - based on the ‘Group’ datasets where labels are computed based on group percentiles
        - Normalization is done based on training data for entire group
	- Users as tasks:
        - based on ‘Personal’ datasets, where labels are based on that person’s label percentiles
        - normalization is done per-person

- Each task is a dict containing 4 keys:
    - ‘Name’: gives the name of the task, eg. "Group_Happiness_Evening_Label" or a user ID
    - ‘X’: the data matrix. Rows are samples, columns are features. Does not contain unnecessary stuff like ‘user_id’ and ‘timestamp’, and has already been normalized and filled
    - ‘Y’: the classification labels for this task, in the same order as the rows of X
    - ‘ModalityDict’: used for MTMKL. Maps modalities like “phys” or “location” to their start index in the feature list 





