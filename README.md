# SEQUENCE LABELING and SENTENCE ANALYSIS for NLP RELATED USE CASES

## DESCRIPTION
### Goal
Given an annotated dataset of named entities, we train, test and validate a few Machine learning algorithms to perform one of the above-mentioned task, NER, POS or chunking, using the available library vsmlib for processing sentences into word vectors and then using these annotations and the word embeddings to train and generate predictions on the dataset or other similar dataset. Next, we will try to use the above said systems to implement basic user command processing or to understand the user’s sentences. 
### TEAM 
- Siddharth Sharma (Siddharth_Sharma1@student.uml.edu)
- Samartha K V (SamarthaKajoor_Venkatramana@student.uml.edu) 
- Larry Okeke (Chinonso_Okeke@student.uml.edu) 
## DETAILS
### Current State-of-Art
Currently we have a lot of existing systems to perform Chunking, POS and NER. We use a subset of these tools and then utilize the existing libraries in python, which help to 
perform the required task. Many research papers talk about analysis of active learning strategies for such tasks. Such comparisons help to make the right choice when we want to improve the accuracy and when we need to decide what exactly we can achieve using these methods and tools. 
### Key Design decisions: 
✓ Choose a right word embedding model, ✓ how much context length is required to get improved accuracy. ✓ Loss function: Categorical entropy,  ✓ Model: Logistic Regression, and ✓ Metric: F1 score
### APPROACH 
We will utilize CONLL dataset to evaluate the performance of word embeddings on sequence labelling tasks. We build the classifier train it and test it and get the performance metrics. Later, we might upgrade the program to perform sentence analysis for helping our system interact and understand the user’s words and commands or we might implement an ANN/RNN to do the same. DATASET 

### DATASET
CONLL-2003 
### LEARNING ALGORITHM
Logistic Regerssion
### TOOLKITS
- VSMlibs helps to perform a range of tasks within a framework of vector space models of computational linguistics.
- scikit-learn

### RESULTS AND TESTING
- Task – n-way classification 
- Evaluation metric – F1 score 
- F1 Score = 2*(Recall * Precision) / (Recall + Precision)
- Maximize F1 score 
### OTHERS
### TIMELINE 
- March P 1 Acquiring dataset/Preprocessing 
- March/April P 2 Choose the model/learning algorithm
- April P 3 Train/validate accuracy 
- May P 4 Experiment with ANNs 
 

## Pre-Work/ Weekly Upd
1. Read and implement the tutorial here at https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
2. Read about pandas, scipy, numpy, others.
3. Work with dataframes, python arrays and learn pandas here http://pandas.pydata.org/pandas-docs/stable/10min.html and mathplotlib.
4. Generate different types of graphs for different datasets.
5. Learn to split the data into test/training sets in numpy or python array.
### TEAM ROLES 
##### Samartha K V 
Designing and developing the final program, Integrating required libraries, and Managing processing speed vs accuracy and CPU overhead and efficiency of the program. 
##### Siddharth Sharma 
Coming up with ideas for further applications and research,  Choosing context length and testing with different lengths, Choose between POS, NER and chunking based on what our application will turn out to be. 
##### Larry Okeke 
Evaluating and choosing the correct model,  Applying the concepts picked up in class to our project Checking the mathematical integrity, generating more datasets to verify maximum of the use cases

##### Current

1. https://stackoverflow.com/questions/11874767/real-time-plotting-in-while-loop-with-matplotlib

2. http://scikit-learn.org/stable/modules/generated/sklearn.svm.libsvm.fit.html

3. https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.split.html

### RELATED - MACHINE LEARNING VIDEO LECTURES
https://www.youtube.com/watch?v=oID20dIrV94&list=PLiPvV5TNogxKKwvKb1RKwkq2hm7ZvpHz0&t=430
