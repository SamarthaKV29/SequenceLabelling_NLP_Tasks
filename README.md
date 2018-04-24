# Error Analysis while performing Sequence Labelling for NLP Tasks

## DESCRIPTION
### Goal
We compare two different Machine Learning approaches and their performance and accuracy using pre-trained word vectors from Glove(Stanford) and CONLL-2003 Shared task dataset (English)
### TEAM 
- Siddharth Sharma (Siddharth_Sharma1@student.uml.edu)
- Samartha K V (SamarthaKajoor_Venkatramana@student.uml.edu) 
- Larry Okeke (Chinonso_Okeke@student.uml.edu) 
## DETAILS
### Current State-of-Art
- Tjong Kim Sang, E. F., & De Meulder, F. (2003, May). Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition. In Proceedings of the seventh conference on Natural language learning at HLT-NAACL 2003-Volume 4 (pp. 142-147). Association for Computational Linguistics.
- Tomanek, K., & Hahn, U. (2009, August). Semi-supervised active learning for sequence labeling. In Proceedings of the Joint Conference of the 47th Annual Meeting of the ACL and the 4th International Joint Conference on Natural Language Processing of the AFNLP: Volume 2-Volume 2 (pp. 1039-1047). Association for Computational Linguistics.

### Key Design decisions: 
- Choose a right word embedding model,
- How much context length is required to get improved accuracy. 
- Metric: F1 score
### APPROACH 
We use two machine learning algorithms, namely Multinomial [Logistic Regression](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) and [Multi-Layer Perceptron](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier), to bring out the differences in performance and accuracy. We check the system for Sequence labeling task over the CONLL-2003 Shared Task English language dataset, to train and test the system. The main goal of this experiment is to perform error analysis and populate the scores over some of the more common tasks of NLP, such as Named Entity Recognition, Parts-Of-Speech Tagging and Chunking. Our dataset is in the form of “WORD NER-tag POS-tag Chunking-tag”. We use the external library vsmlib written in python 3.6 to process the pretrained vectors derived from the [Glove](https://nlp.stanford.edu/projects/glove/), load the model and process all the vocabulary. 
### DATASET
CONLL-2003 
### LEARNING ALGORITHM
- Multinomial Logistic Regression (__Solver__: sag(mulitnomial)/liblinear(OVR))
- Multi-Level Perceptron Neural Network
### TOOLKITS
- [py3.6](https://www.python.org/downloads/release/python-360/)
- [VSMlib](https://github.com/undertherain/vsmlib)
- [scikit-learn](http://scikit-learn.org)
### RESULTS AND TESTING
- Task – n-way classification for NER,POS and chunking.
- Evaluation metric – F1 score 
- F1 Score = 2*(Recall * Precision) / (Recall + Precision)
- Maximize F1 score 
### OTHERS
### TIMELINE 
- March P 1 Acquiring dataset/Preprocessing 
- March/April P 2 Working with LR Classifier (One-vs-Rest or Multinomial)
- April P 3 Working with MLP Classifier
- May P 4 Generate, tabulate and visualize results.
## Pre-Work/ Weekly Upd
#### Week 1
1. Read and implement the tutorial here at https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
2. Read about pandas, scipy, numpy, others.
3. Generate different types of graphs for different datasets.
4. Learn to split the data into test/training sets in numpy or python array. [load_data.py](https://github.com/SamarthaKV29/SequenceLabelling_NLP_Tasks/blob/master/load_data.py)
#### Week 2
5. Creating logistic regression classifier from sklearn.linear_model
6. Training and testing it with different context width and word vector dimensions
#### Week 3
7. Created Linux branch
#### Week 4
8. Created mlp classifier
9. Trained and tested MLP classifer
#### Week 5
10. Results:
    - LRC
         - Accuracy: 70-92 %
         - Performance: Running Time(fitting): 13-40 mins on target system.
    - MLP
         - Accuracy: 90-97 %
         - Performance: Running Time(fitting): 1-4 mins on target system.
11. Performing error analysis.
### TEAM ROLES 
##### Samartha K V 
- Design and implement code
- Performing testing and documentation
- Generating results
- Error analysis and reasoning.
##### Siddharth Sharma 
- Coming up with ideas for further applications and research,  
- Choosing context length and testing with different lengths,
- Generating the final report
- Error analysis and reasoning
##### Larry Okeke 
- Evaluating and choosing the correct model,
- Applying the concepts picked up in class to our project 
- Checking the mathematical integrity
- Finding current state of the art
