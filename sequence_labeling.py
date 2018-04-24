import numpy as np
#from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import yaml
import load_data
import argparse
import vsmlib
import time


def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''

    if win < 1:
       win = 1
    l = list(l)
    # print((int)(win/2))
    lpadded = (int)(win) * [0] + l + (int)(win) * [0]
    out = [lpadded[i:i + win * 2 + 1] for i in range(len(l))]
    if len(out) == len(l):
        return out
    

'''
Get sequence labeling task's inp and output.
'''


def getinpOutput(lex, y, win, idx2word):
    inp = []
    output = []
    for i in enumerate(lex):
        wordListList = contextwin(lex[i], win)
        for j in enumerate(wordListList):
            wordList = wordListList[j]
            realWordList = [idx2word[word] for word in wordList]
            inp.append(realWordList)
            output.append(y[i][j])
    return inp, output


'''
get inp (X) embeddings
'''


def getX(inp, m):
    x = []
    OOV_count = 0
    token_count = 0
    # print(m.matrix.shape[0])
    # just a oOV word to random
    random_vector = m.matrix.sum(axis=0) / m.matrix.shape[0]
    print(random_vector)
    # random_vector = m.matrix[0]
    for wordList in inp:
        v = []
        for word in wordList:
            if m.has_word(word):
                wv = m.get_row(word)
            else:
                wv = random_vector
                OOV_count += 1
            token_count += 1
            v.append(wv)
        v = np.array(v).flatten()  # 2d -> 1d
        x.append(v)

    print("Out of vocabulary rate is: %f" % (OOV_count * 1. / token_count))
    print("Vocabulary cover rate is: %f" %
          ((token_count - OOV_count) * 1. / token_count))
    return x


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_vectors', default='./../../../test/data/embeddings/npy/',
                        help='path to the embeddings')
    parser.add_argument('--path_dataset', default='./../../../test/data/benchmarks/sequence_labeling/',
                        help='path to the dataset')
    parser.add_argument('--window', '-w', default=2, type=int,
                        help='window size')
    args = parser.parse_args()
    return args


options = {}


def main():
    epochs = 10
    # use ArgumentParser
    # args = parse_args()
    stTime = time.time()
    TMUL = 60
    # use yaml
    global options
    if len(sys.argv) > 1:
        path_config = sys.argv[1]
    else:
        print("Run command : python3 sequence_labeling.py config.yaml")
        return

    with open(path_config, "r", encoding='utf-8') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    options["path_vectors"] = cfg["path_vectors"]
    options["path_dataset"] = cfg["path_dataset"]
    options["window"] = cfg["window"]
    options["task"] = cfg["task"]
    # get the embeddings
    m = vsmlib.model.load_from_dir(options['path_vectors'])
    # specify the task (can be ner, pos or chunk)
    task = options['task']

    # get the dataset
    train_set, valid_set, test_set, dic = load_data.load(
        options['path_dataset'], task)
    print(len(dic['words2idx']))

    idx2label = dict((k, v) for v, k in dic['labels2idx'].items())
    idx2word = dict((k, v) for v, k in dic['words2idx'].items())

    train_lex, train_y = train_set
    valid_lex, valid_y = valid_set
    test_lex, test_y = test_set
    # print(train_y)

    # add validation data to training data.
    # train_lex.extend(valid_lex)
    # train_ne.extend(valid_ne)
    # train_y.extend(valid_y)

    vocsize = len(dic['words2idx'])
    nclasses = len(dic['labels2idx'])
    # print(nclasses)

    # get the training and test's inp and output
    my_train_inp, my_train_y = getinpOutput(
        train_lex, train_y, options['window'], idx2word)
    print("TRAIN")
    my_train_x = getX(my_train_inp, m)
    my_test_inp, my_test_y = getinpOutput(
        test_lex, test_y, options['window'], idx2word)
    print("TEST")
    my_test_x = getX(my_test_inp, m)
    print("Fitting the MLP NN Classifier")
    # fit LR classifier
    # lrc = LogisticRegression(tol=0.00011,
    #                          class_weight='balanced', multi_class='ovr', random_state=7, max_iter=200)  # n_jobs=-1, solver='sag',
    # lrc.fit(my_train_x, my_train_y)
    mlp = MLPClassifier()
    mlp.partial_fit(my_train_x, my_train_y, np.unique(my_train_y))
    for i in range(epochs):
        print("Epoch %d " % i)
        mlp.partial_fit(my_train_x, my_train_y)
    # svm = SVC()
    # svm.fit(my_train_x, my_train_y)
    # get results
    print("Elapsed: %f mins" % ((time.time() - stTime)/TMUL))
    print("Generating Results")

    mlppredtrain = mlp.predict(my_train_x)
    mlppredtest = mlp.predict(my_test_x)

    # f1_score_train = f1_score(my_train_y, pred_train, average='weighted')
    # f1_score_test = f1_score(my_test_y, pred_test, average='weighted')

    # print("Training set F1 score: %f" % f1_score_train)
    # print("Test set F1 score: %f" % f1_score_test)
    if task == 'pos':
        train_mlp_score = accuracy_score(my_train_y, mlppredtrain)
        test_mlp_score = accuracy_score(my_test_y, mlppredtest)
        print("Training MLP score: %f" % train_mlp_score )
        print("Testing MLP score: %f" % test_mlp_score)
    else: 
        f1_score_train_mlp = f1_score(my_train_y, mlppredtrain, average='weighted')
        f1_score_test_mlp = f1_score(my_test_y, mlppredtest, average='weighted')

        print("Training MLP set F1 score: %f" % f1_score_train_mlp)
        print("Test MLP set F1 score: %f" % f1_score_test_mlp)
    print("Elapsed: %f mins" % ((time.time() - stTime)/TMUL))
    # pred = svm.predict(my_test_x)
    # f1 = f1_score(my_train_y, pred, average='weighted')
    # print("Test score: %f" % f1  )


if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    main()