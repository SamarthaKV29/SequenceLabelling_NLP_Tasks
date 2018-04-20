import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import sys
import yaml
import load_data
import argparse
import vsmlib


def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    
    assert win >= 1
    l = list(l)
    # print((int)(win/2))
    lpadded = (int)(win) * [0] + l + (int)(win) * [0]
    out = [lpadded[i:i + win * 2 + 1] for i in range(len(l))]
    # print(out)
    assert len(out) == len(l)
    return out


'''
Get sequence labeling task's inp and output.
'''


def getinpOutput(lex, y, win, idx2word):
    inp = []
    output = []
    for i in range(len(lex)):
        wordListList = contextwin(lex[i], win)
        for j in range(len(wordListList)):
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
    print(m.matrix.shape[0])
    #just a oOV word to random
    random_vector = m.matrix.sum(axis=0) / m.matrix.shape[0]
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
        v = np.array(v).flatten() #2d -> 1d
        x.append(v)
    print("out of vocabulary rate : %f" % (OOV_count * 1. / token_count))
    print("vocabulary cover rate : %f" %
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

    # use ArgumentParser
    # args = parse_args()

    # use yaml
    global options
    path_config = "config.yaml"

    with open(path_config, "r", encoding='utf-8') as ymlfile:
        cfg = yaml.load(ymlfile)
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

    idx2label = dict((k, v) for v, k in dic['labels2idx'].items())
    idx2word = dict((k, v) for v, k in dic['words2idx'].items())

    train_lex, train_y = train_set
    valid_lex, valid_y = valid_set
    test_lex, test_y = test_set
    # print(train_y)

    # add validation data to training data.
    train_lex.extend(valid_lex)
    # train_ne.extend(valid_ne)
    train_y.extend(valid_y)

    vocsize = len(dic['words2idx'])
    nclasses = len(dic['labels2idx'])
    # print(nclasses)

    # get the training and test's inp and output
    my_train_inp, my_train_y = getinpOutput(
        train_lex, train_y, options['window'], idx2word)
    my_train_x = getX(my_train_inp, m)
    my_test_inp, my_test_y = getinpOutput(
        test_lex, test_y, options['window'], idx2word)
    my_test_x = getX(my_test_inp, m)
    
    
    # fit LR classifier
    lrc = LogisticRegression(n_jobs=4, solver='lfbgs')
    lrc.fit(my_train_x, my_train_y)
    
    
    
    svm = SVC()
    svm.fit(my_train_x, my_train_y)
    # get results
    if task == 'pos':
        score_train = lrc.score(my_train_x, my_train_y)
        score_test = lrc.score(my_test_x, my_test_y)
        print("training set accuracy: %f" % (score_train))
        print("test set accuracy: %f" % (score_test))
    else:
        pred_train = lrc.predict(my_train_x)
        pred_test = lrc.predict(my_test_x)
        f1_score_train = f1_score(my_train_y, pred_train, average='weighted')
        f1_score_test = f1_score(my_test_y, pred_test, average='weighted')
        print("Training set F1 score: %f" % f1_score_train)
        print("Test set F1 score: %f" % f1_score_test)
    
    pred = svm.predict(my_test_x)
    f1 = f1_score(my_train_y, pred, average='weighted')
    print("Test score: %f" % f1  )


if __name__ == '__main__':
    main()
