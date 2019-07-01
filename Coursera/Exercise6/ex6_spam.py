from scipy.io import loadmat
import numpy as np
from sklearn.svm import SVC

from Coursera.Exercise6.EmailFeatures import email_features
from Coursera.Exercise6.ProcessEmail import process_email
from utils.file_utils import FileUtils

file_path = FileUtils.get_abs_path(__file__, "./data/emailSample1.txt")
vocab_path = FileUtils.get_abs_path(__file__, "./data/vocab.txt")

file_contents = open(file_path, "r").read()
vocabList = open(vocab_path, "r").read()

vocabList = vocabList.split("\n")[:-1]

vocabList_d = {}
for ea in vocabList:
    value, key = ea.split("\t")[:]
    vocabList_d[key] = value

print(file_contents)

word_indices = process_email(file_contents, vocabList_d)
features = email_features(word_indices, vocabList_d)
print("Length of feature vector: ", len(features))
print("Number of non-zero entries: ", np.sum(features))


spam_mat_path = FileUtils.get_abs_path(__file__, "./data/spamTrain.mat")
spam_mat = loadmat(spam_mat_path)
X_train = spam_mat["X"]
y_train = spam_mat["y"]

C = 0.1
spam_svc = SVC(C=0.1, kernel="linear")
spam_svc.fit(X_train, y_train.ravel())
print("Training Accuracy:", (spam_svc.score(X_train, y_train.ravel())) * 100, "%")

spam_mat_test_path = FileUtils.get_abs_path(__file__, "./data/spamTest.mat")
spam_mat_test = loadmat(spam_mat_test_path)
X_test = spam_mat_test["Xtest"]
y_test = spam_mat_test["ytest"]

print("Test Accuracy:", (spam_svc.score(X_test, y_test.ravel())) * 100, "%")


file_path = FileUtils.get_abs_path(__file__, "./data/spamSample1.txt")
file_contents = open(file_path, "r").read()



word_indices = process_email(file_contents, vocabList_d)
features = email_features(word_indices, vocabList_d)
features = features.reshape([1, 1899])

print(spam_svc.predict(features))
print('1 is spam, 0 is not spam')
