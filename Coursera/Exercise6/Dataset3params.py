from sklearn.svm import SVC


def dataset_3_params(X, y, Xval, yval):
    vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    max_score = 0
    best_c = 0
    best_gamma = 0
    y = y.ravel()
    yval = yval.ravel()
    for i in vals:
        for j in vals:
            classifier = SVC(C=i, gamma=j)
            classifier.fit(X, y)
            prediction = classifier.predict(Xval)
            score = classifier.score(Xval, yval)
            if score > max_score:
                max_score = score
                best_c = i
                best_gamma = j

    return best_c, best_gamma
