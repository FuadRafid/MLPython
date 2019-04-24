import re
from nltk.stem import PorterStemmer


def process_email(email_contents, vocabList_d):
    email_contents = email_contents.lower()
    email_contents = re.sub('<[^<>]+>', ' ', email_contents);
    email_contents = re.sub("[0-9]+", "number ", email_contents)
    email_contents = re.sub("[http|https]://[^\s]*", "httpaddr ", email_contents)
    email_contents = re.sub("[^\s]+@[^\s]+", "emailaddr ", email_contents)
    email_contents = re.sub("[$]+", "dollar ", email_contents)
    special_char = ["<", "[", "^", ">", "+", "?", "!", "'", ".", ",", ":", "-"]
    for char in special_char:
        email_contents = email_contents.replace(str(char), "")
    email_contents = email_contents.replace("\n", " ")
    ps = PorterStemmer()
    email_contents = [ps.stem(token) for token in email_contents.split(" ")]
    email_contents = " ".join(email_contents)

    print(email_contents)
    word_indices = []

    for word in email_contents.split():
        if len(word) > 1 and word in vocabList_d:
            word_indices.append(int(vocabList_d[word]))

    return word_indices
