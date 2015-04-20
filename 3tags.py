import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
import csv



TOP_TOPICS = [
    183,
    29,
    189,
    78,
    159,
    142,
    147,
    190,
    45,
    124,
]

X_train = []
Y_train =[]
X_test =[]
tt=[]
MAX_TOPICS = 10

"""X_train = np.array(["new york is a hell of a town",
                    "new york was originally dutch",
                    "the big apple is great",
                    "new york is also called the big apple",
                    "nyc is nice",
                    "people abbreviate new york city as nyc",
                    "the capital of great britain is london",
                    "london is in the uk",
                    "london is in england",
                    "london is in great britain",
                    "it rains a lot in london",
                    "london hosts the british museum",
                    "new york is great and so is london",
                    "i like london better than new york"]) """

ifile  = open("quora_input.txt", "rb")
lines = ifile.readlines()
lines[0].replace('\n','')
print lines[0]

values= lines[0].split()


for i in range(1, 40001):
    lines[i]=lines[i].replace('\n','')
    if (i%2==0):
        X_train.append(lines[i])
    else:
        temp= lines[i].split()
        tt=[]
        for j in range(0, int(temp[0])):
            
            tt.append(int(temp[j+1])-1)
            #print tt
            #print "\n"
        Y_train.append(tt)


for i in range(40001,40101):
    X_test.append(lines[i])


stopWords = stopwords.words('english')

target_names = [i+1 for i in xrange(250)]

classifier = Pipeline([
    ('vectorizer', CountVectorizer(stop_words=stopWords, min_df=1)),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])
classifier.fit(X_train, Y_train)
predicted = classifier.predict(X_test)
#print predicted
ifile  = open("quora_output.txt", "w")
for labels in predicted:
    
    for label in labels:
        ifile.write(str(label))
        ifile.write(" ")
    for i in range(0, 10-len(labels)):
        ifile.write(str(TOP_TOPICS[i])+" ")

    ifile.write("\n")



#   print item +"=>" 
 #   print labels