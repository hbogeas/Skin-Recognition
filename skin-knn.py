import time
start_time = time.clock()

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

df = pd.read_csv('Skin.csv', delimiter='\t')

np.random.seed(123)
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

trainnp = np.array(train)
testnp = np.array(test)
Xtr = trainnp[:,0:3]
Ytr = trainnp[:,3]
Xte = testnp[:,0:3]
Yte = testnp[:,3]

tempYtr = np.zeros([len(Ytr),2])
tempYte = np.zeros([len(Yte),2])

for i in range(len(Ytr)):
    tempYtr[i,(Ytr[i]-1)] = 1        

for i in range(len(Yte)):
    tempYte[i,(Yte[i]-1)] = 1
    
Ytr = tempYtr
Yte = tempYte

xtr = tf.placeholder("float", [None, 3])
xte = tf.placeholder("float", [3])

distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
pred = tf.arg_min(distance, 0)
accuracy = 0.
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # loop over test data
    results = np.zeros([len(Xte),2])
    for i in range(len(Xte)):
        # Get nearest neighbor
        nn_index = sess.run(
            pred,
            feed_dict={
                xtr: Xtr,
                xte: Xte[i, :]
            }
        )
        # Get nearest neighbor class label and compare it to its true label
        results[i,0] = np.argmax(Ytr[nn_index])
        results[i,1] = np.argmax(Yte[i])
        l = (str(i) + ",Prediction: "+ str(np.argmax(Ytr[nn_index])) + ",True Class: " + str(np.argmax(Yte[i])) + '\n')
        with open('knn-results.txt','a+') as f :
                f.write(l)
                f.close()
        print(l)       
        
        # Calculate accuracy
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
    print("Done!")
    y_actu = pd.Series(results[:,1], name='Actual')
    y_pred = pd.Series(results[:,0], name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print("Accuracy:", accuracy)

print(df_confusion)

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

plot_confusion_matrix(df_confusion)

print(time.clock() - start_time, "seconds")    

