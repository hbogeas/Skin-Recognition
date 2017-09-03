e=10
b=100
import pandas as pd
import tensorflow as tf
import numpy as np
import random
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


learning_rate = 0.001
training_epochs = e
batch_size = b
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 3])
y = tf.placeholder(tf.float32, [None, 2])

# Set model weights
tf.set_random_seed(123)
W = tf.Variable(tf.random_normal([3, 2], stddev=0.35), name="weights")
b = tf.Variable(tf.random_normal([2], stddev=0.35), name="biases")


# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
#cost = tf.reduce_mean(-tf.reduce_sum(y*pred, reduction_indices=1))
#cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=pred))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    epochs = []
    tr_acc = []
    test_acc = []
    costs = []
    for epoch in range(training_epochs):
        avg_cost = 0.
        avg_acc = 0.
        total_batch = int(len(Xtr)/batch_size)
        random.seed(training_epochs)
        sample_batches = random.sample(range(1, (total_batch+1)), total_batch)
        for i in sample_batches:
            # Run optimization op (backprop) and cost op (to get loss value)
            end = (batch_size*i)
            batch_xs = Xtr[end-batch_size:end,:]
            batch_ys = Ytr[end-batch_size:end,:]
                        
            _, c, acc = sess.run(
                [
                    optimizer,
                    cost,
                    accuracy
                ],
                feed_dict={
                    x: batch_xs,
                    y: batch_ys
                }
            )
            avg_cost += c / total_batch
            avg_acc  += acc / total_batch
            
        if (epoch+1) % display_step == 0:
            #cmpute test acc
            tes_acc = sess.run( accuracy, feed_dict={ 
                x: Xte,
                y: Yte 
            } )
            #print tes_acc
            l = ("Learning_rate: "+str(learning_rate)+
                 ",Epoch: "+str('%04d' % (epoch+1))+
                 ",Batch_size: "+str('%04d' % batch_size)+
                ",cost="+"{:.5f}".format(avg_cost)+
                ",train accuracy="+"{:.5f}".format(avg_acc)+
                ",test accuracy="+"{:.5f}".format(tes_acc)+'\n')
            with open('log-results.txt','a+') as f :
                f.write(l)
                f.close()
            print(l)
            # add results to matrices
            epochs.append(epoch+1)
            tr_acc.append(avg_acc)
            test_acc.append(tes_acc)
            costs.append(avg_cost)
            #
    print("Optimization Finished!")
sess.close()

#Summarizing Data
dfs = df
dfs['skin'] = dfs['skin'].astype('category')
dfs.dtypes
for i in ['blue','red','green']:
    print(dfs[i].groupby(dfs['skin']).describe())
pd.options.display.mpl_style = 'default'
dfs.groupby('skin').boxplot()

#Learning Curve and Train Cost 
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (20, 6)
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

fig1 = plt.figure()

ax = fig1.add_subplot(121)
ax.clear()

ax.set_title("Learning curve")
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.set_ylim([0, 1])
ax.plot(epochs, tr_acc,     'o-', color="g", label="Train Accuracy")
ax.plot(epochs, test_acc,   'o-', color="r", label="Test Accuracy")

ax2 = fig1.add_subplot(122)
ax2.clear()

ax2.set_title("Train cost")
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Cost')
ax2.set_ylim(ymin=0)
ax2.plot(epochs, costs, 'o-', color="r", label="Train cost")