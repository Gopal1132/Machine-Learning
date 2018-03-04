import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

X = tf.placeholder('float',shape = [None,784])
Y = tf.placeholder('float')
n_classes = 10
Batch_Size = 128

def Conv2d(Input,Filter):
    return tf.nn.conv2d(Input,Filter,strides=[1,1,1,1],padding = 'SAME')

def MaxPool(Data):
    return tf.nn.max_pool(Data,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')

#Conv Layer Model
def Convolutional_Neural_Network(data):
    Weight = {
        'W_Conv1':tf.Variable(tf.random_normal([5,5,1,32])),
        'W_Conv2':tf.Variable(tf.random_normal([5,5,32,64])),
        'Full_Connected':tf.Variable(tf.random_normal([7*7*64,500])),
        'Output_Layer':tf.Variable(tf.random_normal([500,n_classes]))
        }
    Biases = {
         'W_Conv1':tf.Variable(tf.random_normal([32])),
        'W_Conv2':tf.Variable(tf.random_normal([64])),
        'Full_Connected':tf.Variable(tf.random_normal([500])),
        'Output_Layer':tf.Variable(tf.random_normal([n_classes]))
        }

    data = tf.reshape(data,shape = [-1,28,28,1])

    Conv_1 = Conv2d(data,Weight['W_Conv1']) + Biases['W_Conv1']
    Conv_1 = MaxPool(Conv_1)
    Conv_1 = tf.nn.relu(Conv_1)

    Conv_2 = Conv2d(Conv_1,Weight['W_Conv2']) + Biases['W_Conv2']
    Conv_2 = MaxPool(Conv_2)
    Conv_2 = tf.nn.relu(Conv_2)
    Full_Connected = tf.reshape(Conv_2,shape = [-1, 7*7*64])   #Flattening

    Full_Connected = tf.add(tf.matmul(Full_Connected,Weight['Full_Connected']),Biases['Full_Connected'])
    Full_Connected = tf.nn.relu(Full_Connected)

    Output = tf.add(tf.matmul(Full_Connected,Weight['Output_Layer']),Biases['Output_Layer'])
    return Output

def Train_Convolutional_Neural_Network(X):
    Prediction = Convolutional_Neural_Network(X)
    Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Prediction,labels=Y))
    Optimizer = tf.train.AdamOptimizer().minimize(Loss)

    Total_Epochs = 10
    with tf.Session() as Sess:
        Sess.run(tf.initialize_all_variables())

        for Epoch in range(Total_Epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/Batch_Size)):
                Epoc_X,Epoc_Y = mnist.train.next_batch(Batch_Size)
                _,Loss = Sess.run([Optimizer,Loss],feed_dict = {X:Epoc_X,Y:Epoc_Y})
                epoch_loss += Loss
            print('Epoch',Epoch,'Completed Out Of Total',Total_Epochs,'Loss Gained:',epoch_loss)
                
        Correct = tf.equal(tf.argmax(Prediction,axis = 1),tf.argmax(Y,axis = 1))
        Accuracy = tf.reduce_mean(tf.cast(Correct, 'float'))
        print('Accuracy:',Accuracy.eval({X:mnist.test.images, Y:mnist.test.labels}))

if __name__ == '__main__':
    Train_Convolutional_Neural_Network(X)
    


























    
