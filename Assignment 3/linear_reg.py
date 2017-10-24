import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def load_data():
    with np.load("./data/notMNIST.npz") as data :
        Data, Target = data ["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]

        trainData = trainData.reshape(-1, 28 * 28)
        validData = validData.reshape(-1, 28 * 28)
        testData = testData.reshape(-1, 28 * 28)

        return trainData, trainTarget, validData, validTarget, testData, testTarget


def build_graph(learn_rate, decay_lambda):
    W = tf.Variable(tf.truncated_normal(shape=[784, 1], stddev=1.0), name='weights')
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float32, shape = [None, 784], name = 'x_inputs')
    y = tf.placeholder(tf.float32, shape = [None, 1], name = 'y_targets')

    logits = tf.matmul(X, W) + b
    cost_func = tf.reduce_mean(tf.reduce_mean(tf.square(logits-y),
                                              axis=1,
                                              name='square_error')) + tf.reduce_sum(tf.square(W)) * decay_lambda /2

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
    train = optimizer.minimize(loss = cost_func)

    return W, b, X, y, logits, cost_func, train


def run(learn_rate, decay_lambda, batch_size, num_epoch):
    trainData, trainTarget, validData, validTarget, testData, testTarget = load_data()

    W, b, X, y, y_pred, error, train = build_graph(learn_rate, decay_lambda)

    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()
    sess.run(init)

    loss_train = []
    loss_valid = []
    loss_test = []

    accuracy_list = []
    accuracy_valid = []
    accuracy_test = []

    num_update = 0

    num_train_data = trainData.shape[0]
    rand_index = np.arange(num_train_data)
    num_steps = int(np.ceil(num_train_data/batch_size))
    print("Number of steps are:")
    print(num_steps)

    for i in range(num_epoch):
        np.random.shuffle(rand_index)
        train_input = trainData[rand_index]
        train_target = trainTarget[rand_index]

        for step in range(num_steps):
            start = step * batch_size
            end = min(num_train_data, (step + 1) * batch_size)
            x_train = train_input[start : end]
            y_train = train_target[start : end]

            _, err, cur_W, cur_b, y_hat = sess.run([train, error, W, b, y_pred],
                                                   feed_dict={X : x_train, y : y_train})

            #Eval
            train_err, train_pred = sess.run([error, y_pred],
                                             feed_dict={X: trainData, y: trainTarget})
            valid_err, valid_pred = sess.run([error, y_pred],
                                             feed_dict={X: validData, y: validTarget})
            test_err, test_pred = sess.run([error, y_pred],
                                           feed_dict={X: testData, y: testTarget})

            loss_train.append(train_err)
            loss_valid.append(valid_err)
            loss_test.append(test_err)

            train_pred = sign(train_pred)
            valid_pred = sign(valid_pred)
            test_pred = sign(test_pred)

            train_accuracy = accuracy_score(trainTarget,train_pred)
            valid_accuracy = accuracy_score(validTarget, valid_pred)
            test_accuracy = accuracy_score(testTarget, test_pred)

            accuracy_list.append(train_accuracy)
            accuracy_valid.append(valid_accuracy)
            accuracy_test.append(test_accuracy)

            num_update += 1
            print("Epoch: %3d, Iter: %3d, Loss-train: %4.2f, bias: %.2f" % (i, step, train_err, cur_b))

    # Final evaluation on the validation set
    print("-------------------------------------------------")
    print("Final train Cross Entropy+regularization: %.2f" % train_err)
    valid_err, valid_pred = sess.run([error, y_pred], feed_dict={X: validData, y: validTarget})
    print("Final valid Cross Entropy+regularization: %.2f" % valid_err)

    # Final evaluation on the test set
    test_err, test_pred = sess.run([error, y_pred], feed_dict={X: testData, y: testTarget})
    print("Final testing Cross Entropy+regularization: %.2f" % test_err)

    print("-------------------------------------------------")
    print("Train Data Shape:", trainData.shape)
    print("Valid Data Shape:", validData.shape)
    print("Test Data Shape:", testData.shape)

    valid_pred = sign(valid_pred)
    test_pred = sign(test_pred)

    valid_accuracy = accuracy_score(validTarget, valid_pred)
    test_accuracy = accuracy_score(testTarget, test_pred)

    print("-------------------------------------------------")
    print("Train Data Accuracy:", train_accuracy)
    print("Valid Data Accuracy:", valid_accuracy)
    print("Test Data Accuracy:", test_accuracy)

    plt.figure()
    plt.title("Loss learning rate: %.2f batch size: %.2f num_epoch: %.2f" % (learn_rate, batch_size, num_epoch))
    plt.plot(np.arange(num_update), loss_train, label="Training Set")
    plt.plot(np.arange(num_update), loss_valid, label="Validation Set")
    plt.plot(np.arange(num_update), loss_test, label="Test Set")
    plt.legend(loc='upper right')

    plt.ylabel('loss')
    plt.xlabel('num of updates')
    plt.show()

    plt.figure()
    plt.title("Accuracy learning rate: %.2f batch size: %.2f num_epoch: %.2f" % (learn_rate, batch_size, num_epoch))
    plt.plot(np.arange(num_update), accuracy_list, label="Training Set")
    plt.plot(np.arange(num_update), accuracy_valid, label="Validation Set")
    plt.plot(np.arange(num_update), accuracy_test, label="Test Set")
    plt.legend(loc='lower right')
    plt.ylabel('accuracy')
    plt.xlabel('num of updates')
    plt.show()


def sign(X):
    for i, num in enumerate(X):
        if num >= 0.5:
            X[i] = 1
        else:
            X[i] = 0
    return X

if __name__ == '__main__':
    learn_rate = 0.005
    training_epochs = 500
    decay_lambda = 0

    batch_size = 500

    np.set_printoptions(precision=2)
    run(learn_rate, decay_lambda, batch_size, training_epochs)
