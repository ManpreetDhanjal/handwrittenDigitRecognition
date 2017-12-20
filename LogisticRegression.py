import numpy as np
import pandas as pd
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
from decimal import *
import gzip

IMAGE_SIZE = 28
FEATURE_SIZE = IMAGE_SIZE * IMAGE_SIZE + 1
NUM_CLASS = 10
LAMBDA = 0.001

# extract the data from the file and insert 1 for the bias term
def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_SIZE * IMAGE_SIZE)
        # normalise data
        data = normalise_data(data)
        # add 1 in the beginning for bias term
        data = np.insert(data, 0, 1, axis=1)
    return data

# normalise the image data to lie between 0 to 1
def normalise_data(data):
    arr = data.reshape(data.shape[0] * data.shape[1],1)
    max = np.max(arr)
    min = np.min(arr)
    diff = max - min;
    x = arr / diff
    return x.reshape(data.shape)

# extract the labels of the input data from input file
def extract_label(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int16)
    return labels

# applies the softmax and returns array with probabilities of all classes
def softmax(prediction):
    exp_vec = np.exp(prediction)
    res_exp_vec = np.zeros((prediction.shape[0],prediction.shape[1]))
    for i in range(prediction.shape[0]):
        res_exp_vec[i,:] = exp_vec[i,:]/np.sum(exp_vec[i,:])
    return res_exp_vec

# convert the input labels to Nx10 vector representing the probability for each class
def format_labels(labels):
    result = np.zeros((labels.size,NUM_CLASS))
    result[np.arange(result.shape[0]), labels] = 1
    return result

# returns derivative of error for output layer
def get_error(prediction, label, input_data):
    print("get error")
    label = label.reshape(prediction.shape)
    s = np.matmul(input_data.T, prediction-label)
    return s;

# train the logistic regression
# returns weights
def train_logistic_regression():
    total_iterations = 1000
    i = 0
    learning_param = 0.9
    N = train_input.shape[0];
    weights = np.random.rand(FEATURE_SIZE, NUM_CLASS)
    labels = format_labels(train_label)
    
    param = learning_param/N;
    
    while i < total_iterations:
        # NxK
        prediction = np.matmul(train_input, weights)
        exp_vec = softmax(prediction)
        E_D = get_error(exp_vec, labels, train_input)
        reg_wt = LAMBDA * weights;
        reg_wt[0,:] = 0;
        weights = weights - param * (E_D + reg_wt)
        i = i+1
        print(i)
    return weights

# returns classification for test_input       
def test_logistic_regression(test_input, weights):
    return softmax(np.matmul(test_input, weights))

# returns accuracy on the test data
def get_accuracy(weights, input_data, output_labels):
    predicted = softmax(np.matmul(input_data, weights))
    predicted_output = np.argmax(predicted, axis=1)
    corr = np.sum(np.equal(predicted_output,output_labels))
    return corr/predicted.shape[0]

def main():
	test_input = extract_data('/Users/manpreetdhanjal/Downloads/t10k-images-idx3-ubyte.gz', 10000)
	test_label = extract_label('/Users/manpreetdhanjal/Downloads/t10k-labels-idx1-ubyte.gz', 10000)
	train_input = extract_data('/Users/manpreetdhanjal/Downloads/train-images-idx3-ubyte.gz', 60000)
	train_label = extract_label('/Users/manpreetdhanjal/Downloads/train-labels-idx1-ubyte.gz', 60000)

	train_logistic_regression()

main()