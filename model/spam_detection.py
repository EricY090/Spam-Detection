import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from matplotlib import pyplot as plt 
import pickle
from utils import text_preprocess, calculate_tf, build_tf_idf, expit




### Gradient of the loss function in logistic regression respect to w in matrix form: (1/n)(sigmoid(x*w^T)-y)^T *x + 2*alpha*w
def evaluate_gradient_w(x, y, b, w, method, penalty):
    if method ==0:   
        n = len(x)                        ### For mini-batch
        y_hat = np.array(np.dot(x, w.T) + b, dtype=np.float32)
        s = expit(y_hat)            ### (#instances x 1)
        grad = (1/n)*np.dot(((s-y).T), x)
        grad = grad + 2*penalty*w
    else:                                    ### For stochastic
        y_hat = np.array(np.dot(x, w.T) + b, dtype=np.float32)
        s = expit(y_hat)            ### 1 x 1 value
        grad = (s-y)*x
        grad = grad + 2*penalty*w
    return grad


### Gradient of the loss function in logistic regression respect to w in matrix form: (1/n)(sigmoid(x*w^T)-y)^T 
def evaluate_gradient_b(x, y, b, w, method):
    if method ==0:                           ### For mini-batch
        n = len(x)
        s = expit(np.dot(x, w.T) +b)            ### (#instances x 1)
        grad = (1/n)*np.sum(s-y)
    else:                                    ### For stochastic
        s = expit(np.dot(x, w.T) +b)            ### 1 x 1 value
        grad = s-y
    return grad


### GD or Mini-GD when "method" is passed in with 0; Stochastic Gradient Descent when passed in with 1
def GD_MiniGD_SGD(w, b, lr, x, y, epochs, batch_size, method, valid_x, valid_y, alpha):
    cost_hist = []          ### Collect the validation loss after each epoch
    for i in range(epochs):
        print(f"Epoch: {i+1}")
        if method==0:                    ### For mini-batch gradient decent
            j = 0
            while(j < np.shape(y)[0]):   ### fit one mini-batch a time
                k = j+batch_size
                if k > np.shape(y)[0]:
                    k= np.shape(y)[0]
                diff_w = evaluate_gradient_w(x[j:k], y[j:k], b, w, method, alpha)      ### Update the weight
                diff_b = evaluate_gradient_b(x[j:k], y[j:k], b, w, method)
                w = w - lr*diff_w
                b = b - lr*diff_b
                j+=batch_size
        else:                            ### For stochastic gradient descent
            for k in range(x.shape[0]):
                diff_w = evaluate_gradient_w(x[k], y[k][0], b, w, method, alpha)
                diff_b = evaluate_gradient_b(x[k], y[k][0], b, w, method)
                b = b - lr*diff_b
                w = w - lr*diff_w
        y_hat = expit(np.array(np.dot(valid_x, w.T) + b))
        cost= -(1/len(valid_x))*np.sum(valid_y*np.log(y_hat)+(1-valid_y)*np.log(1-y_hat)) + alpha*np.sum(w**2)  ### Loss function
        cost_hist.append(cost)
    return cost_hist, w, b




if __name__ == "__main__":
    # -----Data Preprocessing----- #
    ### Read the SMSSpamCollection file
    df = pd.read_csv("sms_sample_text", sep='\t', header=None)
    data = np.array(df)

    ### Remove punctuation, urls, and numbers. Change text to lower case. Tokenized, word stemming and remove stopwords.
    for indx, i in enumerate(data):
        i[1] = text_preprocess(i[1])
        i[0] = 1 if i[0] =='spam' else 0
    
    print(data[:10])


    ### Split into 80% train, 10% validation, 10% test
    seed = 1689
    data = shuffle(data, random_state=seed)        ### Shuffle the entire dataset by seed
    train, valid = int(0.8*len(data)), int(0.1*len(data))
    train_x, train_y = data[:train][:, 1], data[:train][:, 0]
    valid_x, valid_y = data[train:-valid][:, 1], data[train:-valid][:, 0]
    test_x, test_y = data[-valid:][:, 1], data[-valid:][:, 0]
   
    ### Report the statistic
    train_spam, valid_spam, test_spam = np.sum(train_y), np.sum(valid_y), np.sum(test_y)

    print(f"\tTrain\tValid\tTest\tTotal")
    print(f"Spam\t{train_spam}\t{valid_spam}\t{test_spam}\t{train_spam+valid_spam+test_spam}")
    print(f"Ham\t{len(train_y)-train_spam}\t{len(valid_y)-valid_spam}\t{len(test_y)-test_spam}\t{len(data)-train_spam-valid_spam-test_spam}")
    print(f"Number of training samples: {len(train_x)}")
    print(f"Number of validation samples: {len(valid_x)}")
    print(f"Number of testing samples: {len(test_x)}")


    ### Build tf-idf
    print(f"Extracting tf-idf for training set..., used training set for idf")
    vocab = dict()
    vocab_id = dict()
    tf = []
    ### Build vocabulary dict based on training set, and calculate tf of training set
    ### TF = #occurence in a doc/ total length of document (mail text)
    ### IDF = log (#docs a specific word appears in / #docs)
    for idx, i in enumerate(train_x):
        temp = dict()
        seen_in_same_doc = []
        for word in i:
            if word not in seen_in_same_doc:
                vocab[word] = vocab[word] + 1 if word in vocab else 1
                seen_in_same_doc.append(word)
            temp[word] = temp[word] +1 if word in temp else 1
        temp[0] = len(i)
        tf.append(temp)
    ### Build id -> word based on training set
    for word, _ in vocab.items():
        vocab_id[word] = len(vocab_id)
    
    ### Calculate tf-idf
    total_docs = len(train_x)       ### Total number of docs in idf calculation
    train_x = build_tf_idf(tf, vocab, vocab_id, total_docs)
    print(f"Extracting tf-idf for valiation set...")
    tf_valid_x = calculate_tf(valid_x)          ### Calculate tf for validation
    valid_x = build_tf_idf(tf_valid_x, vocab, vocab_id, total_docs)
    print(f"Extracting tf-idf for test set...")
    tf_test_x = calculate_tf(test_x)            ### Calculate tf for testing
    test_x = build_tf_idf(tf_test_x, vocab, vocab_id, total_docs)
    print(train_x.shape, valid_x.shape, test_x.shape)


    ###----- Build a logistic regression classifier -----###
    ### Innitialize weight and bias
    w = np.zeros(shape=(1, train_x.shape[1]), dtype = np.float32)
    b = 1

    ### Gradient descent/SGD/Mini-batch-GD, un-commented the one wish to use to train
    lr, penalty, batch_size, epoch, method = 0.02, 0.00001, 1, 100, 1       ### Setting I used for SGD
    #lr, penalty, batch_size, epoch, method = 0.05, 0.00001, 16, 200, 0        ### Setting I used for MiniGD
    #lr, penalty, batch_size, epoch, method = 0.9, 0.000001, len(train_x), 500, 0    ### Setting I used for GD
    ###

    if method == 0 and batch_size < len(train_x):
        print(f"Using Mini-batch GD with batch_size = {batch_size}...")
    elif method == 1:
        print(f"Using SGD...")
    else:
        print(f"Using GD...")
    train_y = train_y.reshape(len(train_y), 1)
    train_y = np.array(train_y, dtype = int)
    valid_y = valid_y.reshape(len(valid_y), 1)
    valid_y = np.array(valid_y, dtype = int)
    valid_loss, w, b = GD_MiniGD_SGD(w, b, lr, train_x, train_y, epoch, batch_size, method, valid_x, valid_y, penalty)  # Method = 1: SGD; Method = 0: GD/MiniGD
    print(f"Train with learning rate= {lr}, lambda hyperparameter for L2 regularization= {penalty}, and epoch = {epoch}.")
    ### Plot validation loss after each epoch
    # plt.plot(valid_loss)
    # plt.show()

    # Save the model and artifacts
    print("Saving the model and artifacts...")
    with open("artifacts/w.pkl", "wb") as f:
        pickle.dump(w, f)
    with open("artifacts/b.pkl", "wb") as f:
        pickle.dump(b, f)
    with open("artifacts/vocab_id.pkl", "wb") as f:
        pickle.dump(vocab_id, f)
    with open("artifacts/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("artifacts/total_docs.pkl", "wb") as f:
        pickle.dump(total_docs, f)
    print("Model and artifacts saved.")


    

    


    
