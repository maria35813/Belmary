import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy.random
import root_numpy
import ROOT
import imp
import pandas as pd
# import bnn_smC.py
from keras.callbacks import History
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras import regularizers
from keras.layers import Input, Dense, Dropout, Activation
from keras.optimizers import SGD
# from keras.losses import logcosh
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from keras import metrics


def ReadSample(fileName, dim):
    inputVars = []
    for index in range(dim):
        varName = "var" + str(index + 1)
        inputVars.append(varName)
    # print inputVars
    rfile = ROOT.TFile(fileName)
    tree = rfile.Get('Vars')
    array = root_numpy.tree2array(tree)
    data_array = array[inputVars]
    data_df = pd.DataFrame(data_array)

    weights_array = array["weight"]
    weights_df = pd.DataFrame(weights_array)
    # print "Weights"
    # print weights_df
    target_array = array["target"]
    target_df = pd.DataFrame(target_array)
    input_data = data_df.to_numpy()
    # weights_array = np.genfromtxt('BNN_weight.txt')
    input_weight = weights_array
    input_target = target_array

    # print input_data,input_weight,input_target
    return input_data, input_target, input_weight


def ReadData(trainFile, examFile, dim):
    # train_split_labels
    data_train, labels_train, weights_train = ReadSample(trainFile, dim)
    data_test, labels_test, weights_test = ReadSample(examFile, dim)
    #	print data_train.shape,labels_train.shape,weights_train.shape,data_test.shape,labels_test.shape,weights_test.shape,train_split_labels.shape,test_split_labels.shape

    return data_train, labels_train, weights_train, data_test, labels_test, weights_test


def showHistory(history, model_name):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    # plt.plot(history.history['mean_squared_error'])
    # plt.plot(history.history['val_mean_squared_error'])
    plt.title('model mse')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig(task_name+'_Acc.png')
    #	lt.show()
    plt.tight_layout()
    # plt.figure(1)
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.tight_layout()
    plt.savefig(model_name + '_loss.png')


def createModel(dim):
    model = Sequential()
    inputs = Input(shape=(dim,))
    l1 = Dense(100, kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.001), activation='relu')(
        inputs)  #
    #b1 = BatchNormalization()(l1)
    d1 = Dropout(0.2)(l1)
    #

    l2 = Dense(100, kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.001), activation='relu')(d1)  #
    #b2 = BatchNormalization()(l2)
    d2 = Dropout(0.2)(l2)
    #

    l3 = Dense(100, kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.001), activation='relu')(d2)  #
    #b3 = BatchNormalization()(l3)
    d3 = Dropout(0.2)(l3)
    #

    l4 = Dense(150, kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.001), activation='relu')(d3)  #
    #b4 = BatchNormalization()(l4)
    d4 = Dropout(0.2)(l4)

    out = Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid')(d4)
    model = Model(inputs=inputs, outputs=out)

    from keras.optimizers import Adam, SGD
    adam = Adam(lr=0.003)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['mean_squared_error'])

    return model


def ShowPredict(features_train, labels_train, weights_train, features_test, labels_test, weights_test, model,
                model_name):
    labels_train_backgrount = labels_train[np.where(labels_train < 1.0)]
    labels_train_signal = labels_train[np.where(labels_train >= 1.0)]
    labels_test_backgrount = labels_test[np.where(labels_test < 1.0)]
    labels_test_signal = labels_test[np.where(labels_test >= 1.0)]

    train_len = len(labels_train_signal)
    test_len = len(labels_test_signal)
    predict_test = model.predict(features_test)
    predict_train = model.predict(features_train)

    predict_signal_test = predict_test[:test_len]
    predict_background_test = predict_test[test_len:]
    weight_signal_test = weights_test[:test_len]
    weight_background_test = weights_test[test_len:]
    # predict_train = model.predict(features_train)
    predict_signal_train = predict_train[:train_len]
    predict_background_train = predict_train[train_len:]
    weight_signal_train = weights_train[:train_len]
    weight_background_train = weights_train[train_len:]

    # roc_bnn = roc_auc_score(labels_test,bnn_predict,sample_weight = bnn_weights)
    roc_test = roc_auc_score(labels_test, predict_test, sample_weight=weights_test)
    roc_train = roc_auc_score(labels_train, predict_train, sample_weight=weights_train)
    # title = 'roc_auc='+str(roc)+' roc_auc_bnn='+str(roc_bnn)

    title = 'roc_auc_test=' + "{0:.3f}".format(round(roc_test, 3)) + ' roc_auc_train=' + "{0:.3f}".format(
        round(roc_train, 3))
    plt.clf()
    plt.hist(predict_signal_test, 20, histtype='step', color='g', label='test_signal',
             weights=weight_signal_test)
    plt.hist(predict_background_test, 20, histtype='step', color='y', label='test_background',
             weights=weight_background_test)
    plt.hist(predict_signal_train, 20, histtype='step', color='r', linestyle='--', label='train_signal',
             weights=weight_signal_train)
    plt.hist(predict_background_train, 20, histtype='step', color='b', linestyle='--',
             label='train_background', weights=weight_background_train)
    plt.legend(loc="upper center")
    plt.xlabel(title)
    plt.tight_layout()
    plt.savefig(model_name + '_Discr.pdf')
    plt.savefig(model_name + '_Discr.png')
    plt.clf()

    fpr_bnn, tpr_bnn, _ = roc_curve(labels_train, predict_train, sample_weight=weights_train)
    fpr_dnn, tpr_dnn, _ = roc_curve(labels_test, predict_test, sample_weight=weights_test)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_bnn, tpr_bnn, 'r--', label='train')
    plt.plot(fpr_dnn, tpr_dnn, 'b-', label='test')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(model_name + '_ROC.pdf')
    plt.savefig(model_name + '_ROC.png')

    return predict_test


# labels_test = np.array([1.0]);
if __name__ == "__main__":
    trainFile = 'bnn_sm_complete_trainFile_191214_141938_BCA.root'
    examFile = 'bnn_sm_complete_examFile_191214_154439_CXJ.root'
    dim = 43

    model_name = 'exampleSM'
    #  load data
    features_train, labels_train, weights_train, features_test, labels_test, weights_test = ReadData(trainFile, examFile, dim)
    #  create model
    model = createModel(dim)
    # train
    history = model.fit(features_train, labels_train, epochs=100, batch_size=len(labels_train), shuffle=False, validation_data=(features_test, labels_test, weights_test), sample_weight=weights_train, )
    #  save model
    model.save(model_name + '.h5')
    # results
    showHistory(history, model_name)
    ShowPredict(features_train, labels_train, weights_train, features_test, labels_test, weights_test, model, model_name)

# predict = model.predict(features_test,batch_size=100000)


# np.set_printoptions(threshold=np.nan)
# print predict.shape
# print predict

# n, bins, patches = plt.hist(predict, 100, normed=1, facecolor='green')