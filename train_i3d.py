"""
This script is used for training the model.
"""
import os
import time
import keras
import numpy as np
from keras import backend as K
from datagenerator import VideoClasses, FramesGenerator
from model_i3d import Inception_Inflated3d, add_i3d_top


def layers_freeze(keModel):
    """
    Used for freezing the weights in the model to avoid training them.
    :param keModel: input keras model.
    :return: frozen keras model.
    """
    print("Freeze all %d layers in Model %s" % (len(keModel.layers), keModel.name))
    for layer in keModel.layers:
        layer.trainable = False

    return keModel


def layers_unfreeze(keModel):
    """
    Used for unfreezing the weights in the model for training them.
    :param keModel: input frozen keras model.
    :return: keras model.
    """
    print("Unfreeze all %d layers in Model %s" % (len(keModel.layers), keModel.name))
    for layer in keModel.layers:
        layer.trainable = True

    return keModel


def count_params(keModel: keras.Model):
    """
    Determines the number of parameters in the keras model.
    :param keModel: input keras model.
    :return: None
    """
    trainable_count = int(
        np.sum([K.count_params(p) for p in set(keModel.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(keModel.non_trainable_weights)]))

    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))

    return


def train_I3D_oflow_end2end():
    """
    Training the keras model.
    :return: None
    """
    sClassFile = "class.csv"
    sOflowDir = "Training_data"
    sModelDir = "model"

    diTrainTop = {
        "fLearn": 1e-3,
        "nEpochs": 5}

    diTrainAll = {
        "fLearn": 1e-4,
        "nEpochs": 1}

    nBatchSize = 4

    print("\nStarting I3D end2end training ...")
    print(os.getcwd())

    oClasses = VideoClasses(sClassFile)

    # Load training data
    path = os.path.join(sOflowDir, "train")
    genFramesTrain = FramesGenerator(path, nBatchSize, 40, 224, 224, 2, oClasses.liClasses)
    path = os.path.join(sOflowDir, "val")
    genFramesVal = FramesGenerator(path, nBatchSize, 40, 224, 224, 2, oClasses.liClasses)

    if (genFramesTrain):
        print("Generated training data.")
    if (genFramesVal):
        print("Generated validation data")

    # Load pretrained i3d model and adjust top layer
    print("Load pretrained I3D flow model ...")
    keI3DOflow = Inception_Inflated3d(
        include_top=False,
        weights='flow_imagenet_and_kinetics',
        input_shape=(40, 224, 224, 2))
    print("Add top layers with %d output classes ..." % oClasses.nClasses)
    keI3DOflow = layers_freeze(keI3DOflow)
    keI3DOflow = add_i3d_top(keI3DOflow, oClasses.nClasses, dropout_prob=0.5)
    if (keI3DOflow):
        print("Model loaded successfully")

    sLog = time.strftime("%Y%m%d-%H%M", time.gmtime()) + "-%s%03d-oflow-i3d" % ("ISL", 105)

    # Save the model
    os.makedirs(sModelDir, exist_ok=True)
    cpTopLast = keras.callbacks.ModelCheckpoint(filepath=sModelDir + "/" + sLog + "-top-last.h5", verbose=1,
                                                save_best_only=False, save_weights_only=False)
    cpTopBest = keras.callbacks.ModelCheckpoint(filepath=sModelDir + "/" + sLog + "-top-best.h5", verbose=1,
                                                save_best_only=True, save_weights_only=False)
    cpAllLast = keras.callbacks.ModelCheckpoint(filepath=sModelDir + "/" + sLog + "-entire-last.h5", verbose=1,
                                                save_weights_only=False, save_best_only=False)
    cpAllBest = keras.callbacks.ModelCheckpoint(filepath=sModelDir + "/" + sLog + "-entire-best.h5", verbose=1,
                                                save_best_only=True, save_weights_only=False)
    cbTensorBoard = keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1, update_freq='batch', write_grads=True,
                                                write_graph=True, write_images=True, batch_size=32)
    callbacks1 = [cpTopLast, cpTopBest, cbTensorBoard]
    callbacks2 = [cpAllBest, cpAllLast, cbTensorBoard]

    # Fit top layers
    print("Fit I3D top layers with generator: %s" % (diTrainTop))
    optimizer = keras.optimizers.Adam(lr=diTrainTop["fLearn"])
    keI3DOflow.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    count_params(keI3DOflow)

    keI3DOflow.fit_generator(
        generator=genFramesTrain,
        validation_data=genFramesVal,
        epochs=diTrainTop["nEpochs"],
        workers=4,
        use_multiprocessing=False,
        max_queue_size=8,
        verbose=1,
        callbacks=callbacks1)

    """
    Fit entire I3D model
    print("Finetune all I3D layers with generator: %s" % (diTrainAll))
    keI3DOflow = layers_unfreeze(keI3DOflow)
    optimizer = keras.optimizers.Adam(lr = diTrainAll["fLearn"])
    keI3DOflow.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    count_params(keI3DOflow)    
    
    keI3DOflow.fit_generator(
        generator = genFramesTrain,
        validation_data = genFramesVal,
        epochs = diTrainAll["nEpochs"],
        workers = 4,                 
        use_multiprocessing = False,
        max_queue_size = 8, 
        verbose = 1,
        callbacks=callbacks2)
    """

    return


train_I3D_oflow_end2end()
