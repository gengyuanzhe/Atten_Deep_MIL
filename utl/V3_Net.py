import sys
import time
from random import shuffle
import numpy as np
import argparse
import tensorflow as tf

from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Input, Dense, Layer, Dropout, Conv2D, MaxPooling2D, Flatten, multiply, \
    GlobalAveragePooling2D
from .metrics import bag_accuracy, bag_loss
from .custom_layers import Mil_Attention, Last_Sigmoid, LastSoftmax


def cell_net(input_dim, args, use_mul_gpu=False):
    lr = args.init_lr
    weight_decay = args.init_lr
    momentum = args.momentum

    base_model = InceptionV3(weights=None, input_shape=input_dim, include_top=False)

    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    fc1 = Dense(512, activation='relu', kernel_regularizer=l2(weight_decay), name='fc1')(x)
    fc1 = Dropout(0.5)(fc1)
    fc2 = Dense(512, activation='relu', kernel_regularizer=l2(weight_decay), name='fc2')(fc1)
    fc2 = Dropout(0.5)(fc2)

    alpha = Mil_Attention(L_dim=128, output_dim=1, kernel_regularizer=l2(weight_decay), name='alpha',
                          use_gated=args.useGated)(fc2)
    x_mul = multiply([alpha, fc2])

    out = LastSoftmax(output_dim=2, name='softmax')(x_mul)
    #
    model = Model(inputs=[base_model.input], outputs=[out])

    if use_mul_gpu:
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy',
                               metrics=['categorical_accuracy'])
    else:
        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
        parallel_model = model

    return parallel_model
