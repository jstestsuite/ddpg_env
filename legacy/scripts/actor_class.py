
import rospy
import os
import json
import numpy as np
import random
import time
import sys

from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Input
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import backend

class Actor():
    def __init__(self, sess, state_size, TAU, LEARNING_RATE):
	self.state_size = state_size
	self.sess = sess
	self.TAU = TAU
 	self.LEARNING_RATE = LEARNING_RATE

	backend.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.buildActor(state_size)   
        self.target_model, self.target_weights, self.target_state = self.buildActor(state_size) 
        self.action_gradient = tf.placeholder(tf.float32,[None, 2])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def buildActor(self, state_size):
        state = keras.layers.Input(shape=(state_size,), name='main_input')


        x = Dense(512, activation='relu')(state)
        x = Dense(512, activation='relu')(x)
        x = Dense(512, activation='relu')(x)


        lin_vel = Dense(1, activation='sigmoid', name='lin_vel')(x)
        ang_vel = Dense(1, activation='tanh', name='ang_vel')(x)
        output = keras.layers.concatenate([lin_vel, ang_vel])

        actor = Model(inputs=[state], outputs=[output])

        

        return actor, actor.trainable_weights, state

