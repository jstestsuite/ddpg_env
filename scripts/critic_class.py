
import rospy
import os
import json
import numpy as np
import random
import time
import sys

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Input

from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

class Critic():
    def __init__(self, sess, state_size, TAU, LEARNING_RATE):
	self.state_size = state_size
	self.sess = sess
	self.TAU = TAU
 	self.LEARNING_RATE = LEARNING_RATE
        self.sess.run(tf.global_variables_initializer())


	
	K.set_session(sess)

        #Now create the model
        self.model, self.state, self.action = self.buildCritic(state_size)  
        self.target_model,  self.target_state, self.target_action = self.buildCritic(state_size)  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
	self.sess.run(tf.initialize_all_variables())

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
	for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
})[0]

    def buildCritic(self, state_size):
        s = keras.layers.Input(shape=(state_size,), name='state')
        s1 = Dense(512, activation='relu')(s)
        a = keras.layers.Input(shape=(2,), name='action')
        x = keras.layers.concatenate([s1, a])
        x = Dense(512, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        q_val= Dense(1, activation='linear', name='main_output')(x)
        critic = Model(inputs=[s, a], outputs=[q_val])

        critic.compile(optimizer=tf.train.AdamOptimizer(), loss='mse')
        return critic, s, a
