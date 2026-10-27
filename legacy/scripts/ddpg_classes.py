
import rospy
import os
import json

import random
import time
import sys
import numpy as np
import math
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
from src.turtlebot3_ddpg.environment_stage_1 import Env
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense, Dropout, Activation, Input

from tensorflow import keras

from tensorflow.keras.models import Model


from tensorflow.keras import backend

from actor_class import Actor
from critic_class import Critic

EPISODES = 500
GAMMA = 0.99
TAU = 0.001
LRA = 0.0001    #Learning rate for Actor
LRC = 0.001 #Lerning rate for Critic
state_dim = 14
batch_size = 32
max_step = 401
memory = deque(maxlen=1000000)

load_weight = True

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
	x[0] = abs(x[0])        
	self.x_prev = x
	
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
    


def appendMemory(memory, state, action, reward, next_state, done):
       return memory.append((state, action, reward, next_state, done))

#need to analyze
if __name__ == '__main__':
    epsilon = 1.0
    epsilon_decay = 0.99
    epsilon_min = 0.04
    rospy.init_node('ddpg_classes')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    train_indicator = False

    
    sess = tf.Session()
    backend.set_session(sess)

    actor = Actor(sess, state_dim, TAU, LRA)
    critic = Critic(sess, state_dim, TAU, LRC)

    result = Float32MultiArray()
    get_action = Float32MultiArray()
    past_action = np.zeros((1,2))


    if load_weight == True:
	print("Now we load the weight")
        try:
            actor.model.load_weights("actormodel.h5")
            critic.model.load_weights("criticmodel.h5")
            actor.target_model.load_weights("actormodel.h5")
            critic.target_model.load_weights("criticmodel.h5")
            print("Weight load successfully")
        except:
	    print("Cannot find the weight")

    env = Env(2)
    #env.unpause_proxy()


    global_step = 0
    start_time = time.time()
    print "Starting to train the models.  Success?'"
    print '...'
    print '...'
    print '...'
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(2))

    for e in range(EPISODES):
        done = False
        state = env.reset()
        score = 0
	
        for t in range(max_step):

	    loss = 0

            action = actor.model.predict(state.reshape(1, len(state))) + ( actor_noise()*epsilon/4)
	    
	    
	    #print "ACTION IS PRINTED"
            next_state, reward, done = env.step(action.flatten(), past_action.flatten())
	    #print len(state), "\n"
	    if (train_indicator):            
		appendMemory(memory, state, action, reward, next_state, done)
	    #### Start the training

	    	if len(memory) > batch_size:
            		env.pause_proxy()
            		mini_batch = random.sample(memory, batch_size)
			states = np.empty(([0,14]), dtype=np.float64)
			actions = np.empty(([0,2]), dtype=np.float64)
			rewards = np.empty(([0,1]), dtype=np.float64)
			next_states = np.empty(([0,14]), dtype=np.float64)
			dones = []
			for i in range(batch_size):
            			states = np.append(states, np.asarray(mini_batch[i][0]).reshape(1, states.shape[1]), axis = 0)
            			actions = np.append(actions, np.asarray(mini_batch[i][1]).reshape(1, actions.shape[1]), axis = 0)
            			rewards = np.append(rewards,np.asarray(mini_batch[i][2]).reshape(1, rewards.shape[1]), axis = 0)
            			next_states = np.append(next_states, np.asarray(mini_batch[i][3]).reshape(1, next_states.shape[1]), axis = 0)
            			dones.append(mini_batch[i][4])

#reshape?
	    		target_q = critic.target_model.predict([next_states, actor.target_model.predict(next_states)])

            	    	y_i = []
            	    	for k in range(batch_size):
		    
            	        #if dones[k]:
            	          #  y_i.append(rewards[k])
            	        #else:
			    y_i.append(rewards[k] + GAMMA * target_q[k])

			h = np.asarray(y_i)
			#print actions.shape
			#print states.shape
			#print h.shape
                	loss += critic.model.train_on_batch([states,actions], h) 
                	a_for_grad = actor.model.predict(states)
                	grads = critic.gradients(states, a_for_grad)
                	actor.train(states, grads)
                	actor.target_train()
			critic.target_train()
			env.unpause_proxy()
            	score += reward
            	state = next_state
	    	if np.mod(t, 100) ==0:
	    		print "Episode", e, "Step", global_step, "Action", action, "Reward", reward, "Loss", loss
            if t >= 400:
                rospy.loginfo("Time out!!")
                done = True

            if done:
                result.data = [score, loss]
                pub_result.publish(result)
		print "publishing reulsts"
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                rospy.loginfo('Ep: %d score: %.2f memory: %d  time: %d:%02d:%02d',
                              e, score, len(memory),  h, m, s)
               # param_keys = ['epsilon']
               # param_values = [epsilon]
               # param_dictionary = dict(zip(param_keys, param_values))
                break

            global_step += 1

        if np.mod(e, 10) == 0:
            if (train_indicator):
                print "Now we save model"
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print "TOTAL REWARD @ " + str(e) +"-th Episode  : Reward " + str(score)
        print "Total Step: " + str(global_step)
	print ""
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay



