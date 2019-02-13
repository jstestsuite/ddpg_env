#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import boost_python_catkin_example.mycpplib as cpp
import rospy
import tensorflow as tf
from nav_msgs.msg import Odometry
from move_base_msgs.msg import *
from geometry_msgs.msg import Twist
import math
from tensorflow import keras


#allows starting a roscpp node from python
from moveit_ros_planning_interface._moveit_roscpp_initializer import roscpp_init
roscpp_init('node_name', [])


vel_msg = Twist()


def talker():
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    vel_msg = Twist()
    prep_ml()
 

    pub.publish(vel_msg)

# this is the equivalent to `main()` in c++
if __name__ == '__main__':
    #cpp.hello()


	t = cpp.World()
	#//t.set("bom dia!")
	#print (t.greet())

	t.step([1.0, 2.0, 3.1])
	#test = t.greet()
	#//print test[1]
	t.step([3.0,4.0,5.0])
	test = t.greet()
	x = 0
	while test[x]:
	
		print test[x]
		x=x+1
	#//print (test)

        rospy.init_node('talker', anonymous=True) 

	
   
    	rate = rospy.Rate(10) # 10hz 

	t.pause_proxy()
	#rate.sleep()
	#rate.sleep()
	#rate.sleep()
	t.unpause_proxy()
	#rate.sleep()
	#rate.sleep()
	#rate.sleep()
	print "Reset going?"
	
	t.reset()
	print "reset"
    	rate = rospy.Rate(2) # 10hz
    	while not rospy.is_shutdown():    
		#t.chat() 
		t.step([3.0,4.0,5.0])   
		test = t.greet()
		print "one done"
		x = 0
	        while test[x]:
		
		  print test[x]
		#print test[1]
		  x=x+1
        	rate.sleep()

    #cpp.hello()
