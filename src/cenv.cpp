






#include <iostream>
#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/python/extract.hpp>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose.h>
#include <std_srvs/Empty.h>
#include <gazebo_msgs/SpawnModel.h>
#include <gazebo_msgs/DeleteModel.h>
#include <tf/tf.h>
#include <geometry_msgs/Pose.h>
#include <nav_msgs/Odometry.h>
#include <vector>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/numpy.hpp>
#include <sensor_msgs/LaserScan.h>
#include <std_msgs/String.h>
#include <algorithm> 
#include <cstdlib>
#include <fstream>
#include <tf2/LinearMath/Quaternion.h>

//#include <condition_variable> // std::condition_variable



//using namespace std;
  //using namespace std::placeholders;  
//namespace p = boost::python;
//namespace np = boost::python::numpy;
typedef std::vector<float> MyList;
class World
{
private:
	
    float goal_angle =0.0;
    float heading = 0.0;
    float min_range = 0.15;
    float goal_x = 0.0;
    float goal_y = 0.0;
    std::string mMsg;
    int count = 0;
    float goal_distance = 0.0;
    int old_count = 0;
    float past_distance = 0.0;
    float pos_x = 0.0;
    float pos_y = 0.0;
    geometry_msgs::Twist msg;
    geometry_msgs::Twist msg_empty;
    ros::NodeHandle n;
    MyList action = MyList(2);
    MyList past_action = MyList(2);
    MyList state = MyList(15);
    bool done = false;
    bool get_goalbox = false;
    float distance = 0.0;
    float reward = 0;
    //float past_action = 0;
    geometry_msgs::Pose post;
    MyList goal_x_list = {0.6, 1.9, 0.5, 0.2, -0.8, -1, -1.9, 0.5, 2, 0.5, 0, -0.1, -2};
    MyList goal_y_list = {0, -0.5, -1.9, 1.5, -0.9, 1, 1.1, -1.5, 1.5, 1.8, -1, 1.6, -0.8};
   // int rand();
    double yaw = 0;
    bool init_goal = true;
    




public:    

 //nav_msgs::Odometry::ConstPtr msg;

    ros::Publisher chatter_pub = n.advertise<geometry_msgs::Twist>("cmd_vel", 5);     
    //ros::Subscriber odom_sub = n.subscribe("odom", 1, std::bind(getOdometry, &msg));
 ros::Subscriber sub_t = n.subscribe("odom", 1, &World::getOdometry, this);
 //ros::Subscriber scan = n.subscribe("scan", 1, &World::getScan, this);
sensor_msgs::LaserScanConstPtr test3 = ros::topic::waitForMessage<sensor_msgs::LaserScan>("scan");



//SOMETHING HERE       
 //self.respawn_goal = Respawn()


    ros::ServiceClient reset_proxy = n.serviceClient<std_srvs::Empty>("gazebo/reset_simulation");   
    ros::ServiceClient pauseGazebo = n.serviceClient<std_srvs::Empty>("/gazebo/pause_physics");
    ros::ServiceClient unpauseGazebo = n.serviceClient<std_srvs::Empty>("/gazebo/unpause_physics");
    std_srvs::Empty emptySrv;
    //pauseGazebo.call(emptySrv);
    //msg_empty.linear.x = 0.0;
   // msg_empty.angular.z = 0.0;




//unpause_proxy
    void unpause_proxy() {
        unpauseGazebo.call(emptySrv);
    }

//pause_proxy
    void pause_proxy() {
        pauseGazebo.call(emptySrv);
    }


//getGoaldistance
    float getGoalDistance() {
	goal_distance = hypot(post.position.x - pos_x, post.position.y-pos_y);
	state[13] = goal_distance;

    }

//getOdometry

    void getOdometry(const nav_msgs::Odometry::ConstPtr& msg) {
	pos_x = msg->pose.pose.position.x;
	pos_y = msg->pose.pose.position.y;

	geometry_msgs::Quaternion q = msg->pose.pose.orientation;
	double siny_cosp = +2.0 * (q.w * q.z + q.x * q.y);
	double cosy_cosp = +1.0 - 2.0 * (q.y * q.y + q.z * q.z);  
	yaw = atan2(siny_cosp, cosy_cosp);
	goal_angle = atan2(post.position.x - pos_x, post.position.y-pos_y);
	heading = goal_angle - yaw;

	if (heading > 3.14)
            heading -= 2 * 3.14;

        else if (heading < -3.14)
            heading += 2 * 3.14;
	state[12] = heading;
    }

//getScan
    void getScan(const sensor_msgs::LaserScan::ConstPtr& msg) {
	for (int i =0; i <10; ++i) {
	    state[i] = msg->ranges[i];

	    if (state[i] > 3.5)
		state[i] = 3.5;	        	   	
	}
	    count = msg->header.seq;
    }

//getstate
    void getState() {

	for (int i =0; i<10; ++i) {
		//std::min_element(state.begin(), state.end());
	    if (state[i] < min_range) 
		{done = true;
		//state[12] = state[i];
		//state[13] = 5;
		break;
		}
	}
	getGoalDistance();
	//distance = hypot(goal_x - pos_x, goal_y-pos_y);	    
	if (goal_distance < 0.2) 
	    get_goalbox = true;
    }

//setreward
    void setReward() {
	
        float distance_rate = past_distance - goal_distance;
	reward = (500 * distance_rate);
	past_distance = goal_distance;
	
	if (done) {
	    ROS_INFO("Collision!");
	    reward = -150;
	    chatter_pub.publish(msg_empty);
	    reset();
	    done = false;
	}
	if (get_goalbox) {
	    ROS_INFO("Goal!");
	    reward = 200;
	    deleteGoal();
	    spawnGoal();

	    ///// new goal and get position?
	    /////////////////////////////////
	    /////////////////////////////////
	    getGoalDistance();
	    get_goalbox = false;
	}

	//current_distance = 

    }

//step
    MyList step(boost::python::list msgs, boost::python::list msgs2) {
        long l = len(msgs);
        long l2 = len(msgs2);

        for (long i = 0; i<l; ++i) {

            action[i] = boost::python::extract<float>(msgs[i]);
	}	
        for (long i = 0; i<l2; ++i) {

            past_action[i] = boost::python::extract<float>(msgs2[i]);
	}	
            if (action[1] > 1.0) {
                action[1] = 1.0;
	    }
	    else if (action[1] < -1.0) {
		action[1] = -1.0;
	    }
	state[10] = past_action[0];
	state[11] = past_action[1];
	msg.linear.x = action[0]*0.3;
	msg.angular.z = action[1];
	while (test3 == NULL)
	{
	} 
	
	for (int i =0; i <10; ++i) {
	    state[i] = test3->ranges[i];

	    //if (state[i] > 3.5)
		//state[i] = 3.5;	        	   	
	}
	//state[13] = test3->header.seq;


	test3 = ros::topic::waitForMessage<sensor_msgs::LaserScan>("scan");

	chatter_pub.publish(msg);
	//////////////////////////////////////////
        //need to make so this function can only run once per laser scan...
	//currently it will not publis until there is a new laser scan, I need it to run only once per scan 
	///////////////////////////////////////////
	//////////////////////////////////////////
	/////////////////////////////////////////

	old_count = count;
	getState();
	setReward();
	state[14] = reward;

	return state;

    }

//reset
    void reset() {
	ros::service::waitForService("gazebo/reset_simulation");
	try {
	    reset_proxy.call(emptySrv);
	}
	catch (int e) {
	    ROS_INFO("Unable to reset Gazebo. Service call failed");
	}
	
	if (init_goal == true) {
	    spawnGoal();
	    init_goal = false;
	}
	
	getGoalDistance();
	
    }

    void spawnGoal() {

    	ros::service::waitForService("gazebo/spawn_sdf_model");
    	gazebo_msgs::SpawnModel model;
                
    	ros::ServiceClient spawn_model = n.serviceClient<gazebo_msgs::SpawnModel>("gazebo/spawn_sdf_model");
    	std::ifstream file("/home/justin/ros_test/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_square/goal_box/model.sdf");

  	std::string line;
 
   	while(!file.eof()) // Parse the contents of the given urdf in a string
    	{
      	    std::getline(file,line);
            model.request.model_xml+=line;
    	}
  	file.close();
	//int temp = rand();
     	post.position.x = goal_x_list[rand()%13];
    	post.position.y = goal_y_list[rand()%13];
   
  	model.request.model_name="goal";
  	model.request.reference_frame="world";
  	model.request.initial_pose = post;
  	spawn_model.call(model); //Call the service

	getGoalDistance();

    }


    void deleteGoal() {
	
	ros::service::waitForService("gazebo/delete_model");
    	gazebo_msgs::DeleteModel model;
	ros::ServiceClient delete_model = n.serviceClient<gazebo_msgs::DeleteModel>("gazebo/delete_model");
	model.request.model_name = "goal";
	delete_model.call(model);

	//send an 0 cmd_vel command
	//chatter_pub.publish(msg_empty);

    }



/*



 get position
check model
 def deleteModel(self):
        while True:
            if self.check_model:
                rospy.wait_for_service('gazebo/delete_model')
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                del_model_prox(self.modelName)
                break
            else:
                pass


*/


//used for learning
    void many(boost::python::list msgs) {
        long l = len(msgs);
        std::stringstream ss;
        for (long i = 0; i<l; ++i) {
            if (i>0) ss << ", ";
            std::string s = boost::python::extract<std::string>(msgs[i]);
            ss << s;
        }
        mMsg = ss.str();
    }

//used for learing
    void chat() {
    	msg.linear.x = pos_x;
    	msg.angular.z = 5; 
        chatter_pub.publish(msg);
    }


//returns value
    MyList greet() { return state; }




};


///////////////////////////////////////////////////////////////////////////////////

//namespace bn = boost::python::numpy;


///////////////////////////////////////////////////////////////////////////////////////////
using namespace boost::python;

BOOST_PYTHON_MODULE(cenv)
{

    //boost::python::numpy::initialize();

   //def("testf", testf);

    class_<MyList>("MyList")
      .def(vector_indexing_suite<MyList>() );

    class_<World>("World")


        .def("greet", &World::greet)
        .def("pause_proxy", &World::pause_proxy)
        .def("unpause_proxy", &World::unpause_proxy)
        .def("reset", &World::reset)    
        .def("step", &World::step)
        .def("many", &World::many)
        .def("chat", &World::chat)
        .def("spawnGoal", &World::spawnGoal)
        .def("deleteGoal", &World::deleteGoal)
    ;
};
