//this code subscribe to odometry msg and convert the orientation 
//from quaternion to euler angle and also get the x, y position of
//the robot and publish to a message name ""Robot_State""
//the message contain 3 values which are float32

#include "ros/ros.h"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/OccupancyGrid.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <string>
#include <vector>

using namespace message_filters;
using namespace nav_msgs;

typedef uint32_t index_t;
typedef int16_t coord_t; 

struct Pos
{
    float x;
    float y;
};


struct Info
{
    int c;
    Pos p;
    double d;
};


struct Quaternion{
    double w, x, y, z;
};

struct EulerAngle{
    double roll, pitch, yaw;
};

struct State
{
    double x_pos, y_pos, theta;
};


//helper function to convert quaternion to euler angles
//referrence "https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles"
EulerAngle QuatToEuler(const Quaternion& q);

State currentState(const nav_msgs::Odometry::ConstPtr& msg);

void Callback(const nav_msgs::Odometry::ConstPtr& q ,
                        const nav_msgs::OccupancyGrid::ConstPtr& grid);


int main(int argc, char** argv){
    ros::init(argc, argv, "ref_point");
    ros::NodeHandle nh;

    message_filters::Subscriber<Odometry> odom(nh, "odom", 5);
    message_filters::Subscriber<OccupancyGrid> grid(nh, "map", 1000);
    //TimeSynchronizer<Odometry, OccupancyGrid> sync(odom, grid, 2000);

    typedef sync_policies::ApproximateTime<Odometry, OccupancyGrid> Mysyncpolicy;

    Synchronizer<Mysyncpolicy> sync(Mysyncpolicy(2000), odom, grid);

    ros::Rate loop_rate(100);

    while (ros::ok()){
        sync.registerCallback(boost::bind(&Callback, _1, _2));
        ros::spinOnce();
        loop_rate.sleep();
    }
    
    
    return 0;
}

EulerAngle QuatToEuler(const Quaternion& q){
    EulerAngle angles;

    //ROLL
    double sinr_cosp = 2*(q.w * q.x + q.y * q.z);
    double cosr_cosp = 1 - 2*(q.x * q.x + q.y * q.y);
    angles.roll = std::atan2(sinr_cosp, cosr_cosp);

    //PITCH
    double sinp = 2 * (q.w * q.y + q.y * q.y);
    if(std::abs(sinp) >= 1){
        angles.pitch = std::copysign(M_PI / 2, sinp);
    }else{
        angles.pitch = std::asin(sinp);
    }

    //YAW
    double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 2 * (q.y * q.y + q.z * q.z);
    angles.yaw = std::atan2(siny_cosp, cosy_cosp);

    return angles;

}

State currentState(const nav_msgs::Odometry::ConstPtr& msg){

    double x = msg->pose.pose.position.x;
    double y = msg->pose.pose.position.y;

    Quaternion quat;

    quat.w = msg->pose.pose.orientation.w;
    quat.x = msg->pose.pose.orientation.x;
    quat.y = msg->pose.pose.orientation.y;
    quat.z = msg->pose.pose.orientation.z;

    EulerAngle angles = QuatToEuler(quat);

    State q;

    q.theta = angles.yaw;
    q.x_pos = x;
    q.y_pos = y;

    return q;
}


void Callback(const nav_msgs::Odometry::ConstPtr& q ,
                        const nav_msgs::OccupancyGrid::ConstPtr& grid){
    index_t index = grid->data.size()-1;

    State q_i = currentState(q);


    //grid information
    int height = grid->info.height;
    int width = grid->info.width;
    float resolution = grid->info.resolution;

    //grid origin position in world frame
    int x_o = grid->info.origin.position.x;
    int y_o = grid->info.origin.position.y;

    std::vector<Info> g(grid->data.size());


    double smallest = 1000;
    //convert original map information to a vector containing 
    //a cell position and world position
    for(int i = 0; i<grid->data.size(); i++){
        div_t result = div(i, width);
        coord_t x = result.quot;
        coord_t y = result.rem;

        //convert to position based on grid info
        Pos p;
        p.x = x*resolution+x_o;
        p.y = y*resolution+y_o;


        Info information;
        information.p = p;
        information.c = grid->data[i];


        //calcualte the distance of each point to the state of the robot
        if(information.c > 90){
            double dx = abs(p.x - q_i.x_pos);
            double dy = abs(p.y - q_i.y_pos);
            double ref_d = sqrt(dx*dx + dy*dy);
            information.d = ref_d;
        }else{
            information.d = -1;
        }

        g[i] = information;
        if(g[i].d < smallest && g[i].d != -1){
            smallest = g[i].d;
        }
    }

    
    ROS_INFO("the closest point is [%f]", smallest);
}