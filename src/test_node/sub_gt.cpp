#include <ros/ros.h>
#include <gazebo_msgs/ModelStates.h>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Core>
using namespace std;


ros::Subscriber sub_gt;
Eigen::Matrix4d init_pose = Eigen::Matrix4d::Zero();
void stateCallback(const gazebo_msgs::ModelStatesConstPtr& states){
    string traj_filename = "gt_gazebo.txt";
    string folder = "/home/nuninu98/";
    ofstream traj_file(folder + traj_filename, std::ios_base::app);
    Eigen::Quaterniond q(states->pose.back().orientation.w, states->pose.back().orientation.x, states->pose.back().orientation.y, states->pose.back().orientation.z);
    string time = to_string(ros::Time::now().toSec());

    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3, 3>(0, 0) = q.toRotationMatrix();
    pose(0, 3) = states->pose.back().position.x;
    pose(1, 3) = states->pose.back().position.y;
    pose(2, 3) = states->pose.back().position.z;
    
    if(init_pose == Eigen::Matrix4d::Zero()){
        init_pose = pose;
        traj_file<<time<<" "<< 0.0<<" "<<0.0<<" "<<0.0<<endl;
        return;
    }
    
    Eigen::Matrix4d rel_pose = init_pose.inverse() * pose;
    traj_file<<time<<" "<< rel_pose(0, 3)<<" "<<rel_pose(1, 3)<<" "<<rel_pose(2, 3)<<endl;
}

int main(int argc, char** argv){
    ros::init(argc, argv, "sub_gt");
    ros::NodeHandle nh;
    sub_gt = nh.subscribe("/gazebo/model_states", 1, stateCallback);
    ros::spin();
}