#include <semantic_slam/api_class/raw_wrapper.h>

int main(int argc, char** argv){
    ros::init(argc, argv, "test_semantic_slam");
    RawWrapper wrapper;
    while (ros::ok())
    {
        ros::spinOnce();
        /* code */
    }
}