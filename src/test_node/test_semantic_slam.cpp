#include <semantic_slam/api_class/semantic_slam.h>

int main(int argc, char** argv){
    ros::init(argc, argv, "test_semantic_slam");
    cout<<CV_VERSION<<endl;
    SemanticSLAM slam;
    while (ros::ok())
    {
        ros::spinOnce();
        /* code */
    }
    
}