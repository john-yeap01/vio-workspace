#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

static const std::string WINDOW_NAME = "ROS cv_bridge Image Viewer";

// Callback: called every time an Image message arrives
void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    // Convert ROS Image message -> OpenCV cv::Mat (BGR8)
    cv_bridge::CvImageConstPtr cv_ptr;

    try {
        cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // Display the image
    cv::imshow(WINDOW_NAME, cv_ptr->image);

    // Need waitKey() for OpenCV window to update
    int key = cv::waitKey(1);
    if (key == 27) {  // ESC
        ROS_INFO("ESC pressed in viewer window, shutting down.");
        ros::shutdown();
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_sub_node");
    ros::NodeHandle nh;

    // Name of the topic to subscribe to
    // For D435i color it’s usually: /camera/color/image_raw
    std::string topic_name;
    nh.param<std::string>("image_topic", topic_name, std::string("/camera/color/image_raw"));

    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);

    // Subscribe to the Image topic
    ros::Subscriber sub = nh.subscribe(topic_name, 1, imageCallback);

    ROS_INFO_STREAM("Subscribed to image topic: " << topic_name);

    ros::spin();

    cv::destroyWindow(WINDOW_NAME);
    return 0;
}
