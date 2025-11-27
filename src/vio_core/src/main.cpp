// Point cloud registration from 2 angles using ROS PointCloud2
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>

#include <iostream>

using PointT = pcl::PointXYZ;
using CloudT  = pcl::PointCloud<PointT>;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "icp_two_views_node");
    ros::NodeHandle nh;

    // Topic published by realsense2_camera for point clouds.
    // Check with: rostopic list | grep points
    std::string cloud_topic = "";
    nh.getParam("cloud_topic", cloud_topic);  // optional param override

    std::cout << "Using point cloud topic: " << cloud_topic << std::endl;
    std::cout << "Make sure realsense2_camera is running." << std::endl;

    // 1) Wait for first point cloud
    std::cout << "\nMove the camera to pose 1 and press ENTER..." << std::endl;
    std::cin.get();

    std::cout << "Waiting for first cloud..." << std::endl;
    sensor_msgs::PointCloud2::ConstPtr msg1 =
        ros::topic::waitForMessage<sensor_msgs::PointCloud2>(cloud_topic, nh, ros::Duration(10.0));

    if (!msg1)
    {
        std::cerr << "Did not receive first point cloud within timeout." << std::endl;
        return 1;
    }

    CloudT::Ptr cloud1(new CloudT);
    pcl::fromROSMsg(*msg1, *cloud1);
    std::cout << "Cloud1 has " << cloud1->size() << " points." << std::endl;

    // 2) Wait for second point cloud
    std::cout << "\nNow move the camera to pose 2 and press ENTER..." << std::endl;
    std::cin.get();

    std::cout << "Waiting for second cloud..." << std::endl;
    sensor_msgs::PointCloud2::ConstPtr msg2 =
        ros::topic::waitForMessage<sensor_msgs::PointCloud2>(cloud_topic, nh, ros::Duration(10.0));

    if (!msg2)
    {
        std::cerr << "Did not receive second point cloud within timeout." << std::endl;
        return 1;
    }

    CloudT::Ptr cloud2(new CloudT);
    pcl::fromROSMsg(*msg2, *cloud2);
    std::cout << "Cloud2 has " << cloud2->size() << " points." << std::endl;

    if (cloud1->empty() || cloud2->empty())
    {
        std::cerr << "One of the clouds is empty, aborting." << std::endl;
        return 1;
    }

    // Optional: you can downsample or crop here if you want

    // 3) Run ICP: align cloud2 to cloud1
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setInputSource(cloud2);   // moving
    icp.setInputTarget(cloud1);   // fixed

    icp.setMaximumIterations(50);
    icp.setMaxCorrespondenceDistance(0.05); // 5 cm
    icp.setTransformationEpsilon(1e-8);
    icp.setEuclideanFitnessEpsilon(1e-8);

    CloudT::Ptr aligned(new CloudT);
    icp.align(*aligned);

    if (icp.hasConverged())
    {
        std::cout << "\nICP converged." << std::endl;
        std::cout << "Fitness score: " << icp.getFitnessScore() << std::endl;

        Eigen::Matrix4f T = icp.getFinalTransformation();
        std::cout << "\nEstimated transform (cloud2 -> cloud1):\n" << T << std::endl;
        std::cout << "\nThis 4x4 matrix is your camera motion estimate between the two views.\n";
    }
    else
    {
        std::cout << "ICP did not converge." << std::endl;
    }

    return 0;
}
