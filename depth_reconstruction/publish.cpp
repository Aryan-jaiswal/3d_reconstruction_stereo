#include <ros/ros.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_publisher");
  ros::NodeHandle nh;

    cv_bridge::CvImage cv_imagel;
    cv_bridge::CvImage cv_imager;
    cv_imagel.image = cv::imread("/home/aryan/disparity/libelas/img/left1.png");
    cv_imager.image = cv::imread("/home/aryan/disparity/libelas/img/right1.png");
    cv_imagel.encoding = "bgr8";
    cv_imager.encoding = "bgr8";
    sensor_msgs::Image ros_imagel,ros_imager;
    cv_imagel.toImageMsg(ros_imagel);
    cv_imager.toImageMsg(ros_imager);

  ros::Publisher publ = nh.advertise<sensor_msgs::Image>("camera/left_image", 1);
  ros::Publisher pubr = nh.advertise<sensor_msgs::Image>("camera/right_image", 1);
  ros::Rate loop_rate(100);

  while (nh.ok()) 
  {
    publ.publish(ros_imagel);
    pubr.publish(ros_imager);
    loop_rate.sleep();
  }
  return 0;
}
