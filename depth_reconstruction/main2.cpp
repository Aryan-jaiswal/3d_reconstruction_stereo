#include "StereoEfficientLargeScale.h"

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/ChannelFloat32.h>
#include <geometry_msgs/Point32.h>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <stdio.h>
#include <dynamic_reconfigure/server.h>
#include <fstream>
#include <ctime>
#include <octomap/octomap.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <octomap_ros/conversions.h>
//#include <opencv2/opencv.hpp>
//#include "elas.h"
#include "popt_pp.h"

#define CV_VERSION_NUMBER CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)


#ifdef _DEBUG
#pragma comment(lib, "opencv_viz"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_videostab"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_video"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_ts"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_superres"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_stitching"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_photo"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_ocl"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_objdetect"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_nonfree"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_ml"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_legacy"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_imgproc"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER"d.lib")
//#pragma comment(lib, "opencv_haartraining_engine.lib")
#pragma comment(lib, "opencv_gpu"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_flann"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_features2d"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_core"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_contrib"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_calib3d"CV_VERSION_NUMBER"d.lib")
#else
#pragma comment(lib, "opencv_viz"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_videostab"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_video"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_ts"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_superres"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_stitching"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_photo"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_ocl"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_objdetect"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_nonfree"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_ml"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_legacy"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_imgproc"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER".lib")
//#pragma comment(lib, "opencv_haartraining_engine.lib")
#pragma comment(lib, "opencv_gpu"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_flann"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_features2d"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_core"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_contrib"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_calib3d"CV_VERSION_NUMBER".lib")
#endif


#define baseline 12
#define focal 358
#define kMapResolution 0.2

using namespace cv;
using namespace std;

Mat XR, XT, Q, P1, P2;
Mat R1, R2, K1, K2, D1, D2, R;
Mat lmapx, lmapy, rmapx, rmapy, disp;
Vec3d T;
int debug = 0;
FileStorage calib_file;
Size out_img_size;
Size calib_img_size;

image_transport::Publisher dmap_pub;
ros::Publisher _m_octomap_pub;
Mat leftdpf;
StereoEfficientLargeScale * elas;

void generateDisparityMap(Mat& l, Mat& r, Mat& dmap) {
 
  if (l.empty() || r.empty()) 
    return;
  
  ////////////////////
  
  
  elas->process(l,r,leftdpf,100);
  leftdpf.convertTo(dmap, CV_8U, 1./8);
 \
  /////////////////////

}
void publishPointCloud(Mat& img_left, Mat& dmap) {
  
  Mat V = Mat(4, 1, CV_64FC1);
  Mat pos = Mat(4, 1, CV_64FC1);
  
  octomap_msgs::Octomap map_msg; 

  std::shared_ptr<octomap::OcTree> map_ptr = std::make_shared<octomap::OcTree>(kMapResolution);

  octomap::OcTree* tree = map_ptr.get();
  map_msg.header.stamp = ros::Time::now();
  map_msg.header.frame_id = "Hedwig";
  map_msg.binary = true;
  map_msg.id = "OcTree";
  map_msg.resolution = kMapResolution;
  
  
    /////
  for (int i = 0; i < img_left.cols; i++) {
    for (int j = 0; j < img_left.rows; j++) {
      int d = dmap.at<uchar>(j,i);
      // if low disparity, then ignore
      if (d < 2) {
        continue;
      }
      // V is the vector to be multiplied to Q to get
      // the 3D homogenous coordinates of the image point
      V.at<double>(0,0) = (double)(i);
      V.at<double>(1,0) = (double)(j);
      V.at<double>(2,0) = (double)d;
      V.at<double>(3,0) = 1.;
      pos = Q * V; // 3D homogeneous coordinate
      double X = pos.at<double>(0,0) / pos.at<double>(3,0);
      double Y = pos.at<double>(1,0) / pos.at<double>(3,0);
      double Z = pos.at<double>(2,0) / pos.at<double>(3,0);
      Mat point3d_cam = Mat(3, 1, CV_64FC1);
      point3d_cam.at<double>(0,0) = X;
      point3d_cam.at<double>(1,0) = Y;
      point3d_cam.at<double>(2,0) = Z;
      // transform 3D point from camera frame to robot frame
      Mat point3d_robot = XR * point3d_cam + XT;

      
      tree->updateNode(point3d_robot.at<double>(0,0), point3d_robot.at<double>(1,0), point3d_robot.at<double>(2,0), true);
    }
  }
  //tree->updateInnerOccupancy();
  octomap_msgs::binaryMapToMsg(*(tree), map_msg);
  // if (!dmap.empty()) {
  //   sensor_msgs::ImagePtr disp_msg;
  //   disp_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", dmap).toImageMsg();
  //   dmap_pub.publish(disp_msg);
  // }
  
  _m_octomap_pub.publish(map_msg);
  // pc.channels.push_back(ch);
  // pcl_pub.publish(pc);
}
void findRectificationMap(FileStorage& calib_file, Size finalSize) {
  Rect validRoi[2];
  cout << "Starting rectification" << endl;
  stereoRectify(K1, D1, K2, D2, calib_img_size, R, Mat(T), R1, R2, P1, P2, Q, 
                CV_CALIB_ZERO_DISPARITY, 0, finalSize, &validRoi[0], &validRoi[1]);
  cv::initUndistortRectifyMap(K1, D1, R1, P1, finalSize, CV_32F, lmapx, lmapy);
  cv::initUndistortRectifyMap(K2, D2, R2, P2, finalSize, CV_32F, rmapx, rmapy);
  cout << "Done rectification" << endl;
}

void imgCallback(const sensor_msgs::ImageConstPtr& msg_left, const sensor_msgs::ImageConstPtr& msg_right) {
  Mat tmpL = cv_bridge::toCvShare(msg_left, "mono8")->image;
  Mat tmpR = cv_bridge::toCvShare(msg_right, "mono8")->image;
  if (tmpL.empty() || tmpR.empty())
    return;
  
  // Mat img_left, img_right, img_left_color;
  // remap(tmpL, img_left, lmapx, lmapy, cv::INTER_LINEAR);
  // remap(tmpR, img_right, rmapx, rmapy, cv::INTER_LINEAR);
  
  //cvtColor(img_left, img_left_color, CV_GRAY2BGR);
  
  //Mat dmap = generateDisparityMap(tmpL, tmpR);
  //publishPointCloud(tmpL, dmap);
  
  // imshow("LEFT", img_left);
  // imshow("RIGHT", img_right);
  // imshow("DISP", dmap);
  //waitKey(30);
}

void recur(Mat &l, Mat &r ){

  
  generateDisparityMap(l, r, disp);
  publishPointCloud(l, disp);

  return;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "demo");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);

  elas = new StereoEfficientLargeScale(0,128);
  
  const char* left_img_topic;
  const char* right_img_topic;
  const char* calib_file_name;
  int calib_width, calib_height, out_width, out_height;
  
  static struct poptOption options[] = {
    { "left_topic",'l',POPT_ARG_STRING,&left_img_topic,0,"Left image topic name","STR" },
    { "right_topic",'r',POPT_ARG_STRING,&right_img_topic,0,"Right image topic name","STR" },
    { "calib_file",'c',POPT_ARG_STRING,&calib_file_name,0,"Stereo calibration file name","STR" },
    { "calib_width",'w',POPT_ARG_INT,&calib_width,0,"Calibration image width","NUM" },
    { "calib_height",'h',POPT_ARG_INT,&calib_height,0,"Calibration image height","NUM" },
    { "out_width",'u',POPT_ARG_INT,&out_width,0,"Rectified image width","NUM" },
    { "out_height",'v',POPT_ARG_INT,&out_height,0,"Rectified image height","NUM" },
    { "debug",'d',POPT_ARG_INT,&debug,0,"Set d=1 for cam to robot frame calibration","NUM" },
    POPT_AUTOHELP
    { NULL, 0, 0, NULL, 0, NULL, NULL }
  };

  POpt popt(NULL, argc, argv, options, 0);
  int c;
  while((c = popt.getNextOpt()) >= 0) {}
  
  calib_img_size = Size(calib_width, calib_height);
  out_img_size = Size(out_width, out_height);
  
  calib_file = FileStorage(calib_file_name, FileStorage::READ);
  calib_file["K1"] >> K1;
  calib_file["K2"] >> K2;
  calib_file["D1"] >> D1;
  calib_file["D2"] >> D2;
  calib_file["R"] >> R;
  calib_file["T"] >> T;
  calib_file["XR"] >> XR;
  calib_file["XT"] >> XT;
  
  findRectificationMap(calib_file, out_img_size);

  Mat leftf,rightf,left,right;
  cv::Size s(160,120);
  leftf = cv::imread("/home/aryan/disparity/libelas/img/left1.png",0);
  rightf = cv::imread("/home/aryan/disparity/libelas/img/right1.png",0);
  cv::resize(leftf, left, s);
  cv::resize(rightf, right, s);

  _m_octomap_pub = nh.advertise<octomap_msgs::Octomap>("/hedwig/map", 1);
  // dmap_pub = it.advertise("/camera/left/disparity_map", 1);
  while(nh.ok()){
   
    recur(left,right);
    ros::spinOnce();
}
  // message_filters::Subscriber<sensor_msgs::Image> sub_img_left(nh, "camera/left_image", 1);
  // message_filters::Subscriber<sensor_msgs::Image> sub_img_right(nh, "camera/right_image", 1);
  
  // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
  // message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), sub_img_left, sub_img_right);
  // sync.registerCallback(boost::bind(&imgCallback, _1, _2));
  
  // dynamic_reconfigure::Server<stereo_dense_reconstruction::CamToRobotCalibParamsConfig> server;
  // dynamic_reconfigure::Server<stereo_dense_reconstruction::CamToRobotCalibParamsConfig>::CallbackType f;

  // f = boost::bind(&paramsCallback, _1, _2);
  // server.setCallback(f);
  
 
  //pcl_pub = nh.advertise<sensor_msgs::PointCloud>("/camera/left/point_cloud",1);
  //_m_octomap_pub = nh.advertise<octomap_msgs::Octomap>("/hedwig/map", 1);

  //ros::spin();

  delete elas;
  return 0;
}