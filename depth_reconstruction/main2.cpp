#include "StereoEfficientLargeScale.h"

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <thread>
#include <mutex>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/ChannelFloat32.h>
#include <geometry_msgs/Point32.h>
#include <geometry_msgs/PoseStamped.h>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <stdio.h>
#include <dynamic_reconfigure/server.h>
#include <fstream>
#include <ctime>
#include <omp.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <octomap/octomap.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <octomap_ros/conversions.h>
#include <opencv2/core/eigen.hpp>
#include <unistd.h>
//#include <opencv2/opencv.hpp>
//#include "elas.h"
#include "popt_pp.h"

#define CV_VERSION_NUMBER CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)


#ifdef _DEBUG
#pragma comment(lib, "opencv_viz" CV_VERSION_NUMBER "d.lib")
#pragma comment(lib, "opencv_videostab" CV_VERSION_NUMBER "d.lib")
#pragma comment(lib, "opencv_video" CV_VERSION_NUMBER "d.lib")
#pragma comment(lib, "opencv_ts" CV_VERSION_NUMBER "d.lib")
#pragma comment(lib, "opencv_superres" CV_VERSION_NUMBER "d.lib")
#pragma comment(lib, "opencv_stitching" CV_VERSION_NUMBER "d.lib")
#pragma comment(lib, "opencv_photo" CV_VERSION_NUMBER "d.lib")
#pragma comment(lib, "opencv_ocl" CV_VERSION_NUMBER "d.lib")
#pragma comment(lib, "opencv_objdetect" CV_VERSION_NUMBER "d.lib")
#pragma comment(lib, "opencv_nonfree" CV_VERSION_NUMBER "d.lib")
#pragma comment(lib, "opencv_ml" CV_VERSION_NUMBER "d.lib")
#pragma comment(lib, "opencv_legacy" CV_VERSION_NUMBER "d.lib")
#pragma comment(lib, "opencv_imgproc" CV_VERSION_NUMBER "d.lib")
#pragma comment(lib, "opencv_highgui" CV_VERSION_NUMBER "d.lib")
//#pragma comment(lib, "opencv_haartraining_engine.lib")
#pragma comment(lib, "opencv_gpu" CV_VERSION_NUMBER "d.lib")
#pragma comment(lib, "opencv_flann" CV_VERSION_NUMBER "d.lib")
#pragma comment(lib, "opencv_features2d" CV_VERSION_NUMBER "d.lib")
#pragma comment(lib, "opencv_core" CV_VERSION_NUMBER "d.lib")
#pragma comment(lib, "opencv_contrib" CV_VERSION_NUMBER "d.lib")
#pragma comment(lib, "opencv_calib3d" CV_VERSION_NUMBER "d.lib")
#else
#pragma comment(lib, "opencv_viz" CV_VERSION_NUMBER ".lib")
#pragma comment(lib, "opencv_videostab" CV_VERSION_NUMBER ".lib")
#pragma comment(lib, "opencv_video" CV_VERSION_NUMBER ".lib")
#pragma comment(lib, "opencv_ts" CV_VERSION_NUMBER ".lib")
#pragma comment(lib, "opencv_superres" CV_VERSION_NUMBER ".lib")
#pragma comment(lib, "opencv_stitching" CV_VERSION_NUMBER ".lib")
#pragma comment(lib, "opencv_photo" CV_VERSION_NUMBER ".lib")
#pragma comment(lib, "opencv_ocl" CV_VERSION_NUMBER ".lib")
#pragma comment(lib, "opencv_objdetect" CV_VERSION_NUMBER ".lib")
#pragma comment(lib, "opencv_nonfree" CV_VERSION_NUMBER ".lib")
#pragma comment(lib, "opencv_ml" CV_VERSION_NUMBER ".lib")
#pragma comment(lib, "opencv_legacy" CV_VERSION_NUMBER ".lib")
#pragma comment(lib, "opencv_imgproc" CV_VERSION_NUMBER ".lib")
#pragma comment(lib, "opencv_highgui" CV_VERSION_NUMBER ".lib")
//#pragma comment(lib, "opencv_haartraining_engine.lib")
#pragma comment(lib, "opencv_gpu" CV_VERSION_NUMBER ".lib")
#pragma comment(lib, "opencv_flann" CV_VERSION_NUMBER ".lib")
#pragma comment(lib, "opencv_features2d" CV_VERSION_NUMBER ".lib")
#pragma comment(lib, "opencv_core" CV_VERSION_NUMBER ".lib")
#pragma comment(lib, "opencv_contrib" CV_VERSION_NUMBER ".lib")
#pragma comment(lib, "opencv_calib3d" CV_VERSION_NUMBER ".lib")
#endif


#define baseline 12
#define focal 358
#define kMapResolution 0.2

using namespace cv;
using namespace std;

Mat Q, P1, P2, XR, XT;
Mat R1, R2, K1, K2, D1, D2, R;
//Mat *out;
Mat lmapx, lmapy, rmapx, rmapy, disp;
cv::Mat left, right;
Vec3d T;
Eigen::MatrixXd _Q,_XR,_XT;
Eigen::Vector3d quad_trans;
Eigen::Quaterniond quad_rot;
int from_to[] = {0,0,1,1}, stream;
bool isDispAvailable;
FileStorage calib_file;
Size out_img_size;
Size calib_img_size;

image_transport::Publisher dmap_pub;
ros::Publisher _m_octomap_pub;
ros::Subscriber _global_pose;
Mat leftdpf;
StereoEfficientLargeScale * elas;
std::mutex mtx;
bool run_thread;

void generateDisparityMap() {
 
 while(run_thread){
    if (::left.empty() || ::right.empty()) 
      continue;
    
    ////////////////////
   // cout<<" Called_gdm_init"<<endl;
    mtx.lock();
   // cout<<" Called_gdm"<<endl;
    elas->process(::left,::right,leftdpf,100);
    //cout<<" Called_elas"<<endl;
    leftdpf.convertTo(disp, CV_8U, 1./8);
    mtx.unlock();
   // cout<<" Called_elas2"<<endl;
    mtx.lock();
    isDispAvailable = true;
    mtx.unlock();
    //cout<<" Called_exxit"<<endl;
  }
  /////////////////////

}

void publishPointCloud(ros::Publisher _m_octomap_pub) {
  
  //Mat V = Mat(4, 1, CV_64FC1);

  while(run_thread){

    Eigen::MatrixXd V(4,1);
    Eigen::MatrixXd pos;
    Eigen::MatrixXd point3d_cam(3,1), point3d_robot(3,1);
    double X, Y, Z;
    //Mat pos = Mat(4, 1, CV_64FC1);
    
    octomap_msgs::Octomap map_msg; 

    std::shared_ptr<octomap::OcTree> map_ptr = std::make_shared<octomap::OcTree>(kMapResolution);

    octomap::OcTree* tree = map_ptr.get();
    map_msg.header.stamp = ros::Time::now();
    map_msg.header.frame_id = "Hedwig";
    map_msg.binary = true;
    map_msg.id = "OcTree";
    map_msg.resolution = kMapResolution;
    
    
      ////
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ::left.cols; i++) {
      for (int j = 0; j < ::left.rows; j++) {
        //cout<<" Called_ppc"<<endl;
        // mtx.lock();
       // cout<<" Called_ppc2"<<endl;
        while(!isDispAvailable) {
         // mtx.unlock();
          cout<<"Disp not available"<<endl;
          usleep(1000);
          //mtx.lock();
        }
        int d = disp.at<uchar>(j,i);
        isDispAvailable = true;
        // mtx.unlock();
        // if low disparity, then ignore
        //cout<<"Crossed barriers"<<endl;
        if (d < 2) {
          
          continue;
        }

        // V is the vector to be multiplied to Q to get
        // the 3D homogenous coordinates of the image point
        V(0,0) = (double)(i);
        V(1,0) = (double)(j);
        V(2,0) = (double)d;
        V(3,0) = 1.;
        // cout<<"calculating pose"<<endl;
        // cout<<V.rows()<<V.cols()<<endl;
        //cout<<_Q.rows()<<_Q.cols()<<endl;
        pos = _Q * V; // 3D homogeneous coordinate
       // cout<<"calculated----------------"<<endl;
        X = pos(0,0) / pos(3,0);
        Y = pos(1,0) / pos(3,0);
        Z = pos(2,0) / pos(3,0);
        

        point3d_cam(0,0) = X;
        point3d_cam(1,0) = Y;
        point3d_cam(2,0) = Z;
        // transform 3D point from camera frame to robot frame
        point3d_robot = _XR * point3d_cam + _XT;
        //cout<<"robot"<<endl;
        //transform from robot frame to world frame
        Eigen::Matrix3d R_quad_world = quad_rot.matrix();
        Eigen::MatrixXd point_w_world = R_quad_world * point3d_robot;
        point_w_world = point_w_world + quad_trans;

        //cout<<point_w_world(0,0)<<endl;

        
        tree->updateNode(point_w_world(0,0), point_w_world(1,0), point_w_world(2,0), true);
        //cout<< "end" <<i<<endl;
      }
    }
    //cout<<"published---------->>>>>>"<<endl;
    //tree->updateInnerOccupancy();
    octomap_msgs::binaryMapToMsg(*(tree), map_msg);
    // if (!dmap.empty()) {
    //   sensor_msgs::ImagePtr disp_msg;
    //   disp_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", dmap).toImageMsg();
    //   dmap_pub.publish(disp_msg);
    // }
    
    _m_octomap_pub.publish(map_msg);
    
  }
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
void pose_cb(const geometry_msgs::PoseStampedConstPtr& msg) {
  

  quad_trans(0) = msg->pose.position.x;
  quad_trans(1) = msg->pose.position.y;
  quad_trans(2) = msg->pose.position.z;
  quad_rot.w() = msg->pose.orientation.w;
  quad_rot.x() = msg->pose.orientation.x;
  quad_rot.y() = msg->pose.orientation.y;
  quad_rot.z() = msg->pose.orientation.z;

}
void extract_left_right(const cv::Mat & raw)  {
  Mat out[] = {::left, ::right};
  // raw[0] -> left[0],
  // raw[1] -> right[0]
  cv::mixChannels( &raw, 1, out, 2, from_to, 2 );
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
    { "video_stream",'s',POPT_ARG_INT,&stream,0,"Specify the video stream","NUM" },
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

  cv2eigen(Q,_Q);
  cv2eigen(XR,_XR);
  cv2eigen(XT,_XT);

  cv::VideoCapture cap(stream);
  cap.set(CV_CAP_PROP_CONVERT_RGB, false);

  isDispAvailable = true;
  run_thread = true;

  // Check if camera opened successfully
  if(!cap.isOpened()){
    std::cout << "Error opening video stream\n" << std::endl;
    return -1;
  }
  else
     std::cout << "Opened video stream\n" << std::endl;

  // First Image
  cv::Mat frame,frame_in;
  
  cap >> frame_in; if (frame_in.empty()) return -1;
  
  cv::Size s(188,120);
  cv::resize(frame_in, frame, s);
  
  ::left = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
  ::right = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
  ::disp = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
 
  _m_octomap_pub = nh.advertise<octomap_msgs::Octomap>("/hedwig/map", 1);
  _global_pose = nh.subscribe("/mavros/local_position/pose", 1, &pose_cb);

  // dmap_pub = it.advertise("/camera/left/disparity_map", 1);
  std::thread t1(generateDisparityMap);//thread 1
  //cout<<" Called t1"<<endl;
  std::thread t2(publishPointCloud,_m_octomap_pub);//thread 2
  //cout<<" Called t2"<<endl;
  
  ros::Rate r(8);
  while(nh.ok()) {

    cap >> frame_in; if (frame_in.empty()) break;
    cv::resize(frame_in, frame, s);
    extract_left_right(frame);
    //cout<<"Extracted images"<<endl;
    //master(left,right);
    ros::spinOnce();
    r.sleep();
}
//run_thread = false;
t1.join();
t2.join();

  // message_filters::Subscriber<sensor_msgs::Image> sub_img_left(nh, "camera/left_image", 1);
  // message_filters::Subscriber<sensor_msgs::Image> sub_img_right(nh, "camera/right_image", 1);
  
  // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
  // message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), sub_img_left, sub_img_right);
  // sync.registerCallback(boost::bind(&imgCallback, _1, _2));

  // f = boost::bind(&paramsCallback, _1, _2);
  // server.setCallback(f); 

  delete elas;
  return 0;
}