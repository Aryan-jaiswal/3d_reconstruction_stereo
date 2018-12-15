#include "StereoEfficientLargeScale.h"

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <signal.h>
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
Mat lmapx, lmapy, rmapx, rmapy, disp, disp_c;
sensor_msgs::ImagePtr disp_msg_l, disp_msg_r, disp_msg;
cv::Mat left, right, img_left, img_right, img_left_c, img_right_c;

Vec3d T;
Eigen::MatrixXd _Q,_XR,_XT;
Eigen::Vector3d quad_trans;
Eigen::Quaterniond quad_rot;
int from_to[] = {0,0,1,1}, stream;
bool isDispAvailable;
FileStorage calib_file;
Size out_img_size;
Size calib_img_size;

image_transport::Publisher dmap_pub_l,dmap_pub_r, disp_pub;
ros::Publisher _m_octomap_pub;
ros::Subscriber _global_pose;
Mat leftdpf;
StereoEfficientLargeScale * elas;
std::mutex mtx, dmtx;
bool run_thread;
bool publish_disparity = true;
bool debug_couts = false;
std::condition_variable mcv;
std::condition_variable dcv;
volatile sig_atomic_t flag=0;
//#define EIGEN_DONT_PARALLELIZE
#define EIGEN_FAST_MATH 1

//void set_flag()
//{
//    flag=1;
//}

void generateDisparityMap() {
 
 while(run_thread){
    std::unique_lock<std::mutex> lk(mtx);
    mcv.wait(lk);
    if(debug_couts) cout<<"disparity wait finished\n";
    img_left.copyTo(img_left_c);
    img_right.copyTo(img_right_c);    
    lk.unlock();

    elas->process(::img_left_c,::img_right_c,leftdpf,100);
    if(debug_couts) cout<<"disparity generated\n";
    
    std::lock_guard<std::mutex> lk2(dmtx);
    leftdpf.convertTo(disp, CV_8U, 1.0);
    if(debug_couts) cout<<"disparity available\n";
    dcv.notify_one();
    
    if(publish_disparity){
        disp_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", ::disp).toImageMsg();
        disp_pub.publish(disp_msg);
    }
  }

}

void publishPointCloud(ros::Publisher _m_octomap_pub) {
    octomap_msgs::Octomap map_msg;
    map_msg.header.frame_id = "Hedwig";
    map_msg.binary = true;
    map_msg.id = "OcTree";
    map_msg.resolution = kMapResolution;
    

    while(run_thread){
        
    std::shared_ptr<octomap::OcTree> map_ptr = std::make_shared<octomap::OcTree>(kMapResolution);
    octomap::OcTree* tree = map_ptr.get();
    
    std::unique_lock<std::mutex> lk(dmtx);
    dcv.wait(lk);
    if(debug_couts) cout<<"octomap wait finished\n";
    disp.copyTo(disp_c);
    lk.unlock();
    
    map_msg.header.stamp = ros::Time::now();
    int i, j, d;
    int numthreads = 1;
    //cout << "Max thread" << omp_get_max_threads() <<"\n";
    omp_set_num_threads(numthreads);
    //cout << "Max thread" << omp_get_max_threads() <<"\n";
    
    Eigen::Matrix3d R_quad_world = quad_rot.matrix();
    Eigen::MatrixXd V(4,1);
    Eigen::MatrixXd pos;
    Eigen::MatrixXd point3d_robot(3,1);
    Eigen::MatrixXd point_w_world;

//#pragma omp parallel for collapse(2) private(d) shared(_Q,tree,R_quad_world,quad_trans)
    for (i = 0; i < ::img_left.cols; i++) {
      for (j = 0; j < ::img_left.rows; j++) {
        d = disp_c.at<uchar>(j,i);
        if (d < 2) continue;

        // V is the vector to be multiplied to Q to get
        // the 3D homogenous coordinates of the image point
    
        V(0,0) = (double)(i);
        V(1,0) = (double)(j);
        V(2,0) = (double)d;
        V(3,0) = 1.;
        pos = _Q * V; // 3D homogeneous coordinate
        //cout << i << " " << j << "\n";
        //cout << omp_get_thread_num() << "\n";        
        point3d_robot(0,0) = pos(0,0) / pos(3,0);
        point3d_robot(1,0) = pos(1,0) / pos(3,0);
        point3d_robot(2,0) = pos(2,0) / pos(3,0);
        // transform 3D point from camera frame to robot frame
        //point3d_robot = _XR * point3d_cam + _XT;
        //transform from robot frame to world frame
        point_w_world = R_quad_world * point3d_robot + quad_trans;
        
        tree->updateNode(point_w_world(0,0), point_w_world(1,0), point_w_world(2,0), true);
      }
    }
    if(debug_couts) cout<<"octomap processed\n";
    octomap_msgs::binaryMapToMsg(*(tree), map_msg);    
    _m_octomap_pub.publish(map_msg);
    
  }
}

void findRectificationMap(Size finalSize) {
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
    //Eigen::initParallel();
    ros::init(argc, argv, "demo");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    elas = new StereoEfficientLargeScale(0,40);
    //signal(SIGINT,set_flag);

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

    calib_img_size = Size(188,120);
    out_img_size = Size(188,120);

    calib_file = FileStorage(calib_file_name, FileStorage::READ);
    calib_file["K1"] >> K1;
    calib_file["K2"] >> K2;
    calib_file["D1"] >> D1;
    calib_file["D2"] >> D2;
    calib_file["R"] >> R;
    calib_file["T"] >> T;
    calib_file["XR"] >> XR;
    calib_file["XT"] >> XT;

    findRectificationMap(out_img_size);

    cv2eigen(Q,_Q);
    cv2eigen(XR,_XR);
    cv2eigen(XT,_XT);

    cv::VideoCapture cap(stream);
    cap.set(CV_CAP_PROP_CONVERT_RGB, false);

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
    ::img_left = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
    ::img_right = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
    ::disp = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);

    _m_octomap_pub = nh.advertise<octomap_msgs::Octomap>("/hedwig/map", 1);
    _global_pose = nh.subscribe("/mavros/local_position/pose", 1, &pose_cb);
    dmap_pub_l = it.advertise("/camera/left/image_raw", 1);
    dmap_pub_r = it.advertise("/camera/right/image_raw", 1);
    disp_pub = it.advertise("/camera/disparity/image_raw", 1);
    std::thread t1(generateDisparityMap);//thread 1
    //cout<<" Called t1"<<endl;
    std::thread t2(publishPointCloud,_m_octomap_pub);//thread 2
    //cout<<" Called t2"<<endl;

    ros::Rate r(50);
    while(nh.ok()) {
        cap >> frame_in; if (frame_in.empty()) break;
        cv::resize(frame_in, frame, s);

        std::lock_guard<std::mutex> lk(mtx);    
        extract_left_right(frame);    
        remap(::left,::img_left , lmapx, lmapy, cv::INTER_LINEAR);
        remap(::right, ::img_right, rmapx, rmapy, cv::INTER_LINEAR);


        disp_msg_l = cv_bridge::CvImage(std_msgs::Header(), "mono8", ::left).toImageMsg();
        dmap_pub_l.publish(disp_msg_l);
        disp_msg_r = cv_bridge::CvImage(std_msgs::Header(), "mono8", ::right).toImageMsg();
        dmap_pub_r.publish(disp_msg_r);

        if(debug_couts) cout<<"finished grabbing image\n";

        mcv.notify_one();


        ros::spinOnce();
        r.sleep();
    }
    //run_thread = false;
    //t1.terminate();
    //t2.terminate();
    t1.join();
    t2.join();

    delete elas;
    return 0;
}
