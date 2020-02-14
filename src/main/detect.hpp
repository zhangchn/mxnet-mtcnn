#include <string>
#include <opencv2/opencv.hpp>
#include "mtcnn.hpp"

typedef void (*DetectCallBack)(std::vector<face_box> faceList, void *userData);
typedef void (*DetectCallBack2)(std::vector<face_box> faceList, const cv::Mat &image, void *userData);
bool detectSetup(int numThread, int queueSize, DetectCallBack cb);
bool detectSetup(int numThread, int queueSize, DetectCallBack2 cb);
void detectImage(const cv::Mat &image, void *userData);
void detectStop();
