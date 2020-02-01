#include <string>
#include <opencv2/opencv.hpp>
#include "mtcnn.hpp"

typedef void (*DetectCallBack)(std::vector<face_box> faceList, void *userData);
bool detectSetup(int numThread, int queueSize, DetectCallBack cb);
void detectImage(const cv::Mat &image, void *userData);
void detectStop();