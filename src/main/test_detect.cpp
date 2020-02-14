#include "detect.hpp"
#include <opencv2/opencv.hpp>
#include <mutex>
#define QUIT_KEY     'q'
#define DISP_WINNANE "camera"

std::mutex resultLock;
std::vector<face_box> faceinfo;
void cb1(std::vector<face_box> faceList, void* userData) {
	resultLock.lock();
	faceinfo = faceList;
	resultLock.unlock();
}

int main(int argc, char* argv[])
{
	//cv::VideoCapture camera("rtsp://admin:8358s12s@192.168.1.38", cv::CAP_FFMPEG);
	cv::VideoCapture camera(0);
	detectSetup(1, 3, cb1);
	cv::namedWindow(DISP_WINNANE, cv::WINDOW_AUTOSIZE);
	do {
		cv::Mat img;
		camera >> img;
		detectImage(img, NULL);
		
		resultLock.lock();
		std::vector<face_box> faces = faceinfo;
		resultLock.unlock();

		for (unsigned int i = 0; i < faces.size(); i++) {
			face_box& box = faces[i];

			//draw box 
			cv::rectangle(img, cv::Point(box.x0, box.y0),
				cv::Point(box.x1, box.y1), cv::Scalar(0, 255, 0), 2);

			// draw landmark 
			for (int l = 0; l < 5; l++) {
				cv::circle(img, cv::Point(box.landmark.x[l],
					box.landmark.y[l]), 2, cv::Scalar(0, 0, 255), 2);
			}
		}
		cv::imshow(DISP_WINNANE, img);
	} while (QUIT_KEY != cv::waitKey(1));
	detectStop();
}
