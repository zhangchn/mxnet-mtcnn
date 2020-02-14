#include "detect.hpp"
#include <unistd.h>
#include <libgen.h>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <condition_variable>

#define QUIT_KEY     'q'
#define DISP_WINNANE "camera"

std::mutex resultLock;
std::condition_variable result_saturated;
int pending = 0;

std::vector<face_box> faceinfo;
int current_out_idx;
std::string current_file;

void cb2(std::vector<face_box> faces, const cv::Mat &image, void* userData) {
	//faceinfo = faceList;
	
	for (unsigned int i = 0; i < faces.size(); i++) {
		face_box& box = faces[i];
		std::cout<<"facebox: x(" << box.x0 << ", " << box.x1 << "), y(" << box.y0 << ", "<< box.y1 <<")"<< std::endl;
		if (box.x0 < 0 || box.x1 > image.cols || box.y0 < 0 || box.y1 > image.rows) {
			std::cout << "box exceeds range, abort." << std::endl;
			continue;
		}
		cv::Mat face = image(cv::Range(box.y0, box.y1), cv::Range(box.x0, box.x1));

		imwrite("/tmp/"+ current_file + "." + std::to_string(current_out_idx) + ".jpg", face);
		current_out_idx ++;
	}
	resultLock.lock();
	pending--;
	resultLock.unlock();
	result_saturated.notify_one();
}

static void do_extract(const char *filepath, const std::string &filename)
{
	cv::VideoCapture camera(filepath, cv::CAP_FFMPEG);
	detectSetup(1, 3, cb2);
	cv::namedWindow(DISP_WINNANE, cv::WINDOW_AUTOSIZE);
	current_file = filename;
	current_out_idx=0;
	int frame_stride = 20;
	do {
		cv::Mat img;
		for (int counter = 1; counter < frame_stride; counter++) {
			camera.grab(); 
		}
		camera >> img;
		if (img.data == NULL) {
			break;
		}
		std::unique_lock<std::mutex> lk(resultLock);
		result_saturated.wait(lk, [] {
				return pending < 4;
				});
		pending++;
		lk.unlock();
		detectImage(img, (void *)filepath);
		
		//resultLock.lock();
		//std::vector<face_box> faces = faceinfo;
		//resultLock.unlock();
		/*

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
		*/
	} while (true);
	detectStop();
}

int main(int argc, char* argv[])
{
	//cv::VideoCapture camera("rtsp://admin:8358s12s@192.168.1.38", cv::CAP_FFMPEG);
	for (int arg_idx = 1; arg_idx < argc; arg_idx++) {
		std::string fpath(argv[arg_idx], strlen(argv[arg_idx]));
		char *cfname = basename(argv[arg_idx]);
		std::string name(cfname, strlen(cfname)); 
		if (access(argv[arg_idx], R_OK) != 0) {
			std::cout << "access failed:" <<fpath << std::endl;
			continue;
		}
		std::cout << "processing input:" << fpath << std::endl;
		do_extract(argv[arg_idx], name);
	}
	return 0;
}