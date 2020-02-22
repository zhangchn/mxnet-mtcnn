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

static void do_extract_image(const char *filepath, const std::string &filename)
{
    std::cout << "detect image: "<< filename << std::endl;
	detectSetup(1, 3, cb2);
    cv::Mat img = cv::imread(filepath);
    current_file = filename;
	current_out_idx=0;
    if (!img.data) {
        detectStop();
        return;
    }
    pending = 1;
    detectImage(img, (void *)filepath);
    std::unique_lock<std::mutex> lk(resultLock);
    result_saturated.wait(lk, [] {
            return pending == 0;
            });

	detectStop();
}

static void do_extract_video(const char *filepath, const std::string &filename, int frame_stride)
{
	cv::VideoCapture camera(filepath, cv::CAP_FFMPEG);
	detectSetup(1, 3, cb2);
	//cv::namedWindow(DISP_WINNANE, cv::WINDOW_AUTOSIZE);
	current_file = filename;
	current_out_idx=0;
	//int frame_stride = 20;
	do {
        cv::Mat img;
        for (int counter = 0; counter < frame_stride; counter++) {
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
	} while (true);
	detectStop();
}

int main(int argc, char* argv[])
{
	//cv::VideoCapture camera("rtsp://admin:8358s12s@192.168.1.38", cv::CAP_FFMPEG);
    int video_stride = 0;
	for (int arg_idx = 1; arg_idx < argc; arg_idx++) {
		std::string argument(argv[arg_idx], strlen(argv[arg_idx]));

        if (argument == "--video-stride") {
            if (arg_idx == argc - 1) {
                std::cerr << "Usage: --video-stride N" << std::endl << "  skip n frames before each input frame for video" << std::endl;
                return -1;
            }
            video_stride = atoi(argv[arg_idx+1]);
            arg_idx ++;
            continue;
        }
        std::string &fpath = argument;

		char *cfname = basename(argv[arg_idx]);
		std::string name(cfname, strlen(cfname)); 
		if (access(argv[arg_idx], R_OK) != 0) {
			std::cout << "access failed:" <<fpath << std::endl;
			continue;
		}
		std::cout << "processing input:" << fpath << std::endl;
        if (std::string::npos != name.rfind(".mp4", name.length() - 4, 4)) {
            do_extract_video(argv[arg_idx], name, video_stride);
        } else if (std::string::npos != name.rfind(".jpg", name.length() - 4, 4)
                || std::string::npos != name.rfind(".jpeg", name.length() - 5, 5)
                || std::string::npos != name.rfind(".png", name.length() - 4, 4)) {
            do_extract_image(argv[arg_idx], name);
        }

                
	}
	return 0;
}
