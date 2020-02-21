#include "detect.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>


#include "utils.hpp"

std::deque<cv::Mat> frame_buffer;
std::deque<std::chrono::time_point<std::chrono::system_clock>> frame_timestamp;
std::deque<void*> frame_userdata;
std::vector<std::thread> thread_pool;

std::chrono::time_point<std::chrono::system_clock> frame_timestamp_displayed;
std::mutex fb_mutex;
std::condition_variable non_empty;
//std::vector<face_box> face_info;
std::mutex init_mutex;
//std::chrono::milliseconds timestamp = -1;
bool should_quit = true;
DetectCallBack _cb = nullptr;
DetectCallBack2 _cb2 = nullptr;
int queue_size = 5;

static void detect();
bool detectSetup(int numThread, int queueSize, DetectCallBack cb) {
    if (!should_quit) {
        return false;
    }
    if (cb == NULL) {
        return false;
    }
    should_quit = false;
    _cb = cb;
    for (int ti = 0; ti < numThread; ti++) {
        thread_pool.push_back(std::thread(detect));
    }

    return true;
}

bool detectSetup(int numThread, int queueSize, DetectCallBack2 cb) {
    if (!should_quit) {
        return false;
    }
    if (cb == NULL) {
        return false;
    }
    should_quit = false;
    _cb2 = cb;

    for (int ti = 0; ti < numThread; ti++) {
        thread_pool.push_back(std::thread(detect));
    }

    return true;
}

void detectImage(const cv::Mat& image, void *userData) {
    if (!image.data) {
        return;
    }
    // Queue image into frame_buffer
    std::vector<face_box> temp_faces;
    {
        std::lock_guard<std::mutex> lk(fb_mutex);
        frame_buffer.push_back(image);
        frame_userdata.push_back(userData);
        //frame_timestamp.push_back(std::chrono::system_clock::now());
        if (frame_buffer.size() > 3) {
            frame_buffer.pop_front();
            //frame_timestamp.pop_front();
            frame_userdata.pop_front();
        }
    }
    non_empty.notify_one();
}

void detectStop() {
    should_quit = true;
    non_empty.notify_all();
    for (auto it = thread_pool.begin(); it != thread_pool.end(); it++) {
        (*it).join();
    }
}

static void detect() {
    std::string type = "mxnet";
    std::string model_dir = "../models";
    Mtcnn* p_mtcnn;

    init_mutex.lock();
    p_mtcnn = MtcnnFactory::CreateDetector(type);

    if (p_mtcnn == nullptr) {
        std::cerr << type << " is not supported" << std::endl;
        std::cerr << "supported types: ";
        std::vector<std::string> type_list = MtcnnFactory::ListDetectorType();

        for (unsigned int i = 0; i < type_list.size(); i++)
            std::cerr << " " << type_list[i];

        std::cerr << std::endl;
        init_mutex.unlock();
        return;
    }

    p_mtcnn->SetFactorMinSize(0.709, 120);
    p_mtcnn->LoadModule(model_dir);
    init_mutex.unlock();
    do {
        //fb_mutex.lock();
        std::unique_lock<std::mutex> lk(fb_mutex);
        non_empty.wait(lk, [] {
                return should_quit || !(
                        frame_buffer.empty() 
                        || (_cb == nullptr && _cb2 == nullptr));
                });
        if (should_quit) {
            break;
        }
        cv::Mat frame = frame_buffer.front();
        frame_buffer.pop_front();
        void* userData = frame_userdata.front();
        frame_userdata.pop_front();
        lk.unlock();
        
        std::vector<face_box> temp_faces;
        unsigned long start_time = get_cur_time();
        p_mtcnn->Detect(frame, temp_faces);
        unsigned long end_time = get_cur_time();

        if (_cb != nullptr) {
            _cb(temp_faces, userData);
        } else {
            _cb2(temp_faces, frame, userData);
        }
        std::cout << "total detected: " << temp_faces.size() << " faces. used "
            << (end_time - start_time) << " us" << std::endl;
    } while (!should_quit);
}
