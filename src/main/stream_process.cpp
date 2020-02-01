/*
  Copyright (C) 2017 Open Intelligent Machines Co.,Ltd

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/
#if defined (__APPLE__) || defined(__linux__)
#include <unistd.h>
#endif
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>

#include "mtcnn.hpp"
#include "utils.hpp"

#define DISP_WINNANE "camera"
#define QUIT_KEY     'q'
#define CAMID         0

std::deque<cv::Mat> frame_buffer;
std::deque<std::chrono::time_point<std::chrono::system_clock>> frame_timestamp;
std::chrono::time_point<std::chrono::system_clock> frame_timestamp_displayed;
std::mutex fb_mutex;
std::condition_variable non_empty;
std::vector<face_box> face_info;
std::mutex init_mutex;
//std::chrono::milliseconds timestamp = -1;
bool should_quit = false;
void detect() {
    std::string type = "mxnet";
    std::string model_dir = "../models";
    Mtcnn * p_mtcnn;

    init_mutex.lock();
    p_mtcnn = MtcnnFactory::CreateDetector(type);

    if (p_mtcnn == nullptr) {
        std::cerr << type << " is not supported" << std::endl;
        std::cerr << "supported types: ";
        std::vector<std::string> type_list = MtcnnFactory::ListDetectorType();

        for (unsigned int i = 0; i < type_list.size(); i++)
            std::cerr << " " << type_list[i];

        std::cerr << std::endl;

        return ;
    }

    p_mtcnn->SetFactorMinSize(0.709, 350);
    p_mtcnn->LoadModule(model_dir);
    init_mutex.unlock();
    do {
        //fb_mutex.lock();
        std::unique_lock<std::mutex> lk(fb_mutex);
        non_empty.wait(lk, []{
                return should_quit || !frame_buffer.empty();
                });
        if (should_quit) {
            break;
        }
        //if (frame_buffer.empty()) {
            //fb_mutex.unlock();
        //    std::this_thread::sleep_for(std::chrono::milliseconds(30));
        //    continue;
        //}
        cv::Mat frame = frame_buffer.front();
        auto frame_time = frame_timestamp.front();
        frame_buffer.pop_front();
        frame_timestamp.pop_front();
        //fb_mutex.unlock();
        lk.unlock();
        //non_empty.notify_one();

        if (frame_time < frame_timestamp_displayed) {
            continue;
        } 
        unsigned long start_time = 0;
        unsigned long end_time = 0;
        std::vector<face_box> temp_faces;
        start_time = get_cur_time();
        p_mtcnn->Detect(frame, temp_faces);
        end_time = get_cur_time();

        fb_mutex.lock();
        if (frame_time < frame_timestamp_displayed) {
            // wasted
            std::cout<< "wasted" << std::endl;
            fb_mutex.unlock();
            continue;
        } 
        face_info = temp_faces;
        frame_timestamp_displayed = frame_time;
        fb_mutex.unlock();
        std::cout<< "total detected: " << temp_faces.size() << " faces. used "
            << (end_time-start_time) << " us" << std::endl;
    } while(!should_quit);
}

int main(int argc, char * argv[])
{
    std::string type = "mxnet";
    std::string model_dir = "../models";

    frame_timestamp_displayed = std::chrono::system_clock::now();
    int res;
#if defined(__APPLE__) ||defined(__linux__)
    while ((res = getopt(argc, argv, "t:")) != -1) {
        switch (res) {
            case 't':
                type = std::string(optarg);
                break;
            case 'm':
                model_dir = std::string(optarg);
                break;
            default:
                break;
        }
    }
#endif
    //cv::VideoCapture camera(CAMID);
    cv::VideoCapture camera("rtsp://192.168.1.38", cv::CAP_FFMPEG);

    //camera.set(cv::CAP_PROP_FRAME_WIDTH, 960);
    //camera.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    if (!camera.isOpened()) {
        std::cerr << "failed to open camera" << std::endl;
        return 1;
    }



    cv::namedWindow(DISP_WINNANE, cv::WINDOW_AUTOSIZE);

    unsigned long start_time = 0;
    unsigned long end_time = 0;

    //std::vector<const std::thread &> workers;
    //for (int ti = 0; ti < 3; ti++) {
        //worker.detach();
        //workers.push_back(worker);
    //}
    std::thread worker1(detect);
    std::thread worker2(detect);
    //std::thread worker3(detect);
    //std::thread worker4(detect);
    //std::thread worker5(detect);
    //std::thread worker6(detect);
    do {
        cv::Mat frame;
        camera >> frame;

        if (!frame.data) {
            std::cerr << "Capture video failed" << std::endl;
            break;
        }

        //fb_mutex.lock();
        std::vector<face_box> temp_faces;
        {
            std::lock_guard<std::mutex> lk(fb_mutex);
            frame_buffer.push_back(frame);
            frame_timestamp.push_back(std::chrono::system_clock::now());
            if (frame_buffer.size() > 3) {
                frame_buffer.pop_front();
                frame_timestamp.pop_front();
            }
            temp_faces = face_info;
        }
        //fb_mutex.unlock();
        non_empty.notify_one();

        /*
        start_time = get_cur_time();
        p_mtcnn->Detect(frame, face_info);
        end_time = get_cur_time();
        */

        for (unsigned int i = 0; i < temp_faces.size(); i++) {
            face_box & box = temp_faces[i];

            //draw box 
            cv::rectangle(frame, cv::Point(box.x0, box.y0),
                    cv::Point(box.x1, box.y1), cv::Scalar(0, 255, 0), 2);

            // draw landmark 
            for (int l = 0; l < 5; l++) {
                cv::circle(frame, cv::Point(box.landmark.x[l],
                            box.landmark.y[l]), 2, cv::Scalar(0, 0, 255), 2);
            }
        }



        /* face_info.clear();*/
        cv::imshow(DISP_WINNANE, frame);

    } while (QUIT_KEY != cv::waitKey(1));
    should_quit = true;
    non_empty.notify_all();
    worker1.join();
    worker2.join();
    /* worker3.join();
    worker4.join();
    worker5.join();
    worker6.join();*/
    return 0;
}
