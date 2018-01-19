#include "motion_detection_jni.h"
#include <jni.h>
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui.hpp>  // OpenCV window I/O

#include "opencv2/opencv.hpp"
#include <opencv2/features2d.hpp>
#include "opencv2/optflow/motempl.hpp"
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "opencv2/video/background_segm.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>

#include <vector>
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <iostream>
#include <stdio.h>
#include <android/log.h>
#include <pthread.h>

using namespace cv;
using namespace cv::motempl;
using namespace std;

#ifndef LOG_TAG
#define LOG_TAG "MotionDetectorJni"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG ,__VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG ,__VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,LOG_TAG ,__VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG ,__VA_ARGS__)
#define LOGF(...) __android_log_print(ANDROID_LOG_FATAL,LOG_TAG ,__VA_ARGS__)
#endif

extern "C" {
    // number of cyclic frame buffer used for motion detection
    // (should, probably, depend on FPS)
    const int N = 4;
    const int INSIDE = 0;
    const int OVERLAY = 1;
    const int OUTSIDE = 2;
    const int MAX_AREA = 800;

    // ring image buffer
    vector<Mat> buf(N);
    HOGDescriptor hog;
    Ptr<BackgroundSubtractor> bgS = createBackgroundSubtractorMOG2(500, 25, false);

    void hogDetect(Mat& frame, Mat& output) {//        }
//        if (frame.cols > 800) {
            resize(frame, frame, Size(), 0.5, 0.5);
//        }

        bgS->apply(frame, output);
        erode(output,output,Mat());

        // Find contours
        vector<vector<Point> > contours;
        findContours( output, contours, RETR_LIST, CHAIN_APPROX_SIMPLE );
        LOGD("countours size: %u", contours.size());

        for ( size_t i = 0; i < contours.size(); i++)
        {
            Rect r = boundingRect( contours[i] );
//            rectangle(frame, r.tl(), r.br(), Scalar(0, 255, 0), 2);

            if( r.height > 80 & r.width < r.height )
//            if ( r.width > 300 | r.height > 300)
            {
                r.x -= r.width / 2;
                r.y -= r.height / 2;
                r.width += r.width;
                r.height += r.height;
                r = r & Rect( 0, 0, frame.cols, frame.rows );

                Mat roi;
                cvtColor( frame( r ), roi, COLOR_BGR2GRAY);

                std::vector<Rect> rects;

                if( roi.cols > hog.winSize.width & roi.rows > hog.winSize.height )
                    hog.detectMultiScale( roi, rects);

                for (size_t i=0; i<rects.size(); i++)
                {
                    rects[i].x += r.x;
                    rects[i].y += r.y;

                    rectangle( frame, Point(rects[i].x, rects[i].y),
                               Point(rects[i].x+rects[i].width, rects[i].y+rects[i].height),
                               Scalar( 0, 0, 255 ), 2 );
                }
            }
        }

        resize(frame, frame, Size(), 2, 2);
    }

    Rect merge(Rect firstRect, Rect secondRect) {
        int left = min(firstRect.x, secondRect.x);
        int top = min(firstRect.y, secondRect.y);
        int right = max(firstRect.x + firstRect.width, secondRect.x + secondRect.width);
        int bottom = max(firstRect.y + firstRect.height, secondRect.y + secondRect.height);
        return Rect(left, top, right - left, bottom - top);
    }

    vector <Rect> mergedRects(vector<Rect> rects) {
        LOGD("mergeRects");
        vector <Rect> mergedRects = rects;
        bool foundIntersection;

        do {
            foundIntersection = false;
            for (int i = 0; i < mergedRects.size(); i++) {
                Rect current = mergedRects[i];
                for (int j = i + 1; j < mergedRects.size(); j++) {
                    if ((current & mergedRects[j]).area() > 0) {
                        foundIntersection = true;
                        current = current | mergedRects[j];
                        mergedRects.erase(mergedRects.begin() + j);
                    }
                }
                mergedRects[i] = current;
            }
        } while (foundIntersection);
        LOGD("merged rect size: %u\n", mergedRects.size());
        return mergedRects;
    }

    // Background segmentation
    JavaVM *jvm;
    struct thread_data {
        int index;
        Mat& frame;
        Size sourceSize;
        Rect roiRect;

        thread_data(Mat& rgb) : frame(rgb) {}
    };

    struct DetectedData {
        int type;
        Rect rect;

        DetectedData() {}

        DetectedData(int detectedType, Rect detectedRect) {
            type = detectedType;
            rect = detectedRect;
        }
    };

    bool update_bg_model = true;
    bool smoothMask = true;
    int maxArea;

    Mat img, fgmask;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    //Rect roi_rect(0, 0, 300, 300); // x,y,w,h

    vector<DetectedData> getDetectedData(vector<vector<Point> > contours, Rect roi_rect) {
        LOGD("contours size: %u\n", contours.size());
        vector<DetectedData> detectedDatas;
        vector<Rect> tempAreas;

        for (int i = 0; i < contours.size(); ++i)
        {
            double area = contourArea(contours[i]);
            LOGD("area: %lf\n", area);
            if (area > MAX_AREA) {
                // Draw contours
                Rect roi = boundingRect(contours[i]);
                drawContours(img, contours, i, Scalar(0, 255, 255));
                tempAreas.push_back(roi);
            }
        }

        LOGD("temp area size: %u\n", tempAreas.size());
        vector<Rect> mergedAreas = mergedRects(tempAreas);
//        vector<Rect> mergedAreas = tempAreas;

        LOGD("Draw area");
        // Draw area
        for (int k = 0; k < mergedAreas.size(); k++) {
            DetectedData data;
            Rect caRoi = mergedAreas[k];

            if (caRoi.area() > maxArea) {
                continue;
            }

            Rect rectsIntersection = roi_rect & caRoi;
            LOGD("rectsIntersection area: %d\n", rectsIntersection.area());
            if (rectsIntersection.area() == 0) {
                LOGD("OUTSIDE\n");
                data = DetectedData(OUTSIDE, caRoi);
                rectangle(img, caRoi, Scalar(255, 0, 0), 3);
            } else if (rectsIntersection.area() == caRoi.area()) {
                LOGD("INSIDE\n");
                data = DetectedData(INSIDE, caRoi);
                rectangle(img, caRoi, Scalar(0, 0, 255), 3);
            } else {
                LOGD("OVERLAY\n");
                data = DetectedData(OVERLAY, caRoi);
                rectangle(img, caRoi, Scalar(255, 0, 255), 3);
            }
            detectedDatas.push_back(data);
        }

        return detectedDatas;
    }

    //void * bgDetect(void * threadData) {
    void bgDetect(int index, Mat& frame, thread_data *pData, jobject callback) {
        LOGD("bgDetect");
//        struct thread_data *pData = (struct thread_data *) threadData;
        vector<vector<Point> > contours;
        vector<DetectedData> detected_data;
        JNIEnv *env;

        int rows = pData->frame.rows;
        int cols = pData->frame.cols;
        bool mNeedDetach;
        double newFrameHeight = 640 * pData->frame.rows / pData->frame.cols;
        double sourceWidthScalar = 640 / (double)pData->sourceSize.width;
        double sourceHeightScalar = newFrameHeight / pData->sourceSize.height;
        LOGD("index: %d newFrameHeight: %lf, widthScalar: %lf, heightScalar: %lf\n",
             pData->index, newFrameHeight, sourceWidthScalar, sourceHeightScalar);

        Rect roi_rect((int)(pData->roiRect.x * sourceWidthScalar),
                      (int)(pData->roiRect.y * sourceHeightScalar),
                      (int)(pData->roiRect.width * sourceWidthScalar),
                      (int)(pData->roiRect.height * sourceHeightScalar));

        LOGD("JNI resized rect x: %d y: %d width: %d height: %d)\n",
             roi_rect.x, roi_rect.y, roi_rect.width, roi_rect.height);
        LOGI("frame width: %d height: %d", cols, rows);

        resize(pData->frame, img, Size(640, (int)newFrameHeight), INTER_LINEAR);
        LOGD("resized img width: %d height: %d", img.cols, img.rows);
        maxArea = img.rows * img.cols * 9 / 10;

        // Background subtraction
        bgS->apply(img, fgmask, update_bg_model ? -1 : 0);

        if (smoothMask)
        {
            GaussianBlur(fgmask, fgmask, Size(11, 11), 3.5, 3.5);
            threshold(fgmask, fgmask, 10, 255, THRESH_BINARY);
        }

        // Clean foreground from noise
        morphologyEx(fgmask, fgmask, MORPH_OPEN, kernel);

        // Find contours
        findContours(fgmask.clone(), contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
        LOGD("Contours size: %u\n", contours.size());
        if (!contours.empty())
        {
            detected_data = getDetectedData(contours, roi_rect);
        }

        rectangle(img, roi_rect.tl(), roi_rect.br(), cv::Scalar(0, 255, 0), 2);
        resize(img, pData->frame, Size(cols, rows), INTER_LINEAR);

        LOGD("Release");
        img.release();

        // Send result
        int getEnvStat = jvm->GetEnv((void **) &env, JNI_VERSION_1_6);
        LOGD("getEnvStat: %d\n", getEnvStat);
        if (getEnvStat == JNI_EDETACHED) {
            if (jvm->AttachCurrentThread(&env, NULL) != 0) {
                return;
            }
            mNeedDetach = JNI_TRUE;
        }

        LOGD("mNeedDetach: %d\n", mNeedDetach);
        if (callback == NULL) {
            LOGE("Callbcak is null.\n");
            return;
        }

        jclass cls_ArrayList = env->FindClass("java/util/ArrayList");
        jclass cls_data = env->FindClass("com/fuhu/galileocv/DetectedData");
        jclass cls_callback = env->GetObjectClass(callback);
        if(cls_ArrayList == NULL || cls_data == NULL || cls_callback == NULL)
        {
            LOGE("cls_ArrayList or cls_data is null \n");
            return;
        }

        // Get the ArrayList reference.
        jmethodID construct = env->GetMethodID(cls_ArrayList, "<init>", "()V");
        jobject obj_ArrayList = env->NewObject(cls_ArrayList, construct, "");
        jmethodID arrayList_add = env->GetMethodID(cls_ArrayList, "add", "(Ljava/lang/Object;)Z");

        LOGD("Find ArrayList\n");
        // Get the DetectedData reference.
        jmethodID data_costruct = env->GetMethodID(cls_data,
                                                           "<init>", "(IIIII)V");
        for (int i = 0; i < detected_data.size(); i++) {
            DetectedData data = detected_data[i];
            jobject obj_data = env->NewObject(cls_data, data_costruct, data.type,
                                                      data.rect.x, data.rect.y,
                                                      data.rect.width, data.rect.height);
            env->CallBooleanMethod(obj_ArrayList, arrayList_add, obj_data);
        }
        LOGD("Add data to list\n");

        // Get the Callback reference.
        jmethodID callback_method = env->GetMethodID(
                cls_callback, "onMotionDetected", "(Ljava/util/ArrayList;)V");
        env->CallVoidMethod(callback, callback_method, obj_ArrayList);

        /*

        jclass javaClass = env->GetObjectClass(callback);
        if (javaClass == 0) {
            LOGD("Unable to find class");
            jvm->DetachCurrentThread();
            return NULL;
        }


        if (javaCallbackId == NULL) {
            LOGD("Unable to find method:onProgressCallBack");
            return NULL;
        }
        //执行回调
*/
//        if(mNeedDetach) {
//            jvm->DetachCurrentThread();
//        }


        LOGD("Release env");
        env->DeleteLocalRef(cls_ArrayList);
        env->DeleteLocalRef(cls_data);
        env->DeleteLocalRef(cls_callback);
        env = NULL;
//        pthread_exit(NULL);
    }

    JNIEXPORT void JNICALL Java_com_fuhu_galileocv_MotionDetector_findFeatures(
            JNIEnv*, jobject, jlong addrGray, jlong addrRgba)
    {
        Mat& mGr  = *(Mat*)addrGray;
        Mat& mRgb = *(Mat*)addrRgba;
        vector<KeyPoint> v;

        Ptr<FeatureDetector> detector = FastFeatureDetector::create(50);
        detector->detect(mGr, v);
        LOGD("vector size: %u", v.size());

        int rows = mRgb.rows;
        int cols = mRgb.cols;
        int left = cols / 4;;
        int top = rows / 4;
        int width = cols - (left * 2);
        int height = rows - (top * 2);

//        rectangle(mRgb,
//                  Point(left, top),
//                  Point(left + width, top + height),
//                  Scalar(255, 0, 0, 255),
//                  2
//        );

        for( unsigned int i = 0; i < v.size(); i++ )
        {
            const KeyPoint& kp = v[i];
            circle(mRgb, Point((int)kp.pt.x, (int)kp.pt.y), 10, Scalar(255,0,0,255));
        }
    }

    JNIEXPORT JNICALL void Java_com_fuhu_galileocv_MotionDetector_detect(
            JNIEnv *jenv, jobject, jint index, jlong addrRgba, jobject sizeObject,
            jobject rectObject, jobject jCallback) {

        Mat& mRgb = *(Mat*)addrRgba;
        jclass sizeClass = jenv->GetObjectClass(sizeObject);
        jclass roiRectClass = jenv->GetObjectClass(rectObject);

        if (mRgb.empty()) {
            LOGE("Rgba is empty.");
            return;
        }

        if (sizeClass == NULL || roiRectClass == NULL) {
            LOGE("GetObjectClass failed\n");
            return;
        }

        // Cache JavaVM here
        jenv->GetJavaVM(&jvm);
        //callback = jenv->NewGlobalRef(jCallback);
        LOGD("index: %d\n", index);

        thread_data data = thread_data(mRgb);
        data.index = index;
        data.frame = mRgb;

        // Get the frame size.
        jfieldID widthFieldID = jenv->GetFieldID(sizeClass, "width", "D");
        jfieldID heightFieldID = jenv->GetFieldID(sizeClass, "height", "D");
        jdouble width = jenv->GetDoubleField(sizeObject, widthFieldID);
        jdouble height = jenv->GetDoubleField(sizeObject, heightFieldID);

        data.sourceSize = Size((int)width, (int)height);
        LOGD("JNI Source size: (%d, %d)\n", data.sourceSize.width, data.sourceSize.height);

        // Get the ROI.
        jfieldID roiXFieldID = jenv->GetFieldID(roiRectClass, "x", "I");
        jfieldID roiYFieldID = jenv->GetFieldID(roiRectClass, "y", "I");
        jfieldID roiWidthFieldID = jenv->GetFieldID(roiRectClass, "width", "I");
        jfieldID roiHeightFieldID = jenv->GetFieldID(roiRectClass, "height", "I");
        jint roiX = jenv->GetIntField(rectObject, roiXFieldID);
        jint roiY = jenv->GetIntField(rectObject, roiYFieldID);
        jint roiWidth = jenv->GetIntField(rectObject, roiWidthFieldID);
        jint roiHeight = jenv->GetIntField(rectObject, roiHeightFieldID);

        data.roiRect = Rect(roiX, roiY, roiWidth, roiHeight);
        LOGD("JNI ROI x: %d y: %d width: %d height: %d)\n",
             data.roiRect.x, data.roiRect.y, data.roiRect.width, data.roiRect.height);

        //pthread_create(&pthread, NULL, bgDetect, (void *) &data);

        bgDetect(index, mRgb, &data, jCallback);
//        pthread_exit(NULL);
    }

    JNIEXPORT JNICALL void Java_com_fuhu_galileocv_MotionDetector_objectTracking(
            JNIEnv* jenv, jobject, jlong addrRgba, jlong addrMotion, jstring typeString, jint index) {

        const char *cType = jenv->GetStringUTFChars(typeString, NULL);
        Mat& mRgb = *(Mat*)addrRgba;
        Mat& mMotion  = *(Mat*)addrMotion;
        hog.setSVMDetector(hog.getDefaultPeopleDetector());

        LOGD("type: %s index %d", cType, index);
//        hogDetect(mRgb, mMotion);
//        bgDetect(mRgb, mMotion);
    }

    JNIEXPORT JNICALL void Java_com_fuhu_galileocv_MotionDetector_testJNI(
            JNIEnv* jenv, jobject, jobject rectObject) {

        jclass roiRectClass = jenv->GetObjectClass(rectObject);

        if (roiRectClass == NULL) {
            LOGD("GetObjectClass failed\n");
            return;
        }

        LOGD("");

        jfieldID roiXFieldID = jenv->GetFieldID(roiRectClass, "x", "I");
        jint x = jenv->GetIntField(rectObject, roiXFieldID);

//        LOGD("JNI ROI x: %d y: %d width: %d height: %d)\n", roi.x, roi.y, roi.width, roi.height);
    }
}