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

#define SSTR( x ) static_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()

extern "C" {
    // various tracking parameters (in seconds)
    float MHI_DURATION = 0.05;
    float MAX_TIME_DELTA = 12500.0;
    float MIN_TIME_DELTA = 5;

//    const double MHI_DURATION = 1;
//    const double MAX_TIME_DELTA = 0.5;
//    const double MIN_TIME_DELTA = 0.05;
    // number of cyclic frame buffer used for motion detection
    // (should, probably, depend on FPS)
    const int N = 4;

    // ring image buffer
    vector<Mat> buf(N);
    int last=0;

    Mat mhi; // MHI
    HOGDescriptor hog;

    Ptr<BackgroundSubtractor> bgS = createBackgroundSubtractorMOG2();

    // parameters:
    //  img - input video frame
    //  dst - resultant motion picture
    //  args - optional parameters
    void  update_mhi( const Mat& img, Mat& dst, int diff_threshold )
    {
        double timestamp = (double)clock()/CLOCKS_PER_SEC; // get current time in seconds
        int idx1 = last, idx2;
        Mat tmp, silh, orient, mask, segmask;

        cvtColor( img, buf[last], CV_BGR2GRAY ); // convert frame to grayscale

        idx2 = (last + 1) % N; // index of (last - (N-1))th frame
        last = idx2;

        if( buf[idx1].size() != buf[idx2].size() )
            silh = Mat::ones(img.size(), CV_8U)*255;
        else
            absdiff(buf[idx1], buf[idx2], silh); // get difference between frames

        threshold( silh, silh, diff_threshold, 1, CV_THRESH_BINARY ); // and threshold it
        if( mhi.empty() )
            mhi = Mat::zeros(silh.size(), CV_32F);
        updateMotionHistory( silh, mhi, timestamp, MHI_DURATION ); // update MHI

        // convert MHI to blue 8u image
        mhi.convertTo(mask, CV_8U, 255./MHI_DURATION,
                      (MHI_DURATION - timestamp)*255./MHI_DURATION);
        tmp = Mat::zeros(mask.size(), CV_8UC3);
        insertChannel(mask, tmp, 0);

        // calculate motion gradient orientation and valid orientation mask
        calcMotionGradient( mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3 );

        // segment motion: get sequence of motion components
        // segmask is marked motion components map. It is not used further
        vector<Rect> brects;
        segmentMotion(mhi, segmask, brects, timestamp, MAX_TIME_DELTA );

        LOGD("brects size: %d", brects.size());

        // iterate through the motion components,
        // One more iteration (i == -1) corresponds to the whole image (global motion)
        for( int i = -1; i < (int)brects.size(); i++ ) {
            Rect roi; Scalar color; double magnitude;
            Mat maski = mask;
            if( i < 0 ) { // case of the whole image
                roi = Rect(0, 0, img.cols, img.rows);
                color = Scalar::all(255);
                magnitude = 100;
            }
            else { // i-th motion component
                roi = brects[i];

                if (roi.area() < 3000) { // reject very small components
                    continue;
                } else {
                    color = Scalar(0, 0, 255);
                    magnitude = 30;
                    maski = mask(roi);
                    LOGD("roi.area(): %d", roi.area());

                    rectangle(img, roi.tl(), roi.br(), Scalar(0, 255, 0), 2);
                }
            }



//            // calculate orientation
//            double angle = calcGlobalOrientation( orient(roi), maski, mhi(roi), timestamp, MHI_DURATION);
//            angle = 360.0 - angle;  // adjust for images with top-left origin
//
//            int count = norm( silh, NORM_L1 ); // calculate number of points within silhouette ROI
//            // check for the case of little motion
//            if( count < roi.area() * 0.05 )
//                continue;
//
//            // draw a clock with arrow indicating the direction
//            Point center( roi.x + roi.width/2, roi.y + roi.height/2 );
//            circle( img, center, cvRound(magnitude*1.2), color, 3, CV_AA, 0 );
//            line( img, center, Point( cvRound( center.x + magnitude*cos(angle*CV_PI/180)),
//                                      cvRound( center.y - magnitude*sin(angle*CV_PI/180))), color, 3, CV_AA, 0 );

            maski.release();
        }

        tmp.release();
        silh.release();
        orient.release();
        mask.release();
        segmask.release();
    }

    void hogDetect(Mat& frame, Mat& output) {//        }
//        if (frame.cols > 800) {
            resize(frame, frame, Size(), 0.5, 0.5);
//        }

        bgS->apply(frame, output);
        erode(output,output,Mat());

        // Find contours
        vector<vector<Point> > contours;
        findContours( output, contours, RETR_LIST, CHAIN_APPROX_SIMPLE );
        LOGD("countours size: %d", contours.size());

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
        vector <Rect> mergedRects = rects;
        bool foundIntersection = false;

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

        return mergedRects;
    }


    // Background segmentation

    bool update_bg_model = true;
    bool smoothMask = true;
    int maxArea;

    Mat img, fgmask;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    //Rect roi_rect(0, 0, 300, 300); // x,y,w,h

    void bgDetect(Mat& frame, Size sourceSize, Rect originRoiRect) {
        int rows = frame.rows;
        int cols = frame.cols;
        double newFrameHeight = 640 * frame.rows / frame.cols;
        double sourceWidthScalar = 640 / (double)sourceSize.width;
        double sourceHeightScalar = newFrameHeight / sourceSize.height;
        LOGD("newFrameHeight: %lf, widthScalar: %lf, heightScalar: %lf\n",
             newFrameHeight, sourceWidthScalar, sourceHeightScalar);

        Rect roi_rect(originRoiRect.x * sourceWidthScalar,
                      originRoiRect.y * sourceHeightScalar,
                      originRoiRect.width * sourceWidthScalar,
                      originRoiRect.height * sourceHeightScalar);

        LOGD("JNI resized rect x: %d y: %d width: %d height: %d)\n",
             roi_rect.x, roi_rect.y, roi_rect.width, roi_rect.height);
        LOGI("frame width: %d height: %d", cols, rows);

        resize(frame, img, Size(640, newFrameHeight), INTER_LINEAR);
        LOGD("resized img width: %d height: %d", img.cols, img.rows);
        maxArea = img.rows * img.cols * 9 / 10;

//        if (fgimg.empty())
//            fgimg.create(img.size(), img.type());

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
        vector<vector<Point> > contours;
        findContours(fgmask.clone(), contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
        if (!contours.empty())
        {
            vector<Rect> tempAreas;
            for (int i = 0; i < contours.size(); ++i)
            {
                double area = contourArea(contours[i]);
                if (area > 300) {
                    // Draw contours
                    Rect roi = boundingRect(contours[i]);
                    drawContours(img, contours, i, Scalar(0, 255, 255));
                    tempAreas.push_back(roi);
                    /*
                    if ((roi_rect & roi).area() > 0) {
                    rectangle(img, roi, Scalar(255, 0, 255), 3);
                    }
                    else {
                    rectangle(img, roi, Scalar(255, 0, 0), 3);
                    }
                    */
                }
            }

            vector <Rect> mergedAreas = mergedRects(tempAreas);

            // Draw area
            for (int k = 0; k < mergedAreas.size(); k++) {
                Rect caRoi = mergedAreas[k];

                if (caRoi.area() > maxArea) {
                    continue;
                }

                if ((roi_rect & caRoi).area() > 0) {
                    rectangle(img, caRoi, Scalar(255, 0, 255), 3);
                }
                else {
                    rectangle(img, caRoi, Scalar(0, 0, 255), 3);
                }
            }
        }

//        fgimg = Scalar::all(0);
//        roi_img.copyTo(fgimg, fgmask);

//        Mat bgimg;
//        bgS->getBackgroundImage(bgimg);

        //size_str << "width: " << img0.cols << " height: " << img0.rows;
        //putText(img, size_str.str(), Point(70, 70), CV_FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2, 8, false);

        rectangle(img, roi_rect.tl(), roi_rect.br(), cv::Scalar(0, 255, 0), 2);

        resize(img, frame, Size(cols, rows), INTER_LINEAR);
        img.release();
    }

    JNIEXPORT void JNICALL Java_com_fuhu_galileocv_MotionDetector_findFeatures(
            JNIEnv*, jobject, jlong addrGray, jlong addrRgba)
    {
        Mat& mGr  = *(Mat*)addrGray;
        Mat& mRgb = *(Mat*)addrRgba;
        vector<KeyPoint> v;

        Ptr<FeatureDetector> detector = FastFeatureDetector::create(50);
        detector->detect(mGr, v);
        LOGD("vector size: %d", v.size());

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
            circle(mRgb, Point(kp.pt.x, kp.pt.y), 10, Scalar(255,0,0,255));
        }
    }

    JNIEXPORT JNICALL void Java_com_fuhu_galileocv_MotionDetector_detect(
            JNIEnv *jenv, jobject, jlong addrRgba, jint sourceWidth, jint sourceHeight,
            jint roiX, jint roiY, jint roiWidth, jint roiHeight) {

        Mat& mRgb = *(Mat*)addrRgba;
//        jclass roiRectClass = jenv->GetObjectClass(roiRect);

        if (mRgb.empty()) {
            return;
        }

//        if (roiRectClass == NULL) {
//            LOGE("GetObjectClass failed\n");
//            return;
//        }

        // Get the source frame size.
        Size sourceSize = Size(sourceWidth, sourceHeight);
        LOGD("JNI Source size: (%d, %d)\n", sourceSize.width, sourceSize.height);

        // Get the ROI.
//        jfieldID roiXFieldID = jenv->GetFieldID(roiRectClass, "x", "I");
//        jfieldID roiYFieldID = jenv->GetFieldID(roiRectClass, "y", "I");
//        jfieldID roiWidthFieldID = jenv->GetFieldID(roiRectClass, "width", "I");
//        jfieldID roiHeightFieldID = jenv->GetFieldID(roiRectClass, "height", "I");
//        jint x = jenv->GetIntField(roiRectClass, roiXFieldID);
//        jint y = jenv->GetIntField(roiRectClass, roiYFieldID);
//        jint width = jenv->GetIntField(roiRectClass, roiWidthFieldID);
//        jint height = jenv->GetIntField(roiRectClass, roiHeightFieldID);

        Rect roi = Rect(roiX, roiY, roiWidth, roiHeight);
        LOGD("JNI ROI x: %d y: %d width: %d height: %d)\n", roi.x, roi.y, roi.width, roi.height);

        bgDetect(mRgb, sourceSize, roi);
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
}