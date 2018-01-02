package com.fuhu.galileocv;

import android.content.Context;
import android.util.Log;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * Created by allanshih on 2017/12/29.
 */

public class GalileoCVManager implements IVisionManager, CvCameraViewListener2 {

    private static final String     TAG                 = GalileoCVManager.class.getSimpleName();
    private static final Scalar     FACE_RECT_COLOR     = new Scalar(0, 255, 0, 255);
    public static final int         JAVA_DETECTOR       = 0;
    public static final int         NATIVE_DETECTOR     = 1;

    private Context                 mContext;
    private Mat                     mRgba;
    private Mat                     mGray;
    private File                    mCascadeFile;
    private CascadeClassifier       mJavaDetector;
//    private DetectionBasedTracker  mNativeDetector;

    private int                     mDetectorType       = JAVA_DETECTOR;
    private String[]                mDetectorName;
    private boolean                 isDetecting;

    private float                   mRelativeFaceSize   = 0.2f;
    private int                     mAbsoluteFaceSize   = 0;

    private CameraBridgeViewBase    mOpenCvCameraView;
    private MotionDetector          mMotionDetector;

    private String []               trackerTypes = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN"};

    private int                     index;

    private BaseLoaderCallback      mLoaderCallback;

    private long                    mStartedTimeMs;

    public GalileoCVManager(Context context, CameraBridgeViewBase openCvCameraView) {
        this.mContext = context;
        this.mOpenCvCameraView = openCvCameraView;

        this.mDetectorName = new String[2];
        this.mDetectorName[JAVA_DETECTOR] = "Java";
        this.mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";
        this.mOpenCvCameraView.setCvCameraViewListener(this);
    }


    public void enable() {
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, mContext, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            if (mLoaderCallback == null) {
                mLoaderCallback = new BaseLoaderCallback(mContext) {
                    @Override
                    public void onManagerConnected(int status) {
                        switch (status) {
                            case LoaderCallbackInterface.SUCCESS: {
                                Log.i(TAG, "OpenCV loaded successfully");

                                // Load native library after(!) OpenCV initialization
                                System.loadLibrary("motion_detection");

                                try {
                                    // load cascade file from application resources
                                    InputStream is = mContext.getResources().openRawResource(R.raw.lbpcascade_frontalface);
                                    File cascadeDir = mContext.getDir("cascade", Context.MODE_PRIVATE);
                                    mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                                    FileOutputStream os = new FileOutputStream(mCascadeFile);

                                    byte[] buffer = new byte[4096];
                                    int bytesRead;
                                    while ((bytesRead = is.read(buffer)) != -1) {
                                        os.write(buffer, 0, bytesRead);
                                    }
                                    is.close();
                                    os.close();

                                    mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                                    if (mJavaDetector.empty()) {
                                        Log.e(TAG, "Failed to load cascade classifier");
                                        mJavaDetector = null;
                                    } else
                                        Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

//                        mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);
                                    mMotionDetector = new MotionDetector();
                                    index = 0;
                                    cascadeDir.delete();

                                } catch (IOException e) {
                                    e.printStackTrace();
                                    Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                                }

                                mOpenCvCameraView.enableView();
                                isDetecting = true;
                            }
                            break;
                            default: {
                                super.onManagerConnected(status);
                            }
                            break;
                        }
                    }
                };
            }
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void disable() {
        isDetecting = false;
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

//    public void startDetecting() {
//        isDetecting = true;
//    }
//
//    public void stopDetecting() {
//        isDetecting = false;
//    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

//        Mat motion = new Mat();
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        if (isDetecting) {
            mStartedTimeMs = System.currentTimeMillis();
//        mMotionDetector.findFeatures(mGray.getNativeObjAddr(), mRgba.getNativeObjAddr());
//        mMotionDetector.objectTracking(mRgba.getNativeObjAddr(), motion.getNativeObjAddr(),
//                "BOOSTING", index++);
            mMotionDetector.detect(mRgba.getNativeObjAddr(), mRgba.getNativeObjAddr());

//        if (mAbsoluteFaceSize == 0) {
//            int height = mGray.rows();
//            if (Math.round(height * mRelativeFaceSize) > 0) {
//                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
//            }
////            mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
//        }
//
//        MatOfRect faces = new MatOfRect();
//
//        if (mDetectorType == JAVA_DETECTOR) {
//            if (mJavaDetector != null)
//                mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
//                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
//        }
//        else if (mDetectorType == NATIVE_DETECTOR) {
////            if (mNativeDetector != null)
////                mNativeDetector.detect(mGray, faces);
//        }
//        else {
//            Log.e(TAG, "Detection method is not selected!");
//        }
//
//        Rect[] facesArray = faces.toArray();
//        for (int i = 0; i < facesArray.length; i++)
//            Imgproc.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
//
//        motion.release();
            Log.d(TAG, " completed: delay= " + (System.currentTimeMillis() - mStartedTimeMs) + "ms");
        }

        return mRgba;
    }

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    private void setDetectorType(int type) {
        if (mDetectorType != type) {
            mDetectorType = type;

//            if (type == NATIVE_DETECTOR) {
//                Log.i(TAG, "Detection Based Tracker enabled");
//                mNativeDetector.start();
//            } else {
//                Log.i(TAG, "Cascade detector enabled");
//                mNativeDetector.stop();
//            }
        }
    }

    public void release() {

    }
}
