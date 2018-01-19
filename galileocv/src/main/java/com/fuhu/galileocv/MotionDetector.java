package com.fuhu.galileocv;

import org.opencv.core.Rect;
import org.opencv.core.Size;

/**
 * Created by allanshih on 2017/12/21.
 */

public class MotionDetector {

    public static native void findFeatures(long matAddrGr, long matAddrRgba);

    public static native void detect(int index, long matAddrRgba, Size roiSize, Rect roiRect,
                                     MotionDetectionCallback detectionCallback);

    public static native void objectTracking(long matAddrRgba ,long matAddrMotion,
                                             String typeString, int index);
    public static native void testJNI(Rect rect);
}
