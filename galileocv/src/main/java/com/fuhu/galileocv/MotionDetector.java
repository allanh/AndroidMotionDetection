package com.fuhu.galileocv;

/**
 * Created by allanshih on 2017/12/21.
 */

public class MotionDetector {

    public static native void findFeatures(long matAddrGr, long matAddrRgba);

    public static native void detect(long matAddrRgba ,long matAddrMotion);

    public static native void objectTracking(long matAddrRgba ,long matAddrMotion, String typeString, int index);

}
