package com.fuhu.galileocv;

import java.util.ArrayList;

/**
 * Created by allanshih on 2018/1/16.
 */

public interface MotionDetectionCallback {
    public void onMotionDetected(ArrayList<DetectedData> detectedDataList);
}
