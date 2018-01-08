package com.fuhu.galileocv;

import org.opencv.core.Rect;
import org.opencv.core.Size;

/**
 * Created by allanshih on 2017/12/29.
 */

public interface IVisionManager {
    public void enable();
    public void disable();
    public void setRoi(Size frameSize, Rect rect);
//    public void startDetecting();
//    public void stopDetecting();
    public void release();
}
