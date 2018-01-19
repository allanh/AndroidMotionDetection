package com.fuhu.galileocv;

import org.opencv.core.Rect;

/**
 * Created by allanshih on 2018/1/18.
 */

public class DetectedData {
    private int type;
    private Rect rect;

    public DetectedData() {}

    public DetectedData(int detectedType, Rect rect) {
        this.type = detectedType;
        this.rect = rect;
    }

    public DetectedData(int detectedType, int x, int y, int width, int height) {
        this.type = detectedType;
        this.rect = new Rect(x, y, width, height);
    }

    public int getType() {
        return type;
    }

    public void setType(int type) {
        this.type = type;
    }

    public Rect getRect() {
        return rect;
    }

    public void setRect(Rect rect) {
        this.rect = rect;
    }
}
