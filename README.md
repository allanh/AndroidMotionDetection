# MotionDetection
Motion detection with OpenCV3

# Installation 

1. Before you can use the motion detection, OpenCV 3.3.1 Android SDK with contrib modules must be installed. 
You can [follow this link](https://github.com/chaoyangnz/opencv3-android-sdk-with-contrib) for more information how to use it in your Android project.

2. After you installed OpenCV SDK you need to set openCV SDK path into your App.
Change `OPENCV_ANDROKD_SDK` in `galileocv/src/main/jni/Android.mk`:

```
ifdef OPENCV_ANDROID_SDK
  ifneq ("","$(wildcard $(OPENCV_ANDROID_SDK)/OpenCV.mk)")
    include ${OPENCV_ANDROID_SDK}/OpenCV.mk
  else
    include ${OPENCV_ANDROID_SDK}/sdk/native/jni/OpenCV.mk
  endif
else
  # Please change this path to your opencv sdk path
  include /Users/allan.shih/Projects/opencv3-android-sdk-with-contrib/OpenCV-android-sdk/sdk/native/jni/OpenCV.mk
endif

```

# Screenshots

- BLUE rectangle indicated the object is INSIDE the safe zone (Green rectangle)
![](https://i.imgur.com/Qbr1Tmx.png)

- Purple rectangle indicated the object is OVERLAP to the safe zone
![](https://i.imgur.com/IidGoFA.png)

- Red rectangle indicated the moving object OUTSIDE the safe zone
![](https://i.imgur.com/1NdplR7.png)
