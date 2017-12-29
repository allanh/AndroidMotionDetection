LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

#OPENCV_INSTALL_MODULES:=off
#OPENCV_LIB_TYPE:=SHARED
ifdef OPENCV_ANDROID_SDK
  ifneq ("","$(wildcard $(OPENCV_ANDROID_SDK)/OpenCV.mk)")
    include ${OPENCV_ANDROID_SDK}/OpenCV.mk
  else
    include ${OPENCV_ANDROID_SDK}/sdk/native/jni/OpenCV.mk
  endif
else
  include /Users/allanshih/Opencv/opencv3-android-sdk-with-contrib/OpenCV-android-sdk/sdk/native/jni/OpenCV.mk
endif

LOCAL_SRC_FILES  := motion_detection_jni.cpp
# DetectionBasedTracker_jni.cpp
LOCAL_C_INCLUDES += $(LOCAL_PATH)
LOCAL_LDLIBS     += -llog -ldl

LOCAL_MODULE     := motion_detection
# detection_based_tracker

include $(BUILD_SHARED_LIBRARY)
