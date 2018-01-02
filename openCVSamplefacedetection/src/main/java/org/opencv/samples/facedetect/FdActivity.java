package org.opencv.samples.facedetect;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;

import com.fuhu.galileocv.GalileoCVManager;
import com.fuhu.galileocv.IVisionManager;

import org.opencv.android.CameraBridgeViewBase;

public class FdActivity extends Activity {

    private static final String    TAG                 = "OCVSample::Activity";


    private MenuItem               mItemFace50;
    private MenuItem               mItemFace40;
    private MenuItem               mItemFace30;
    private MenuItem               mItemFace20;
    private MenuItem               mItemType;

    private IVisionManager mVisionManager;
    private CameraBridgeViewBase mOpenCvCameraView;

    public FdActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);

        mVisionManager = new GalileoCVManager(this, mOpenCvCameraView);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        mVisionManager.disable();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        mVisionManager.enable();
    }

    public void onDestroy() {
        super.onDestroy();
        mVisionManager.disable();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
//        mItemFace50 = menu.add("Face size 50%");
//        mItemFace40 = menu.add("Face size 40%");
//        mItemFace30 = menu.add("Face size 30%");
//        mItemFace20 = menu.add("Face size 20%");
//        mItemType   = menu.add(mDetectorName[mDetectorType]);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
//        if (item == mItemFace50)
//            setMinFaceSize(0.5f);
//        else if (item == mItemFace40)
//            setMinFaceSize(0.4f);
//        else if (item == mItemFace30)
//            setMinFaceSize(0.3f);
//        else if (item == mItemFace20)
//            setMinFaceSize(0.2f);
//        else if (item == mItemType) {
//            int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
//            item.setTitle(mDetectorName[tmpDetectorType]);
//            setDetectorType(tmpDetectorType);
//        }
        return true;
    }

}
