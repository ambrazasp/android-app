/*
* Copyright 2015 The Android Open Source Project
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

package com.example.ambrazasp.androidtensorflow.camera;

import android.content.Context;
import android.hardware.Camera;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Camera preview that displays a {@link Camera}.
 *
 * Handles basic lifecycle methods to display and stop the preview.
 * <p>
 * Implementation is based directly on the documentation at
 * http://developer.android.com/guide/topics/media/camera.html
 */
public class CameraPreview extends SurfaceView implements SurfaceHolder.Callback {

    private static final String TAG = "CameraPreview";
    private SurfaceHolder mHolder;
    private Camera mCamera;
    private Camera.CameraInfo mCameraInfo;
    private int mDisplayOrientation;
    private int INPUT_SIZE = 300;

    public CameraPreview(Context context, Camera camera, Camera.CameraInfo cameraInfo,
            int displayOrientation) {
        super(context);

        if (camera == null || cameraInfo == null) {
            return;
        }
        mCamera = camera;
        mCameraInfo = cameraInfo;
        mDisplayOrientation = displayOrientation;

        mHolder = getHolder();
        mHolder.addCallback(this);
    }

    public void surfaceCreated(SurfaceHolder holder) {
        Camera.Parameters params = mCamera.getParameters();
        List<Camera.Size> sizes = params.getSupportedPictureSizes();
        List<Camera.Size> sizesP = params.getSupportedPreviewSizes();


        for (int i = 0; i<sizes.size(); i++)
            if (sizes.get(i).height == 480 && sizes.get(i).width == 640)
                params.setPictureSize(sizes.get(i).width, sizes.get(i).height);

        for (int i = 0; i<sizesP.size(); i++)
            if (sizesP.get(i).height == 1200 && sizesP.get(i).width == 1600)
                params.setPreviewSize(sizesP.get(i).width, sizesP.get(i).height);

        mCamera.setParameters(params);
        mCamera.startPreview();

        Log.d(TAG, "Camera preview started.");
    }

    public void surfaceDestroyed(SurfaceHolder holder) {
        // empty. Take care of releasing the Camera preview in your activity.
    }

    public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {

        if (mHolder.getSurface() == null) {
            Log.d(TAG, "Preview surface does not exist");
            return;
        }

        try {
            mCamera.stopPreview();
            Log.d(TAG, "Preview stopped.");

            int orientation = calculatePreviewOrientation(mCameraInfo, mDisplayOrientation);
            mCamera.setDisplayOrientation(orientation);

            mCamera.setPreviewDisplay(mHolder);
            mCamera.startPreview();
            Log.d(TAG, "Camera preview started.");

        } catch (Exception e) {
            Log.d(TAG, "Error starting camera preview: " + e.getMessage());
        }


    }

    public static int calculatePreviewOrientation(Camera.CameraInfo info, int rotation) {
        int degrees = 0;

        switch (rotation) {
            case Surface.ROTATION_0:
                degrees = 0;
                break;
            case Surface.ROTATION_90:
                degrees = 90;
                break;
            case Surface.ROTATION_180:
                degrees = 180;
                break;
            case Surface.ROTATION_270:
                degrees = 270;
                break;
        }

        int result;
        if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
            result = (info.orientation + degrees) % 360;
            result = (360 - result) % 360;  // compensate the mirror
        } else {  // back-facing
            result = (info.orientation - degrees + 360) % 360;
        }

        return result;
    }
}
