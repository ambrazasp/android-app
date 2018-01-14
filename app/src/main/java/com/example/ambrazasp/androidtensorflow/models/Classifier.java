package com.example.ambrazasp.androidtensorflow.models;

import android.graphics.Bitmap;

import java.util.List;

/**
 * Created by Piasy{github.com/Piasy} on 29/05/2017.
 */

//public interface for the classifer
    //exposes its name and the recognize function
    //which given some drawn pixels as input
    //classifies what it sees as an MNIST image
public interface Classifier {
    List<Classification> recognize(final Bitmap image);
}
