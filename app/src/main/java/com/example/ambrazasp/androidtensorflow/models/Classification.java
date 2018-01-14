package com.example.ambrazasp.androidtensorflow.models;

import android.graphics.RectF;

/**
 * Created by marianne-linhares on 20/04/17.
 */

public class Classification {

    //conf is the output
    private int id;
    private float conf;
    //input label
    private String label;
    private RectF location;

    Classification(int id ) {
        this.conf = -1.0F;
        this.label = null;
        this.id = id;
    }
    Classification(int id, float conf, String label, RectF location) {
        this.conf = conf;
        this.label = label;
        this.id = id;
        this.location = location;
    }

    void update(float conf, String label) {
        this.conf = conf;
        this.label = label;
    }

    public String getLabel() {
        return label;
    }

    public float getConf() {
        return conf;
    }

    public RectF getLocation() {
        return new RectF(location);
    }
}
