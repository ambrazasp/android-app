package com.example.ambrazasp.androidtensorflow.models;


//Provides access to an application's raw asset files;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.util.Log;
//Reads text from a character-input stream, buffering characters so as to provide for the efficient reading of characters, arrays, and lines.
import java.io.BufferedReader;
//for erros
import java.io.IOException;
//An InputStreamReader is a bridge from byte streams to character streams:
// //It reads bytes and decodes them into characters using a specified charset.
// //The charset that it uses may be specified by name or may be given explicitly, or the platform's default charset may be accepted.
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
//made by google, used as the window between android and tensorflow native C++
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

/**
 * Changed from https://github.com/MindorksOpenSource/AndroidTensorFlowMNISTExample/blob/master
 * /app/src/main/java/com/mindorks/tensorflowexample/TensorFlowImageClassifier.java
 * Created by marianne-linhares on 20/04/17.
 */

//lets create this classifer
public class TensorFlowClassifier implements Classifier {

    // Only returns if at least this confidence
    //must be a classification percetnage greater than this
    private static final float THRESHOLD = 0.1f;
    private static int MAX_RESULTS = 100;
    private TensorFlowInferenceInterface inferenceInterface;

    private String inputName;
    private int inputSize;

    private List<String> labels;
//    private float[] output;
    private String[] outputNames;
    private float[] outputLocations;
    private float[] outputScores;
    private float[] outputClasses;
    private float[] outputNumDetections;


    private byte[] byteValues;
    private int[] intValues;

    //given a saved drawn model, lets read all the classification labels that are
    //stored and write them to our in memory labels list
    private static List<String> readLabels(AssetManager am, String fileName) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(am.open(fileName)));

        String line;
        List<String> labels = new ArrayList<>();
        while ((line = br.readLine()) != null) {
            labels.add(line);
        }

        br.close();
        return labels;
    }

   //given a model, its label file, and its metadata
    //fill out a classifier object with all the necessary
    //metadata including output prediction
    public static TensorFlowClassifier create(
            final AssetManager assetManager,
            final String modelFilename,
            final String labelFilename,
            final int inputSize) throws IOException {
        //intialize a classifier
        TensorFlowClassifier c = new TensorFlowClassifier();

        //read labels for label file
        c.labels = readLabels(assetManager, labelFilename);

        c.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

        final Graph g = c.inferenceInterface.graph();

        c.inputName = "image_tensor";
        // The inputName node has a shape of [N, H, W, C], where
        // N is the batch size
        // H = W are the height and width
        // C is the number of channels (3 for our purposes - RGB)
        final Operation inputOp = g.operation(c.inputName);
        if (inputOp == null) {
            throw new RuntimeException("Failed to find input Node '" + c.inputName + "'");
        }
        c.inputSize = inputSize;
        // The outputScoresName node has a shape of [N, NumLocations], where N
        // is the batch size.
        final Operation outputOp1 = g.operation("detection_scores");
        if (outputOp1 == null) {
            throw new RuntimeException("Failed to find output Node 'detection_scores'");
        }
        final Operation outputOp2 = g.operation("detection_boxes");
        if (outputOp2 == null) {
            throw new RuntimeException("Failed to find output Node 'detection_boxes'");
        }
        final Operation outputOp3 = g.operation("detection_classes");
        if (outputOp3 == null) {
            throw new RuntimeException("Failed to find output Node 'detection_classes'");
        }

        c.outputNames = new String[] {"detection_boxes", "detection_scores",
                "detection_classes", "num_detections"};
        c.intValues = new int[c.inputSize * c.inputSize];
        c.byteValues = new byte[c.inputSize * c.inputSize * 3];
        c.outputScores = new float[MAX_RESULTS];
        c.outputLocations = new float[MAX_RESULTS * 4];
        c.outputClasses = new float[MAX_RESULTS];
        c.outputNumDetections = new float[1];
        return c;
    }


    @Override
    public List<Classification> recognize(final Bitmap bitmap) {

        // convert to pixels array
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            byteValues[i * 3 + 2] = (byte) (intValues[i] & 0xFF);
            byteValues[i * 3 + 1] = (byte) ((intValues[i] >> 8) & 0xFF);
            byteValues[i * 3 + 0] = (byte) ((intValues[i] >> 16) & 0xFF);
        }

        inferenceInterface.feed(inputName, byteValues, 1, inputSize, inputSize, 3);

        inferenceInterface.run(outputNames);

        //get the output
        if(outputNames.length != 4)
            return null;

        outputLocations = new float[MAX_RESULTS * 4];
        outputScores = new float[MAX_RESULTS];
        outputClasses = new float[MAX_RESULTS];
        outputNumDetections = new float[1];
        inferenceInterface.fetch(outputNames[0], outputLocations);
        inferenceInterface.fetch(outputNames[1], outputScores);
        inferenceInterface.fetch(outputNames[2], outputClasses);
        inferenceInterface.fetch(outputNames[3], outputNumDetections);

        PriorityQueue<Classification> pq =
            new PriorityQueue<Classification>(
                1,
                new Comparator<Classification>() {
                    @Override
                    public int compare(Classification lhs, Classification rhs) {
                        // Intentionally reversed to put high confidence at the head of the queue.
                        return Float.compare(rhs.getConf(), lhs.getConf());
                    }
                });
        for (int i = 0; i < outputScores.length; ++i){
            final RectF detection =
                    new RectF(
                            outputLocations[4 * i + 1] * inputSize,
                            outputLocations[4 * i] * inputSize * 1.3333333f,
                            outputLocations[4 * i + 3] * inputSize,
                            outputLocations[4 * i + 2] * inputSize * 1.3333333f);
            pq.add(new Classification(i, outputScores[i], labels.get((int) outputClasses[i]), detection));
        }

        final ArrayList<Classification> recognitions = new ArrayList<Classification>();
        for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
            recognitions.add(pq.poll());
        }

        return recognitions;
    }
}
