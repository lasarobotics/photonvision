package org.photonvision.vision.pipe.impl;

import java.awt.Color;
import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.photonvision.common.configuration.NeuralNetworkModelManager;
import org.photonvision.common.util.ColorHelper;
import org.photonvision.jni.PyTorchDetectorJNI.PyTorchObjectDetector;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.opencv.Releasable;
import org.photonvision.vision.pipe.CVPipe;

public class PyTorchDetectionPipe 
        extends CVPipe<CVMat, List<NeuralNetworkPipeResult>, PyTorchDetectionPipe.PyTorchDetectionPipeParams>
        implements Releasable {
    private PyTorchObjectDetector detector;

    public PyTorchDetectionPipe() {
        // For now this is hard-coded to defaults. Should be refactored into set pipe
        // params, though.
        // And ideally a little wrapper helper for only changing native stuff on content
        // change created.
        this.detector = new PyTorchObjectDetector("bleh", NeuralNetworkModelManager.getInstance().getLabels());
    }

    private static class Letterbox {
        double dx;
        double dy;
        double scale;

        public Letterbox(double dx, double dy, double scale) {
            this.dx = dx;
            this.dy = dy;
            this.scale = scale;
        }
    }

    @Override
    protected List<NeuralNetworkPipeResult> process(CVMat in) {
        var frame = in.getMat();

        // Make sure we don't get a weird empty frame
        if (frame.empty()) {
            return List.of();
        }

        // // letterbox
        var letterboxed = new Mat();
        var scale =
                letterbox(frame, letterboxed, new Size(640, 640), ColorHelper.colorToScalar(Color.GRAY));

        if (letterboxed.width() != 640 || letterboxed.height() != 640) {
            // huh whack give up lol
            throw new RuntimeException("RGA bugged but still wrong size");
        }
        var ret = detector.detect(letterboxed, params.nms, params.confidence);
        letterboxed.release();

        return resizeDetections(ret, scale);
    }

    private List<NeuralNetworkPipeResult> resizeDetections(
            List<NeuralNetworkPipeResult> unscaled, Letterbox letterbox) {
        var ret = new ArrayList<NeuralNetworkPipeResult>();

        for (var t : unscaled) {
            var scale = 1.0 / letterbox.scale;
            var boundingBox = t.bbox;
            double x = (boundingBox.x - letterbox.dx) * scale;
            double y = (boundingBox.y - letterbox.dy) * scale;
            double width = boundingBox.width * scale;
            double height = boundingBox.height * scale;

            ret.add(
                    new NeuralNetworkPipeResult(new Rect2d(x, y, width, height), t.classIdx, t.confidence));
        }

        return ret;
    }

    private static Letterbox letterbox(Mat frame, Mat letterboxed, Size newShape, Scalar color) {
        // from https://github.com/ultralytics/yolov5/issues/8427#issuecomment-1172469631
        var frameSize = frame.size();
        var r = Math.min(newShape.height / frameSize.height, newShape.width / frameSize.width);

        var newUnpad = new Size(Math.round(frameSize.width * r), Math.round(frameSize.height * r));

        if (!(frameSize.equals(newUnpad))) {
            Imgproc.resize(frame, letterboxed, newUnpad, Imgproc.INTER_LINEAR);
        } else {
            frame.copyTo(letterboxed);
        }

        var dw = newShape.width - newUnpad.width;
        var dh = newShape.height - newUnpad.height;

        dw /= 2;
        dh /= 2;

        int top = (int) (Math.round(dh - 0.1f));
        int bottom = (int) (Math.round(dh + 0.1f));
        int left = (int) (Math.round(dw - 0.1f));
        int right = (int) (Math.round(dw + 0.1f));
        Core.copyMakeBorder(
                letterboxed, letterboxed, top, bottom, left, right, Core.BORDER_CONSTANT, color);

        return new Letterbox(dw, dh, r);
    }

    public static class PyTorchDetectionPipeParams {
        public double confidence;
        public double nms;
        public int max_detections;

        public PyTorchDetectionPipeParams() {}
    }

    public List<String> getClassNames() {
        return detector.getClasses();
    }

    @Override
    public void release() {
        detector.release();
    }
}