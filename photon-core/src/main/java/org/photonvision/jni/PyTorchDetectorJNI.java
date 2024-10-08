/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.jni;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.Collectors;
import org.opencv.core.Mat;
import org.opencv.core.Rect2d;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.TestUtils;
import org.photonvision.vision.pipe.impl.NeuralNetworkPipeResult;

public class PyTorchDetectorJNI extends PhotonJNICommon {
    private static final Logger logger = new Logger(PyTorchDetectorJNI.class, LogGroup.General);
    private boolean isLoaded;
    private static PyTorchDetectorJNI instance = null;

    private PyTorchDetectorJNI() {
        isLoaded = false;
    }

    public static PyTorchDetectorJNI getInstance() {
        if (instance == null) instance = new PyTorchDetectorJNI();

        return instance;
    }

    @Override
    public boolean isLoaded() {
        return isLoaded;
    }

    @Override
    public void setLoaded(boolean state) {
        isLoaded = state;
    }

    public static class PyTorchObjectDetector {
        private List<String> labels;

        public PyTorchObjectDetector(String modelPath, List<String> labels) {
            this.labels = labels;
        }

        public List<String> getClasses() {
            return labels;
        }

        /**
         * Detect forwards using this model
         *
         * @param in The image to process
         * @param nmsThresh Non-maximum supression threshold. Probably should not change
         * @param boxThresh Minimum confidence for a box to be added. Basically just confidence
         *     threshold
         */
        public List<NeuralNetworkPipeResult> detect(Mat in, double nmsThresh, double boxThresh) {
            List<NeuralNetworkPipeResult> tempResult = new ArrayList<NeuralNetworkPipeResult>();
            tempResult.add(new NeuralNetworkPipeResult(new Rect2d(10, 10, 10, 10), 0, 0.9));
            return tempResult;
        }

        public void release() {

        }
    }
}