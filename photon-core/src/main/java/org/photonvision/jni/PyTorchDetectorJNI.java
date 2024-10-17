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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

import org.opencv.core.Mat;
import org.opencv.core.Rect2d;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
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
        ProcessBuilder processBuilder;
        Process pythonScript;

        public PyTorchObjectDetector(String modelPath, List<String> labels) {
            // Path to your Python script
            String scriptPath = "../test.py";

            processBuilder = new ProcessBuilder("python", scriptPath);

            CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
                try {
                    // Start the process
                    pythonScript = processBuilder.start();

                    // Read and print the output of the Python script
                    BufferedReader reader = new BufferedReader(new InputStreamReader(pythonScript.getInputStream()));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        System.out.println(line);
                    }

                    // Wait for the process to complete
                    int exitCode = pythonScript.waitFor();
                    System.out.println("Python script exited with code: " + exitCode);

                } catch (IOException | InterruptedException e) {
                    e.printStackTrace();
                }
            });

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
            tempResult.add(new NeuralNetworkPipeResult(new Rect2d(10, 10, 50, 60), 0, 0.95));
            return tempResult;
        }

        public void release() {
            if (pythonScript != null && pythonScript.isAlive()) {
                pythonScript.destroy();
            }
        }
    }
}