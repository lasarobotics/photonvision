package org.photonvision.vision.pipeline;

public class CustomMLDetectionPipelineSettings extends AdvancedPipelineSettings {
    public double confidence;

    public CustomMLDetectionPipelineSettings() {
        super();
        this.pipelineType = PipelineType.CustomMLDetection;
        this.outputShowMultipleTargets = true;
        cameraExposure = 20;
        cameraAutoExposure = false;
        ledMode = false;
        confidence = .9;
    }
}