# LPRNet Optimization Project

This repository contains work focused on optimizing the **LPRNet PyTorch model** for efficient vehicle license plate recognition. The objective was to enhance the model's performance by reducing its size, improving inference speed, and maintaining high accuracy.

## Project Overview

In this project, various optimization techniques were applied to the LPRNet model to achieve the following goals:

- **Model Compression**:
  - Fine-grained pruning to eliminate redundant weights and reduce model complexity.
  - Static quantization to lower the precision and size of the model while preserving its predictive capabilities.

- **Model Compilation**:
  - Graph optimization and compilation using tools such as TVM Auto-Scheduler for inference speedup.
## Objectives

- **Model Size**: Achieve a significant reduction in model size.
- **Inference Speed**: Enhance the speed of inference on platforms like Google Colab.
- **Accuracy**: Ensure high test accuracy is maintained or improved post-optimization.

## Tools and Techniques

- **Optimization Methods**:
  - Fine-grained pruning
  - Static quantization
  - Graph Optimization
  - TVM Auto-Schedulert

- **Evaluation Metrics**:
  - **Accuracy**: Test accuracy on vehicle license plate datasets.
  - **Inference Speed**: Speed improvement measured on Google Colab and Android platforms.
  - **Model Size**: ONNX model size before and after optimizations.

## Results and Insights

The project successfully demonstrated:
- A reduction in the model's size through pruning and quantization.
- Noticeable improvements in inference speed on both Google Colab and Android devices.
- High accuracy maintained, comparable to or better than the original LPRNet model.
