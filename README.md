# AEFFNET
yolo for light
# AEFFNet for brake-light detection

*Accompanying manuscript — AEFFNet: A Task-Oriented Illumination-Aware Feature Fusion Network for Real-Time Brake-Light Perception in Intelligent Vehicles.*

AEFFNet (Adaptive Enhanced Feature Fusion Network) is a one-stage detector built on YOLOv11-N. It locates vehicle brake lights and classifies their state as **running** or **braking**. Distant lamps occupy few pixels, their outlines vary with viewpoint, and nearby reflections can resemble an illuminated lamp.

The final detector in the paper is **AEFFNet-PD**, the pruned and distilled student. **AEFFNet-N** is its uncompressed teacher. The paper reports the student as **AEFFNet (Ours)** in the main TLD comparison.

## What changes from YOLOv11-N?

| Component                                                    | Change in the network                                        | Purpose in brake-light detection                             |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **DCAC** · Directional and Contrast-Guided Adaptive Convolution | Four depthwise filters (1×3, 3×1, 1×7, 7×1) capture short and long edges. A fifth branch subtracts a local 3×3 mean to isolate detail. Local contrast, horizontal and vertical Sobel responses, and pooled context guide a softmax that mixes the five outputs at each location. | Gives a small lamp different amounts of directional context depending on its outline and the brightness of its surroundings. |
| **WCA** · Window Convolutional Attention                     | A fixed 13×13 kernel supplies wider context and small dynamic kernels adapt to local patterns. The convolutional attention operates on one quarter of the channels. Its output is concatenated with the remaining channels and fused by a 1×1 convolution. | Relates a bright patch to the surrounding vehicle while limiting the cost of processing the wider neighborhood. |
| **EDF** · Enhanced Dynamic Feature Fusion                    | Learnable frequency weights and regularized inverse filtering with Converse2D refine the high-resolution P2 features. Scaled nearest-neighbor upsampling and space-to-depth operations align pyramid features. A multilayer perceptron and softmax weight the aligned features across scales. | Combines fine lamp boundaries with the vehicle context present in coarser feature maps. |
| **RPD** · Re-Parameterized Detection Head                    | After group normalization, the head shares four training branches across scales. They use a 1×1 convolution, a 1×1 followed by a 3×3 convolution, average pooling, and a 3×3 convolution. Compatible branches can be fused for inference. | Supplies features for bounding-box regression and brake-state classification with a smaller head. |

## How the final student is trained

1. **Train the teacher.** Train YOLOv11-N with all four modules on the TLD training split. Keep this AEFFNet-N checkpoint as both the starting point for pruning and the fixed teacher for distillation.
2. **Prune linked channels.** Score channels using layer-adaptive magnitude-based pruning (LAMP), rank the scores globally, and remove dependent channels together so connected layers retain compatible shapes. The configured pruning target is `speed_up=1.5` over 50 iterations.
3. **Distill the student.** Use the raw pruned checkpoint to initialize the student. The student learns from ground-truth detections and the frozen teacher’s class responses and box predictions on the same images. The BCKD loss has a constant weight of 1.0. A separate supervised fine-tuning run starts from the same pruned checkpoint. Only the distilled student is used for final inference.

## What the ablation shows

Selected rows from Table 4 follow one cumulative sequence on the same TLD validation split. P is precision, R is recall, and `mAP@0.5` uses an intersection-over-union threshold of 0.5.

| Configuration                             |     P (%) |     R (%) | `mAP@0.5` (%) | Parameters (M) |  GFLOPs |
| ----------------------------------------- | --------: | --------: | ------------: | -------------: | ------: |
| YOLOv11-N                                 |     89.26 |     88.02 |         94.58 |           2.58 |     6.3 |
| + DCAC                                    |     89.83 |     89.08 |         95.03 |           2.63 |     6.6 |
| + DCAC, WCA                               |     90.39 |     90.44 |         95.03 |           2.83 |     6.7 |
| + DCAC, WCA, EDF                          |     92.06 |     88.00 |         95.40 |           2.89 |     9.2 |
| + DCAC, WCA, EDF, RPD (AEFFNet-N teacher) |     92.58 |     89.25 |         95.32 |           2.72 |     8.4 |
| **AEFFNet-PD (final student)**            | **92.51** | **89.03** |     **95.27** |       **1.28** | **5.2** |

Adding RPD raises recall from 88.00% to 89.25% and reduces compute from 9.2 to 8.4 GFLOPs, while `mAP@0.5` changes from 95.40% to 95.32%. Pruning and distillation then reduce the teacher’s parameter count by **52.94%** and its FLOPs by **38.10%**, with a **0.05 percentage-point decrease** in `mAP@0.5`. The final student exceeds YOLOv11-N by **0.69 percentage points** in `mAP@0.5` and reaches **90.74% F1** on TLD.

## Reproducing the comparisons

- **Main TLD experiment (Tables 2 and 4).** Convert the COCO annotations for the study’s 26,043 images to YOLO format and use the paper’s 80/20 training/validation split. Train the baseline and uncompressed teacher from scratch. Validate every configuration on the same image list. [TLD source dataset](https://huggingface.co/datasets/ChaiJohn/TLD)
- **Illumination experiment (Table 3).** Divide TLD into normal illumination (12,456 images), low illumination (8,327), and nighttime (5,260). For each detector, train a separate model within each condition and evaluate on that condition’s validation partition. Each detector therefore has three sets of weights for this comparison.
- **VD-NUS experiment (Table 5).** Train and validate on a separate 80/20 split of 6,000 VD-NUS images annotated by the authors for brake-light detection. The reported AEFFNet row reaches 94.62% `mAP@0.5` and 90.07% F1 on this experiment. [VD-NUS source dataset](https://github.com/Txy21012/VD-NUS)

### Running compression and validation

Use the project’s custom Ultralytics environment with the AEFFNet modules, `DetectionCompressor`, `DetectionFinetune`, and `DetectionDistiller`. The paper reports Python 3.10, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, and an NVIDIA GeForce RTX 4090. Pruning and distillation use 640×640 inputs. Supervised fine-tuning and distillation use SGD.

1. In the pruning driver, set `model` to the trained AEFFNet-N checkpoint, `data` to the TLD dataset YAML, `sl_hyp` to the pruning hyperparameter YAML, and `project` to a local output directory. Run the driver from the project root (`python "compress(1).py"` for the supplied copy). Keep the raw `prune.pt` produced by the compressor.
2. In `distill.py`, set `model` to that `prune.pt`, `teacher_weights` to the original AEFFNet-N checkpoint, and `teacher_cfg` to its matching model YAML. Reuse the same `data` setting. Run `python distill.py` with `kd_loss_type='logical'`, `logical_loss_type='BCKD'`, and `logical_loss_ratio=1.0`.
3. To evaluate the distilled checkpoint, run the following Python snippet in the same environment and enter the paths for your run.

```python
from ultralytics import YOLO

student_weights = input("Path to the distilled student checkpoint: ").strip()
dataset_yaml = input("Path to the TLD dataset YAML: ").strip()
metrics = YOLO(student_weights).val(data=dataset_yaml, split="val", imgsz=640)
print(f"TLD validation mAP@0.5 = {metrics.box.map50:.2%}")
```

## Method references

[LAMP](https://arxiv.org/abs/2010.07611) · [BCKD](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Bridging_Cross-task_Protocol_Inconsistency_for_Distillation_in_Dense_Object_Detection_ICCV_2023_paper.html) · [Converse2D](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_Reverse_Convolution_and_Its_Applications_to_Image_Restoration_ICCV_2025_paper.pdf)
