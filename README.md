Brain Tumor Segmentation using U-Net++ with Transformer-based Encoder

This project performs multi-class brain tumor segmentation on the BraTS2020 dataset using a hybrid deep learning architecture that combines the U-Net++ decoder design with a Transformer-based encoder, enabling the model to capture both fine-grained local features and long-range global context from multi-modal MRI scans.

Overview

Brain tumor segmentation from MRI is a critical task in clinical diagnosis and treatment planning. Traditional CNN-based encoders like ResNet34 are strong at capturing local spatial patterns but struggle with long-range dependencies across the image. This project addresses that limitation by combining:

U-Net++: A nested, densely-connected decoder architecture that improves gradient flow and captures fine-grained details through multiple skip pathways.
Transformer Encoder: Self-attention layers that model global contextual relationships across the entire image, helping the network better distinguish tumor sub-regions that traditional convolutions may miss.

This hybrid design aims to combine the strengths of both approaches — local precision from convolutional/nested skip connections, and global context from self-attention — for more accurate tumor boundary delineation.

Dataset
Dataset: BraTS2020 (Brain Tumor Segmentation Challenge 2020)
Input Modalities (4 channels): FLAIR, T1, T1CE, T2
Output Classes (4 classes):
Background
Necrotic/Non-enhancing tumor core
Peritumoral edema
Enhancing tumor
