# medieegnet
This repository implements a robust, multi-stage self-supervised learning framework designed for multi-channel time-series physiological signals (e.g., EEG). Addressing the common challenges of limited labeled data and inherent label noise, the architecture integrates masked channel modeling and contrastive learning within a dual-encoder paradigm.

Architecture Documentation: Multi-Channel Time-Series Self-Supervised Framework
This document details the architecture and functionality of the self-supervised learning framework defined across modeling_medieeg_helper.py and modeling_medieeg.py. This framework is designed for multi-channel time-series signals (e.g., EEG) and combines Masked Channel Modeling with Contrastive Learning to learn robust, generalizable representations, followed by a regularized fine-tuning paradigm for downstream tasks.
1. Foundational Components (modeling_medieeg_helper.py)
This module contains the building blocks of the neural network, covering feature embedding, temporal-spatial feature extraction, attention mechanisms, and the decoder structures.
1.1 Feature Embedding
SpectralEmbedding: Projects the multi-band features of the input signal (e.g., 5 frequency bands) into a higher-dimensional latent space via a linear transformation. This amplifies the discriminative patterns across different frequency bands.

1.2 Temporal Feature Extraction
unit_tcn / TCN: Temporal Convolutional Network units. They utilize 1D causal convolutions to capture temporal dependencies. A custom delect_padding (Chomp) operation is applied to trim future padding, ensuring strict causality and preventing information leakage from future time steps.
TemporalBlock: Multi-Scale Temporal Fusion Block. It consists of three parallel temporal convolution pathways with varying kernel sizes (e.g., 2, 5, 7) to capture short-, medium-, and long-term temporal dependencies. The multi-scale outputs are fused via element-wise summation.

1.3 Attention Mechanisms
Attention: Standard Multi-Head Self-Attention mechanism. Used to model spatial dependencies and global interactions across different channels.
CrossAttention: Cross-Attention mechanism. Used in the decoding phase, where visible channel features act as Key/Value to query and predict the latent representations of masked channels.

1.4 Core Building Blocks
Block: Standard Transformer block consisting of Attention, LayerNorm, MLP, and DropPath, supporting residual connections and layer scaling.
RegressorBlock: Cross-Attention Regression Block. Replaces self-attention with CrossAttention. It is specifically designed to regress unknown masked features based on known contextual features.

1.5 Core Network Modules
VisionTransformerEncoder: The core backbone encoder.
Function: Extracts spatio-temporal joint representations from input signals.
Pipeline: Input data sequentially passes through Spectral Embedding -> Multi-Scale Temporal Fusion -> Temporal Enhancement (TCN) -> Addition of Sinusoidal/Cosine Spatial Positional Encoding -> Stacked Blocks (Spatial Self-Attention).
Features: Accepts a boolean mask matrix to process only the visible channels and can return intermediate layer features for subsequent contrastive learning.
VisionTransformerNeck: Masked Reconstruction Decoder.
Function: Solves the pretext task of reconstructing masked channels.
Pipeline: Takes mask tokens and visible channel features from the encoder, passes through stacked RegressorBlocks (using cross-attention to predict masked latent representations), then through stacked Blocks (self-attention for refinement), and finally maps back to the original feature dimension via a linear layer.
FeaturePredictor: Feature Projection Head.
Function: Maps the encoder's output representations into a specific metric space for contrastive learning.
Pipeline: Consists of stacked Blocks and a linear layer. The output is passed through ReLU activation and L2 normalization for computing contrastive similarity.

2. Integrated Framework Model (modeling_medieeg.py)
This module defines the main model class REmoNet (representing a generic dual-pathway self-supervised framework). It integrates the foundational components and implements the complete forward propagation logic for both pre-training and fine-tuning modes.
2.1 Architecture Initialization
The model initializes the following core components:
self.encoder: The primary encoder (serving as the base network for representation learning).
self.teacher: Momentum encoder. Its parameters are initially identical to encoder but are frozen (no gradients). It is updated via Exponential Moving Average (EMA) to ensure representation stability during contrastive learning.
self.pretext_neck: The masked reconstruction decoder head.
self.feature_predictor: The feature projection head for contrastive learning.
self.linear: Classification head for downstream tasks during fine-tuning.
self.logit_scale: A learnable temperature parameter to scale the cosine similarity logits in contrastive loss computation.

2.2 Core Methods
momentum_update: Updates the teacher parameters using EMA based on the encoder parameters, ensuring smooth and stable target generation for contrastive learning.

2.3 Forward Logic (forward)
The model flexibly controls data flow for different training phases using finetune and is_teacher flags:
Mode A: Teacher Forward (is_teacher=True)
Extracts features using the teacher and outputs them via the feature_predictor. This is used to generate target features for contrastive learning without computing gradients.
Mode B: Fine-Tuning Mode (finetune=True)
Extracts features using the encoder.
Passes the CLS token through self.linear for downstream classification, outputting classification logits.
Passes features through feature_predictor to output normalized embeddings. If samples are provided, it computes contrastive similarity logits for regularization.
Mode C: Pre-Training Mode (Default)
This mode executes a complex joint optimization logic:
Encode Visible Parts: The encoder processes unmasked channels, outputting visible channel features (x_unmasked).
Alignment Target Generation: The frozen teacher processes the *masked* channels to generate latent representation targets (latent_target), used for auxiliary alignment loss.
Momentum Update: Calls momentum_update to update the teacher parameters.
Masked Reconstruction: Concatenates mask tokens with visible features and feeds them into pretext_neck, outputting reconstructed original features (logits) and predicted latent representations (latent_pred).
Contrastive Feature Extraction: Feeds the reconstructed features into feature_predictor to output normalized embeddings for contrastive learning.
Return Values: Returns reconstruction logits, latent predictions, latent targets, and contrastive logits. These are used by external loss functions to compute the total pre-training loss (typically Reconstruction Loss + Latent Alignment Loss + Contrastive Loss).
