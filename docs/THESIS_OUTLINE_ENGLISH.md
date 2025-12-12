# Master's Thesis Outline: Neural Network-Based Speech Separation

**Working Title Suggestions:**
- *Comparative Analysis of Neural Network Architectures for Speech Separation in Polish Language Datasets*
- *Time-Domain Neural Approaches to Multi-Speaker Separation: Architecture Comparison and Evaluation*
- *End-to-End Speech Separation: Architectural Trade-offs and Cross-Dataset Evaluation*

---

## Chapter 1: Introduction

### 1.1 Motivation and Background
- The "Cocktail Party Problem" as a fundamental challenge in auditory perception
- Real-world applications and significance:
  - Hearing aids and assistive devices
  - Conference call systems and telecommunication
  - Automatic Speech Recognition (ASR) preprocessing
  - Audio production and post-processing
- Growing importance in human-computer interaction

### 1.2 The Polish Language Context
- Limited availability of large-scale Polish speech datasets
- Unique linguistic characteristics relevant to separation:
  - Phonetic properties of Polish
  - Prosodic patterns
  - Acoustic differences from well-studied languages (English, Mandarin)
- Motivation for research on Polish speech separation

### 1.3 Research Context
- Gap in current literature regarding Polish speech separation
- Need for comparative evaluation of modern architectures
- Trade-offs between model complexity and performance
- Transfer learning potential across languages and datasets
- *[Placeholder: Specific research questions will be added based on final thesis]*

### 1.4 Thesis Objectives
- Primary objective: *[To be defined based on research thesis]*
- Secondary objectives:
  - Comprehensive theoretical analysis of neural separation architectures
  - Implementation of modular experimental framework
  - Cross-dataset evaluation methodology
  - Performance benchmarking on Polish datasets
- *[Placeholder: Hypothesis statement]*

### 1.5 Thesis Structure Overview
- Brief description of each chapter's content and purpose
- Reading guide and chapter dependencies

---

## Chapter 2: Problem Description and Background

### 2.1 Historical Overview of Source Separation

#### 2.1.1 Classical Approaches
- Blind Source Separation (BSS) and Independent Component Analysis (ICA)
- Computational Auditory Scene Analysis (CASA)
- Time-frequency masking methods
- Limitations of classical approaches:
  - Assumptions about signal mixing (linear, instantaneous)
  - Requirement for equal or more mixtures than sources
  - Failure in reverberant conditions

#### 2.1.2 Machine Learning Era
- Transition from handcrafted features to learned representations
- Early neural network approaches (MLPs, RNNs on spectrograms)
- Deep learning revolution in speech processing
- Current state-of-the-art: end-to-end time-domain methods

### 2.2 Acoustic Foundations of Speech

#### 2.2.1 Digital Signal Representation
- Time-domain representation (waveform)
- Frequency-domain representation (Fourier Transform)
- Time-frequency representations:
  - Short-Time Fourier Transform (STFT)
  - Spectrograms (magnitude and phase)
  - Mel-frequency representations

#### 2.2.2 Speech Production and Characteristics
- Source-filter model of speech production
- Harmonic structure and pitch
- Formants and vocal tract resonances
- Voiced vs. unvoiced speech
- Temporal dynamics and prosody

#### 2.2.3 Acoustic Properties of Multi-Speaker Scenarios
- Overlap patterns in conversational speech
- Speaker-specific characteristics:
  - Fundamental frequency (F0) differences
  - Spectral envelope differences
  - Speaking rate and rhythm
- Environmental effects:
  - Room acoustics and reverberation
  - Background noise and interference
  - The importance of spatial cues (in multi-channel scenarios)

### 2.3 Mathematical Problem Formulation

#### 2.3.1 Signal Mixing Models
- Linear instantaneous mixing: $\mathbf{x}(t) = \mathbf{A}\mathbf{s}(t)$
- Convolutive mixing (reverberant environments): $x(t) = \sum_{i=1}^{N} h_i(t) * s_i(t) + n(t)$
- Single-channel mixture model: $x(t) = \sum_{i=1}^{N} s_i(t) + n(t)$
- Additive noise component $n(t)$

#### 2.3.2 Separation Objectives
- Goal: Estimate source signals $\hat{s}_i(t)$ from mixture $x(t)$
- Challenges:
  - Underdetermined problem (single mixture, multiple sources)
  - Permutation ambiguity
  - Scale ambiguity
  - Phase reconstruction problem (for T-F methods)

#### 2.3.3 Monaural (Single-Channel) Separation
- Why monaural separation is harder than multi-channel
- Loss of spatial information
- Reliance on learned source characteristics
- Relevance to real-world applications

### 2.4 Current State of Research

#### 2.4.1 Dominant Paradigms
- Time-frequency masking vs. time-domain separation
- Discriminative vs. generative approaches
- Single-stage vs. multi-stage pipelines
- Task-specific vs. universal separation models

#### 2.4.2 Key Research Directions
- Model efficiency and computational cost
- Generalization across domains and languages
- Real-time processing capabilities
- Multi-task learning (separation + enhancement + recognition)
- Few-shot and zero-shot separation

#### 2.4.3 Open Challenges
- Long-context modeling (minutes of audio)
- Handling unknown number of speakers
- Robustness to diverse acoustic conditions
- Low-resource language adaptation
- Evaluation methodology limitations

---

## Chapter 3: Theoretical Foundations

### 3.1 Digital Signal Processing for Speech

#### 3.1.1 Fundamental Transforms
- Fourier Transform and frequency analysis
- Discrete Fourier Transform (DFT) and FFT algorithm
- Short-Time Fourier Transform (STFT):
  - Windowing functions (Hann, Hamming)
  - Time-frequency resolution trade-off
  - Inverse STFT and overlap-add reconstruction

#### 3.1.2 Time-Frequency Representations
- Magnitude and phase components
- Log-magnitude spectrograms
- Mel-scale and perceptually-motivated representations
- Phase importance in audio reconstruction

#### 3.1.3 Signal Quality Metrics
- Time-domain metrics: MSE, MAE
- Frequency-domain metrics: LSD (Log-Spectral Distance)
- Signal-to-Noise Ratio (SNR) and its variants
- Scale-Invariant SNR (SI-SNR/SI-SDR):
  - Mathematical formulation
  - Invariance properties
  - Why it's preferred for separation tasks

### 3.2 Machine Learning Foundations for Audio

#### 3.2.1 Supervised Learning Framework
- Training, validation, and test sets
- Loss functions and optimization
- Gradient descent and backpropagation
- Regularization techniques (dropout, weight decay)
- Batch normalization and layer normalization

#### 3.2.2 Neural Network Building Blocks
- Fully connected layers (MLPs)
- Convolutional layers:
  - 1D convolutions for temporal signals
  - Receptive field and context window
  - Dilated convolutions
  - Depthwise separable convolutions
- Recurrent layers:
  - Vanilla RNN limitations
  - LSTM (Long Short-Term Memory)
  - GRU (Gated Recurrent Unit)
  - Bidirectional RNNs
- Attention mechanisms:
  - Scaled dot-product attention
  - Multi-head attention
  - Self-attention and cross-attention

#### 3.2.3 Sequence Modeling Paradigms
- Encoder-decoder architectures
- Autoregressive vs. non-autoregressive models
- Teacher forcing and inference strategies
- Handling variable-length sequences

### 3.3 Speech Separation Methodologies

#### 3.3.1 Time-Frequency Masking Approaches
- Binary vs. soft masking
- Ideal Binary Mask (IBM)
- Ideal Ratio Mask (IRM)
- Complex Ideal Ratio Mask (cIRM)
- Phase-sensitive masking
- Limitations:
  - Phase reconstruction problem
  - Information loss in magnitude-only processing
  - Artifacts from STFT/iSTFT pipeline

#### 3.3.2 Time-Domain End-to-End Approaches
- Motivation: avoiding phase problems
- Learned filterbanks vs. fixed transforms
- Direct waveform-to-waveform mapping
- Encoder-Separator-Decoder paradigm
- Advantages:
  - No phase reconstruction needed
  - Learnable representations
  - Fewer artifacts
  - Better performance empirically

#### 3.3.3 Training Objectives and Loss Functions
- Mean Squared Error (MSE) on waveforms
- Scale-Invariant Signal-to-Noise Ratio (SI-SNR) loss:
  - Mathematical derivation
  - Connection to projection in signal space
- Permutation Invariant Training (PIT):
  - Label permutation problem
  - Utterance-level PIT (uPIT)
  - Combinatorial optimization during training
  - Why it's essential for speaker-independent separation
- Multi-objective losses (combining waveform and spectrogram)

### 3.4 Neural Architecture for Speech Separation

#### 3.4.1 Convolutional Time-Domain Models

**Conv-TasNet (Convolutional Time-domain Audio Separation Network)**
- Overall architecture: Encoder → Separation → Decoder
- **Encoder:**
  - Learned convolutional basis (vs. fixed STFT)
  - 1D convolutions with kernel size K and stride
  - Output: latent representation of dimension N
- **Temporal Convolutional Network (TCN) Separator:**
  - Stacked 1D dilated convolutional blocks
  - Exponentially growing dilation factors: $2^0, 2^1, 2^2, ...$
  - Depthwise separable convolutions for efficiency
  - Bottleneck architecture with hidden dimension B
  - Skip connections and residual paths
  - Global layer normalization
  - Receptive field analysis
- **Decoder (Mask-based Reconstruction):**
  - Linear layer to generate C masks (C = number of sources)
  - Element-wise multiplication with encoder output
  - Transposed convolution (learned basis functions) to reconstruct waveform
- **Mathematical formulation:**
  - Encoder: $w = \text{Conv1D}(x)$, shape: $(B, N, T)$
  - Mask estimation: $M = \text{Separator}(w)$, shape: $(B, C, N, T)$
  - Decoder: $\hat{s}_i = \text{Basis}^{-1}(M_i \odot w)$
- Parameter count and computational complexity
- Trade-offs: receptive field vs. model size

#### 3.4.2 Recurrent Neural Network Models

**DPRNN (Dual-Path Recurrent Neural Network)**
- Motivation: modeling long-range dependencies in audio
- **Architecture Overview:**
  - Input segmentation (chunking)
  - Dual processing paths:
    1. Intra-chunk RNN (within segments)
    2. Inter-chunk RNN (across segments)
  - Alternating processing strategy
- **Intra-Chunk Processing:**
  - Models local temporal patterns within each chunk
  - Bidirectional LSTM over time dimension
  - Chunk size K (e.g., 100 frames)
- **Inter-Chunk Processing:**
  - Models global dependencies across chunks
  - Bidirectional LSTM over chunk dimension
  - Captures long-term structure
- **Advantages:**
  - Efficient long-sequence modeling (linear growth vs. quadratic)
  - Parallelizable compared to full-sequence RNN
  - Good balance between local and global context
- Mathematical notation for dual-path processing
- Comparison with standard LSTM on full sequence

#### 3.4.3 Transformer-Based Models

**SepFormer (Separation Transformer)**
- Motivation: self-attention for speech separation
- **Multi-Head Self-Attention Mechanism:**
  - Query, Key, Value projections
  - Scaled dot-product attention: $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
  - Multi-head parallel attention
  - Positional encoding for sequence order
- **Dual-Path Transformer Architecture:**
  - Similar chunking strategy as DPRNN
  - Intra-chunk Transformer blocks
  - Inter-chunk Transformer blocks
  - Feed-forward networks (FFN) after each attention layer
  - Layer normalization and residual connections
- **Complete Architecture:**
  - Encoder: learned convolutional basis (like Conv-TasNet)
  - SepFormer blocks: alternating intra/inter transformers
  - Decoder: mask-based reconstruction
- **Computational Complexity:**
  - Self-attention: $O(T^2 \cdot d)$ for sequence length T
  - Dual-path reduces to: $O(K^2 \cdot d \cdot N_{chunks})$
  - Trade-off: performance vs. computational cost
- Parameter count typically larger than Conv-TasNet
- When Transformers excel vs. when they don't

#### 3.4.4 State-Space Models and Recent Advances

**SPMamba and Mamba-based Architectures**
- Motivation: linear complexity alternative to Transformers
- **State-Space Models (SSM) Basics:**
  - Continuous-time state equations: $\dot{h}(t) = Ah(t) + Bx(t)$, $y(t) = Ch(t)$
  - Discretization for neural networks
  - Structured matrices for efficiency
- **Mamba Architecture:**
  - Selective state-space layers
  - Data-dependent parameters (input-dependent $A$, $B$, $C$)
  - Hardware-efficient implementation on GPUs
  - $O(N)$ complexity in sequence length
- **SPMamba for Speech Separation:**
  - Dual-path structure with Mamba blocks
  - Frequency-domain processing (STFT → Mamba → iSTFT)
  - Comparison with Transformer-based approaches
  - Trade-offs: efficiency vs. expressiveness
- **Practical Considerations:**
  - Linux + CUDA requirement for `mamba-ssm` library
  - Training stability and hyperparameter sensitivity
- Position in the landscape of modern architectures

#### 3.4.5 Architectural Comparison Framework
- **Time-domain vs. Frequency-domain processing**
- **Local vs. Global modeling:**
  - Conv-TasNet: local receptive fields via dilated convolutions
  - DPRNN: explicit local (intra) and global (inter) modeling
  - SepFormer: global attention with chunking
  - SPMamba: global modeling with linear complexity
- **Computational efficiency:**
  - FLOPs, memory footprint
  - Real-time factor (RTF)
  - Scalability to long sequences
- **Parallelizability and hardware utilization**
- **Inductive biases:**
  - Convolutional: locality and translation equivariance
  - Recurrent: temporal sequentiality
  - Attention: permutation equivariance, global dependencies
  - SSM: continuous-time dynamics
- Table summarizing key properties of each architecture

### 3.5 Evaluation Methodology

#### 3.5.1 Objective Quality Metrics

**Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)**
- Mathematical definition: $\text{SI-SDR} = 10 \log_{10} \frac{\|\mathbf{s}_{\text{target}}\|^2}{\|\mathbf{e}_{\text{noise}}\|^2}$
- Where $\mathbf{s}_{\text{target}} = \frac{\langle \hat{\mathbf{s}}, \mathbf{s} \rangle}{\|\mathbf{s}\|^2} \mathbf{s}$
- Scale invariance property
- Why it's the primary metric in modern separation

**Signal-to-Distortion Ratio (SDR)**
- Source separation metrics from BSS-Eval
- Decomposition into target, interference, and artifacts
- Relationship to SI-SDR

**Perceptual Quality Metrics**
- **PESQ (Perceptual Evaluation of Speech Quality):**
  - ITU-T P.862 standard
  - Models human perception of speech quality
  - Range: -0.5 to 4.5, higher is better
  - Narrowband vs. wideband versions
  - Computational considerations
- **STOI (Short-Time Objective Intelligibility):**
  - Correlation with speech intelligibility
  - Range: 0 to 1, higher is better
  - Short-time segments and correlation analysis
  - Advantages for noisy and reverberant conditions

**Other Relevant Metrics**
- PESQ-WB (Wideband PESQ)
- Extended STOI (ESTOI)
- ViSQOL (Virtual Speech Quality Objective Listener)
- When to use which metric (task-dependent)

#### 3.5.2 Subjective Evaluation Methods
- Mean Opinion Score (MOS)
- AB preference tests
- MUSHRA (Multiple Stimuli with Hidden Reference and Anchor)
- Limitations of objective metrics
- When subjective evaluation is necessary

#### 3.5.3 Computational Efficiency Metrics
- Model parameters (trainable weights)
- Floating Point Operations (FLOPs/MACs)
- Real-Time Factor (RTF): $\text{RTF} = \frac{\text{processing time}}{\text{audio duration}}$
- Memory footprint (GPU/CPU RAM)
- Latency and throughput
- Energy consumption (for edge deployment)

### 3.6 Datasets for Speech Separation

#### 3.6.1 Standard Benchmarks (English)
- **WSJ0-2mix and WSJ0-3mix:**
  - Wall Street Journal corpus base
  - 2-speaker and 3-speaker mixtures
  - Clean, anechoic conditions
  - Standard train/validation/test splits
- **LibriMix:**
  - Based on LibriSpeech corpus
  - Multiple mixture types (Libri2Mix, Libri3Mix)
  - Noisy variants (with WHAM! noise)
  - Larger scale than WSJ0-mix
  - 8kHz and 16kHz versions
- **WHAM! and WHAMR!:**
  - Real-world noise backgrounds
  - Reverberant versions (WHAMR!)
  - Challenges beyond clean separation

#### 3.6.2 Polish Language Datasets
- **PolSESS (Polish Speech Separation in Everyday Soundscapes):**
  - Characteristics and motivation
  - Recording conditions (indoor/outdoor)
  - Speaker demographics
  - Multi-modal components:
    - Speech (sp1, sp2)
    - Environmental sounds/events
    - Room reverberation
    - Background scenes
  - Dataset structure (train/val/test splits)
  - Size and duration statistics
- Challenges specific to Polish datasets:
  - Limited scale compared to English datasets
  - Need for data augmentation
  - Transfer learning opportunities

#### 3.6.3 Data Augmentation Strategies
- **On-the-fly mixing:**
  - Dynamic source combination during training
  - SNR randomization
  - Temporal alignment variations
- **Multi-Modal Indoor-Outdoor Corpus (MM-IPC) Augmentation:**
  - Concept: varying background complexity
  - Indoor variants (with reverb):
    - SER: Speech + Event + Reverb
    - SR: Speech + Reverb
    - ER: Event + Reverb
    - R: Reverb only
  - Outdoor variants (no reverb):
    - SE: Speech + Event
    - S: Speech only
    - E: Event only
  - Random variant selection during training
  - Benefits: robustness to diverse acoustic conditions
  - Implementation via lazy loading
- Other augmentation techniques:
  - Room Impulse Response (RIR) simulation
  - Noise addition
  - Speed and pitch perturbation

#### 3.6.4 Cross-Dataset Evaluation and Transfer Learning
- **Motivation:**
  - Generalization beyond training distribution
  - Domain shift challenges
  - Low-resource scenario handling
- **Transfer learning strategies:**
  - Pre-training on large dataset (e.g., LibriMix)
  - Fine-tuning on target dataset (e.g., PolSESS)
  - Freezing vs. fine-tuning strategies
- **Domain adaptation techniques:**
  - Multi-task learning
  - Adversarial domain adaptation
  - Meta-learning for few-shot adaptation
- Evaluation protocols:
  - Train on A, test on A (matched)
  - Train on A, test on B (cross-dataset)
  - Train on A+B, test on C (multi-domain)

### 3.7 Advanced Training Techniques

#### 3.7.1 Curriculum Learning
- Motivation: easier-to-harder training progression
- Strategies for speech separation:
  - Starting with limited acoustic variants, gradually adding complexity
  - SNR-based curriculum (high SNR → low SNR)
  - Sequence length curriculum (short → long)
- Implementation considerations:
  - Epoch-based variant scheduling
  - Dynamic difficulty adjustment
  - Learning rate coordination

#### 3.7.2 Optimization and Training Stability
- Gradient clipping for RNN stability
- Learning rate schedules:
  - Warmup strategies
  - ReduceLROnPlateau
  - Cosine annealing
- Mixed precision training (AMP):
  - Float16 vs. Float32 trade-offs
  - Gradient scaling
  - Numerical stability considerations (epsilon values)
  - Expected speedup and memory savings

#### 3.7.3 Multi-Task and Multi-Objective Learning
- Joint separation and enhancement
- Auxiliary tasks (speaker identification, ASR)
- Loss weighting strategies
- Pareto optimality in multi-objective scenarios

---

## Chapter 4: Technological Implementation and Tools

### 4.1 Software Stack and Development Environment

#### 4.1.1 Deep Learning Framework
- **PyTorch:**
  - Version considerations (1.x vs. 2.x)
  - Dynamic computation graphs
  - Autograd and backpropagation
  - GPU acceleration with CUDA
  - torch.compile for performance (PyTorch 2.0+)
- Alternatives and comparisons (TensorFlow, JAX)

#### 4.1.2 Audio Processing Libraries
- **torchaudio:**
  - Audio I/O operations
  - Transform implementations (STFT, resampling)
  - Integration with PyTorch
- **librosa:** Feature extraction and analysis
- **soundfile / scipy.io.wavfile:** File format handling
- **pydub:** Audio manipulation utilities

#### 4.1.3 Speech-Specific Toolkits
- **SpeechBrain:**
  - Pre-built modules for speech tasks
  - Separation model implementations
  - Training recipes and utilities
  - Integration considerations and customization
- **ESPnet:** Alternative toolkit overview
- **Asteroid:** Separation-focused library

#### 4.1.4 Experiment Tracking and Management
- **Weights & Biases (wandb):**
  - Experiment logging and visualization
  - Hyperparameter tracking
  - Model versioning
  - Collaborative features
- Alternatives: MLflow, TensorBoard, Comet
- Version control for experiments (Git, DVC)

#### 4.1.5 Evaluation and Metrics Libraries
- **torchmetrics:** GPU-accelerated metric computation
- **mir_eval:** Standard implementations of separation metrics
- **pesq / pystoi:** Perceptual metrics
- Custom metric implementations

#### 4.1.6 Development Tools
- Python version and environment management (venv, conda)
- Code quality tools (pytest, coverage)
- Configuration management (YAML, dataclasses)
- Documentation (Sphinx, docstrings)

### 4.2 Hardware and Computational Resources

#### 4.2.1 Training Infrastructure
- GPU requirements and recommendations:
  - VRAM needs for different batch sizes
  - CUDA compute capability
  - Multi-GPU considerations (if applicable)
- CPU requirements for data loading
- RAM and storage considerations
- *[Specific hardware used in this work: e.g., RTX 4070, 12GB VRAM]*

#### 4.2.2 Performance Optimization Techniques
- **Automatic Mixed Precision (AMP):**
  - Implementation with torch.cuda.amp
  - GradScaler for gradient scaling
  - Speedup measurements (30-40% typical)
  - Numerical stability patches (e.g., SpeechBrain EPS)
- **Model Compilation:**
  - torch.compile (PyTorch 2.x)
  - Platform considerations (Linux vs. Windows)
  - Expected speedup (10-20%)
- Data loading optimization:
  - Multi-process DataLoader workers
  - Pinned memory
  - Prefetching strategies
- Memory management:
  - Gradient checkpointing
  - Batch size tuning
  - Avoiding OOM errors

#### 4.2.3 Reproducibility Considerations
- Random seed setting (Python, NumPy, PyTorch, CUDA)
- Deterministic algorithms vs. performance trade-offs
- Environment specification (requirements.txt, environment.yml)
- Checkpoint saving and loading strategies

### 4.3 Software Architecture and Design Patterns

#### 4.3.1 Modular Design Philosophy
- Separation of concerns:
  - Data loading and preprocessing
  - Model definition
  - Training logic
  - Evaluation and inference
- Code readability and maintainability
- Research-focused vs. production-focused architectures
- Avoiding over-engineering: simplicity first

#### 4.3.2 Configuration Management System
- **Hierarchical Configuration:**
  - DataConfig, ModelConfig, TrainingConfig
  - Nested parameter classes (e.g., ConvTasNetParams, PolSESSParams)
- **Configuration Sources and Priority:**
  1. Default values (hardcoded in dataclasses)
  2. Environment variables
  3. YAML configuration files
  4. Command-line arguments (highest priority)
- YAML structure and CLI argument parsing
- Validation and error handling

#### 4.3.3 Registry Pattern for Extensibility
- **Model Registry:**
  - `@register_model("model_name")` decorator
  - `get_model("model_name")` retrieval
  - Automatic discovery of registered models
  - Easy addition of new architectures
- **Dataset Registry:**
  - `@register_dataset("dataset_name")` decorator
  - `get_dataset("dataset_name")` retrieval
  - Consistent interface (`allowed_variants` parameter)
  - Pluggable dataset implementations
- Benefits: modularity, extensibility, no core code changes needed

#### 4.3.4 Training Infrastructure
- **Trainer Class Design:**
  - Encapsulation of training loop
  - Epoch and batch iteration logic
  - Loss computation and backpropagation
  - Gradient accumulation handling (if used)
  - Validation and checkpointing
  - Integration with experiment tracking (wandb)
- **Checkpointing Strategy:**
  - Hierarchical directory structure: `checkpoints/{model}/{task}/{run_id}/`
  - Saving both model weights and configuration
  - Symlink to latest checkpoint for convenience
  - Handling torch.compile wrapped models (unwrapping for compatibility)
  - Best model selection based on validation metric

#### 4.3.5 Data Pipeline Architecture
- **Dataset Class Structure:**
  - PyTorch `Dataset` interface compliance
  - Lazy loading strategies for efficiency
  - MM-IPC variant selection logic
  - Metadata parsing (CSV files)
  - Audio file path resolution
- **Custom Collate Functions:**
  - Handling variable-length sequences
  - Padding and masking strategies
  - Batch formation
- **DataLoader Configuration:**
  - Batch size considerations
  - Number of workers for parallel loading
  - Shuffle strategies (train vs. val/test)
  - Drop last batch handling

### 4.4 Implementation Details

#### 4.4.1 Model Instantiation
- Translating theoretical architectures to code
- Hyperparameter initialization from config
- Weight initialization strategies
- Model size and parameter counting
- Device placement (CPU vs. GPU)

#### 4.4.2 Training Loop Implementation
- Epoch and batch loops
- Forward pass: `output = model(batch)`
- Loss computation with PIT (if applicable)
- Backward pass: `loss.backward()`
- Optimizer step and gradient clipping
- Learning rate scheduling
- Logging and progress tracking (progress bars, metrics)

#### 4.4.3 Evaluation and Inference Pipeline
- Model loading from checkpoint
- Evaluation mode: `model.eval()`, `torch.no_grad()`
- Batch processing for efficiency
- Metric computation (SI-SDR, PESQ, STOI)
- Per-variant evaluation (MM-IPC variants)
- Results aggregation and reporting
- Audio output generation for listening tests

#### 4.4.4 Debugging and Profiling Tools
- PyTorch debugging utilities (anomaly detection)
- Memory profiling (torch.cuda.memory_summary)
- Timing and bottleneck identification
- Gradient flow visualization
- Common issues and solutions:
  - NaN losses (AMP underflow, exploding gradients)
  - OOM errors (batch size, memory leaks)
  - Slow training (data loading bottleneck, inefficient operations)

---

## Chapter 5: Methodology

*This chapter contains both completed sections (evaluation protocols, experiment infrastructure) and placeholders for research-specific content that depends on the final thesis direction.*

### 5.1 Datasets and Experimental Setup

#### 5.1.1 Dataset Preparation
- **PolSESS Dataset:**
  - Directory structure and file organization
  - Metadata CSV parsing
  - Train/validation/test split statistics
  - Audio format specifications (sample rate, bit depth, duration)
  - MM-IPC variant distribution
- **LibriMix Dataset:**
  - Configuration (Libri2Mix, sample rate 8kHz)
  - Purpose: cross-dataset evaluation and transfer learning
  - Comparison with PolSESS characteristics
- **[Placeholder: Third Dataset]:**
  - *To be specified based on final research direction*
  - Rationale for inclusion
  - Specific characteristics

#### 5.1.2 Data Augmentation Protocol
- MM-IPC augmentation implementation:
  - Random variant selection during training
  - Lazy loading of audio components
  - Distribution of variants in training batches
- Curriculum learning schedule (if used):
  - Epoch-based variant progression
  - Learning rate adjustments
  - Rationale for chosen curriculum

#### 5.1.3 Experimental Environment
- Hardware specifications used
- Software versions (PyTorch, Python, CUDA)
- Training hyperparameters:
  - Batch size and gradient accumulation (if used)
  - Learning rate and scheduler
  - Number of epochs
  - Early stopping criteria
- Reproducibility measures:
  - Random seeds
  - Deterministic algorithm settings
  - Environment snapshots

### 5.2 Model Configurations

#### 5.2.1 Conv-TasNet Configuration
- Encoder parameters: N (filters), kernel size, stride
- TCN separator parameters:
  - B (bottleneck channels)
  - H (convolutional channels)
  - P (kernel size)
  - X (number of repeats)
  - R (number of blocks)
- Decoder parameters
- Total parameter count
- Receptive field calculation
- Configuration files (YAML) used

#### 5.2.2 DPRNN Configuration
- Chunk size and overlap
- RNN hidden dimensions
- Number of dual-path layers
- Intra-chunk vs. inter-chunk settings
- Total parameter count
- Configuration files (YAML) used

#### 5.2.3 SepFormer Configuration
- Encoder/decoder parameters
- Number of Transformer blocks
- Attention heads and hidden dimensions
- Feed-forward network dimensions
- Chunk size for dual-path
- Total parameter count
- Configuration files (YAML) used

#### 5.2.4 SPMamba Configuration (if included)
- Mamba block parameters
- State-space dimensions
- Frequency-domain processing details
- Total parameter count
- Platform requirements (Linux + CUDA)
- Configuration files (YAML) used

#### 5.2.5 Model Size Variants (if applicable)
- Small, medium, large configurations
- Parameter count scaling
- Trade-off analysis: size vs. performance

### 5.3 Training Protocol

#### 5.3.1 Loss Function and Optimization
- Primary loss: SI-SNR loss
- Permutation Invariant Training (PIT) implementation
- Optimizer: Adam (or specify alternative)
- Learning rate and schedule
- Gradient clipping threshold
- Weight decay / regularization

#### 5.3.2 Training Process
- Training duration (epochs, wall-clock time)
- Convergence behavior observation
- Validation frequency (every N epochs)
- Best model selection criterion
- Checkpoint saving strategy
- Experiment tracking with wandb

#### 5.3.3 Computational Budget
- GPU hours required per model
- Memory footprint during training
- Throughput (samples/sec or batches/sec)
- Cost considerations (if using cloud resources)

### 5.4 Evaluation Protocol

#### 5.4.1 Evaluation Metrics
- Primary: SI-SDR improvement (SI-SDRi)
- Secondary: PESQ, STOI
- Computational efficiency: RTF, parameter count, FLOPs

#### 5.4.2 Evaluation Scenarios
- **Matched Condition:**
  - Train on PolSESS, evaluate on PolSESS test set
  - Per-variant breakdown (SER, SR, ER, R, SE, S, E)
- **Cross-Dataset:**
  - Train on LibriMix, evaluate on PolSESS
  - Train on PolSESS, evaluate on LibriMix
  - Analysis of generalization
- **[Placeholder: Additional evaluation scenarios based on research thesis]**

#### 5.4.3 Statistical Analysis
- Mean and standard deviation across test set
- Per-speaker or per-utterance analysis (if applicable)
- Confidence intervals (if multiple runs performed)
- Statistical significance testing (if comparing approaches)

#### 5.4.4 Qualitative Analysis
- Listening tests (informal or formal)
- Spectrogram visualization of separated outputs
- Error analysis: failure cases and patterns
- Comparison of model behaviors

### 5.5 Research Experiments

*This section will contain the specific experiments based on the final research thesis. Placeholders indicate where research-specific content will be inserted.*

#### 5.5.1 [Placeholder: Primary Research Question]
- Hypothesis: *[To be defined]*
- Experimental design: *[To be defined]*
- Variables and controls: *[To be defined]*
- Expected outcomes: *[To be defined]*

#### 5.5.2 [Placeholder: Secondary Research Questions]
- *[Multiple subsections as needed]*

#### 5.5.3 [Placeholder: Ablation Studies]
- *[If applicable based on research thesis]*
- Example potential topics:
  - Effect of MM-IPC augmentation (with vs. without)
  - Curriculum learning effectiveness
  - Impact of model size scaling
  - Component-level analysis (encoder, separator, decoder)

#### 5.5.4 [Placeholder: Comparative Analysis]
- *[Architecture comparisons based on specific research focus]*
- Example potential topics:
  - Efficiency vs. performance trade-offs
  - Time-domain vs. frequency-domain processing
  - Convolutional vs. attention-based vs. RNN-based
  - Single-task vs. multi-task learning

#### 5.5.5 [Placeholder: Transfer Learning Experiments]
- *[If part of research thesis]*
- Pre-training strategies
- Fine-tuning protocols
- Cross-lingual or cross-domain transfer analysis

### 5.6 Experimental Challenges and Solutions

#### 5.6.1 Technical Challenges Encountered
- AMP numerical stability (EPS patching)
- Memory limitations and batch size tuning
- torch.compile compatibility issues (Windows vs. Linux)
- Checkpoint loading with compiled models
- [Other challenges encountered during implementation]

#### 5.6.2 Methodological Considerations
- Choice of validation set (small val set issue)
- Evaluation metric selection rationale
- Handling of edge cases in audio data
- Computational constraints and their impact

---

## Chapter 6: Summary and Conclusions

*This chapter will primarily be completed after experiments are done, but structure is outlined here.*

### 6.1 Summary of Theoretical Contributions
- Comprehensive review of neural separation architectures
- Comparative framework for understanding model trade-offs
- Analysis of time-domain end-to-end approaches
- Integration of classical signal processing with modern deep learning

### 6.2 Summary of Implementation Contributions
- Modular experimental framework design
- Registry pattern for extensibility
- MM-IPC augmentation methodology
- Cross-dataset evaluation protocols

### 6.3 [Placeholder: Research Findings]
- *[Summary of experimental results]*
- *[Answer to primary research question]*
- *[Answers to secondary research questions]*
- *[Key insights and discoveries]*

### 6.4 [Placeholder: Hypothesis Verification]
- *[Verification or refutation of stated hypothesis]*
- *[Discussion of unexpected results]*
- *[Limitations of findings]*

### 6.5 [Placeholder: Achievement of Objectives]
- *[Assessment of primary objective achievement]*
- *[Assessment of secondary objectives]*
- *[Degree of success and limitations]*

### 6.6 Practical Implications
- Applicability to Polish speech processing
- Trade-offs for real-world deployment
- Recommendations for architecture selection based on use case
- Insights for low-resource language scenarios

### 6.7 Limitations and Constraints
- Dataset size and diversity limitations
- Computational resource constraints
- Evaluation methodology limitations (objective metrics vs. subjective quality)
- Scope limitations (2 speakers, 8kHz, specific acoustic conditions)
- Generalization concerns

### 6.8 Future Research Directions
- **Model Architecture:**
  - Hybrid architectures combining strengths of different approaches
  - More efficient attention mechanisms
  - Continual learning and adaptation
- **Dataset and Evaluation:**
  - Larger Polish speech datasets
  - More diverse acoustic conditions
  - Subjective evaluation studies (MOS)
- **Applications:**
  - Real-time implementation
  - Integration with downstream tasks (ASR, speaker recognition)
  - Multi-channel extension (spatial information)
- **Transfer Learning:**
  - Few-shot adaptation techniques
  - Cross-lingual transfer mechanisms
  - Foundation models for speech separation
- **Multi-Task Learning:**
  - Joint separation and enhancement
  - End-to-end systems (separation + ASR)
- *[Other directions based on research findings]*

### 6.9 Closing Remarks
- Significance of the work
- Contributions to the field
- Personal reflections on the research process

---

## Appendices

### Appendix A: Configuration Files
- Complete YAML configurations used in experiments
- Model hyperparameter tables

### Appendix B: Additional Results
- Extended evaluation tables (all metrics, all variants)
- Statistical test results
- Additional visualizations (spectrograms, learning curves)

### Appendix C: Code Structure
- Overview of repository organization
- Key modules and their responsibilities
- Usage examples and documentation snippets

### Appendix D: Hardware and Software Specifications
- Complete list of dependencies with versions
- Hardware specifications
- Training time measurements

### Appendix E: [Placeholder: Additional Appendices as Needed]
- Audio samples (if thesis includes accompanying materials)
- Derivations of mathematical formulations
- Extended literature review tables

---

## List of Figures
*To be populated during writing*
- Block diagrams of model architectures
- Training and validation curves
- Evaluation metric comparisons
- Spectrogram examples
- [Other figures based on research]

## List of Tables
*To be populated during writing*
- Dataset statistics
- Model parameter counts
- Evaluation results across all conditions
- Computational efficiency metrics
- [Other tables based on research]

## Bibliography
*To be compiled during writing*
- Foundational papers (Conv-TasNet, DPRNN, SepFormer, etc.)
- Dataset papers (PolSESS, LibriMix, etc.)
- Evaluation metrics papers
- Related work in Polish speech processing
- [Other references based on research]

---

# Notes on This Outline

## Completed Sections (Can Be Written Now)
- **Chapter 1** (except 1.3 and 1.4 - research-specific)
- **Chapter 2** (entire background and problem description)
- **Chapter 3** (entire theoretical foundations)
- **Chapter 4** (entire technological implementation)
- **Chapter 5.1-5.4** (methodology infrastructure, evaluation protocol)
- **Chapter 6.1-6.2, 6.6-6.9** (non-research-specific parts)

## Placeholder Sections (Require Research Thesis)
- **1.3**: Specific research questions and hypothesis
- **1.4**: Exact thesis objectives
- **5.5**: All research experiments (design and execution)
- **6.3-6.5**: Research findings, hypothesis verification, achievement assessment

## Flexibility Notes
- Section numbering can be adjusted as needed
- Subsections can be expanded or contracted based on content depth
- Appendices can be added/removed based on final needs
- The outline is comprehensive enough to guide writing but flexible enough to accommodate changes

## Estimated Page Distribution (rough guide)
- Chapter 1: 5-8 pages
- Chapter 2: 12-18 pages
- Chapter 3: 25-35 pages (longest, most substantial)
- Chapter 4: 12-18 pages
- Chapter 5: 20-30 pages (depends on number of experiments)
- Chapter 6: 8-12 pages
- **Total: 82-121 pages** (fits 50-100 page target with flexibility)
