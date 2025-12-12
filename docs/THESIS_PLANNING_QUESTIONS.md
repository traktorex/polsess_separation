# Master's Thesis Planning Questions

**Purpose**: Strategic thinking before committing to project architecture
**Date**: 2025-11-29
**Status**: In Progress

---

## Your Answers So Far

### Research Direction
**Focus Areas**:
- Comparison of model architectures and their efficiency in different scenarios
  - Deployment considerations (websites, mobile apps)
  - Parameter count vs. performance trade-offs
- End-to-end vs. modular networks comparison
  - Single multi-task model (enhancement + separation) vs. pipeline approach
  - Multi-task learning investigation
- Context: Noisy speech dataset with sound events AND reverberation
- **NOT focusing on**: Specific applications (hearing aids, voice assistants)
- **Focus on**: Technology and science
- **Interest**: Transfer learning across datasets, possibly foundational models

### Novelty & Contribution
- **No new model architectures** (too much for master's thesis)
- Adapt existing models to Polish language datasets (nature currently unknown)
- **Definite**: MM-IPC experiments
- **Tool Development**: Research-focused, not commercial product
  - But closer to "training app" than originally planned
  - Clarity needed on scope

### Scope Boundaries
- Number of experiments: Unknown yet
- **Ablation studies**: New to you, seems interesting
- **Cross-dataset evaluation**: Definitely yes
  - Train on X, test on Y
  - Use PESQ and other metrics beyond SI-SDR
- **Ensemble methods**: Interesting but probably too much

### Dataset Requirements
**Confirmed Datasets**:
1. PolSESS
2. LibriMix
3. One other (currently unspecified)

**Constraints**:
- **Sample rate**: 8kHz only (simplification)
- **Speakers**: 2 speakers only (simplification)

**Augmentation**:
- New dataset will probably need augmentation
- Should be implemented within dataset class file
- Should NOT affect rest of project (modularity)

---

## Your Key Concerns

### Architecture Philosophy
- Want modularity without bloat
- Code should demonstrate process and ideas
- **NOT** building a ready-to-ship commercial product
- Clean and readable FIRST
- Questioning whether trainer_factory is necessary
- Avoid convolution and bloat

### Current Uncertainties
- Exact number of experiments
- Nature of new Polish datasets (will clarify thesis scope)
- Thesis statement (will come later)
- Many answers will emerge during thesis work

---

## Complete Question Set

### 🎯 Thesis Scope & Research Questions

#### 1. Research Direction
- What general area of speech separation interests you most?
  - Model architecture comparisons? ✓
  - Dataset-specific challenges (low-resource languages, reverberant conditions)? ✓
  - Practical applications (hearing aids, voice assistants)? ✗
  - Transfer learning across datasets? ✓ (very interesting)
  - Efficiency vs. performance trade-offs? ✓

#### 2. Novelty & Contribution
- Are you planning to:
  - Propose a new model architecture? ✗
  - Adapt existing models to a new problem (e.g., Polish language specificity)? ✓
  - Conduct comprehensive benchmarking/comparison studies? ✓
  - Investigate a specific phenomenon (e.g., multi-modal corpus augmentation like MM-IPC)? ✓
  - Develop practical tools/frameworks? ~ (research tool, not commercial)

#### 3. Scope Boundaries
- How many experiments do you envision running? (10? 50? 100?) → Unknown
- Will you need to run ablation studies (varying single parameters)? → Interesting, new to you
- Do you plan to do cross-dataset evaluation (train on A, test on B)? → ✓ Definitely
- Will you need ensemble methods or model combinations? → Interesting but probably too much

---

### 📊 Dataset Requirements

#### 4. Dataset Diversity
- Beyond PolSESS, which datasets are you considering?
  - WSJ0-2Mix/3Mix (English, standard benchmark)? → No
  - LibriMix (modern, larger scale)? → ✓
  - WHAM!/WHAMR! (with noise/reverb)? → No
  - DNS Challenge (Microsoft, diverse noise)? → No
  - Language-specific datasets (Polish, others)? → ✓ (one more, unspecified)
  - Custom datasets you'll create? → Unknown

#### 5. Dataset Characteristics ✓ ANSWERED
- What variations do you need to handle?
  - Sample rates: 8kHz only ✓
  - Number of speakers: 2 only ✓
  - Background conditions: clean, noisy, reverberant → noisy + reverberant ✓
  - Recording conditions: indoor, outdoor, multi-modal → Unknown
  - Languages: monolingual vs. multilingual → Polish focus

#### 6. Data Loading & Preprocessing ✓ ANSWERED
- Will you need:
  - On-the-fly augmentation (pitch shift, speed change, noise injection)? → Probably yes
  - Different mixture types (anechoic, reverberant, dynamic mixing)? → Reverberant yes
  - Variant selection (like your current MM-IPC system)? → ✓ Definitely
  - Different task modes beyond ES/EB/SB? → Unknown
  - Memory-efficient loading for large datasets? → Probably yes
- **Key constraint**: Augmentation should be modular (in dataset class only)

---

### 🏗️ Model Architecture Requirements

#### 7. Model Selection Criteria
- What matters most for your experiments?
  - Maximum performance regardless of cost? → No
  - Efficiency (real-time capable, low-resource)? → ✓ (deployment consideration)
  - Interpretability (understanding what the model learns)? → Unknown
  - Diversity (comparing different architectural families)? → ✓
  - Specific properties (causal, streaming, online learning)? → Unknown

#### 8. Model Configurations
- Will you need:
  - Different model sizes (small/medium/large) for the same architecture? → Possibly
  - Pretrained models vs. training from scratch? → Unknown
  - Fine-tuning capabilities? → Unknown
  - Multi-task models (e.g., separation + enhancement)? → ✓ Very interesting
  - Curriculum learning or progressive training? → Unknown

#### 9. Custom Modifications
- Might you need to:
  - Modify existing architectures (add/remove components)? → Probably not
  - Experiment with different loss functions? → Unknown
  - Try different optimization strategies? → Unknown
  - Implement custom attention mechanisms or modules? → Probably not

---

### 🧪 Experimental Design

#### 10. Training Configurations
- How much flexibility do you need in:
  - Hyperparameter tuning (learning rates, batch sizes, schedulers)? → Unknown
  - Regularization techniques (dropout, weight decay, data augmentation)? → Unknown
  - Mixed precision training (AMP) vs. full precision? → Already using AMP
  - Gradient accumulation strategies? → Already using
  - Early stopping criteria? → Unknown

#### 11. Experiment Tracking
- What information do you need to log?
  - Just final metrics, or full training curves? → Probably both
  - Audio samples at checkpoints for listening tests? → Unknown
  - Model predictions on specific test cases? → Unknown
  - Computational resources (GPU memory, time, energy)? → Probably yes (efficiency focus)
  - Intermediate representations (attention maps, embeddings)? → Unknown

#### 12. Reproducibility
- How important is exact reproducibility?
  - Same results with same seed (strict)? → Unknown
  - Similar trends (statistical reproducibility)? → Unknown
  - Do you need to share code/models publicly? → Unknown
  - Will you publish in venues requiring code release? → Unknown

---

### 📈 Evaluation & Metrics

#### 13. Evaluation Metrics
- Beyond SI-SDR, what metrics matter for your thesis?
  - Perceptual metrics (PESQ, STOI)? → ✓ PESQ definitely, others unknown
  - Word Error Rate (if using ASR downstream)? → Unknown
  - Computational metrics (latency, throughput, memory)? → ✓ (efficiency focus)
  - Subjective listening tests (MOS scores)? → Unknown
  - Task-specific metrics (e.g., intelligibility for hearing aids)? → No (not application-focused)

#### 14. Evaluation Scenarios ✓ PARTIALLY ANSWERED
- What test conditions do you need?
  - Matched (train and test on same dataset)? → ✓
  - Cross-dataset (generalization)? → ✓ Definitely
  - Cross-language? → Unknown
  - Different SNR levels? → Unknown
  - Different number of speakers than training? → No (2 speakers only)
  - Unseen background types? → Probably yes

#### 15. Statistical Analysis
- Will you need:
  - Multiple runs with different seeds for statistical significance? → Unknown
  - Confidence intervals or error bars? → Unknown
  - Hypothesis testing (t-tests, ANOVA)? → Unknown
  - Correlation analysis between metrics? → Unknown

---

### 🔄 Workflow & Usability

#### 16. User Interface
- How will you interact with the system?
  - Command-line only (current approach)? → Probably yes
  - Configuration files (YAML) exclusively? → Probably yes
  - Interactive notebooks for exploration? → Unknown
  - Web interface for listening tests? → Probably not
  - Automated pipelines for batch experiments? → Unknown

#### 17. Experiment Management
- Do you need:
  - Automated hyperparameter search (grid, random, Bayesian)? → Unknown
  - Experiment queuing system (run 10 experiments overnight)? → Unknown
  - Resume from interruption (cluster preemption)? → Unknown
  - Comparison dashboards (side-by-side results)? → Unknown
  - Version control for experiments (like DVC, MLflow)? → Unknown

#### 18. Output & Artifacts
- What do you need to save?
  - Only best checkpoint, or all checkpoints? → Unknown
  - Separated audio files for qualitative analysis? → Probably yes
  - Visualizations (spectrograms, attention maps)? → Unknown
  - Summary tables (LaTeX format for thesis)? → Probably yes
  - Intermediate results for debugging? → Unknown

---

### ⚙️ Technical Constraints

#### 19. Computational Resources
- What hardware do you have access to?
  - Single GPU (what model?)? → Unknown
  - Multiple GPUs (how many, distributed training)? → Unknown
  - CPU-only fallback needed? → Unknown
  - Cloud resources (AWS, Google Cloud)? → Unknown
  - Time limits (cluster wall-time)? → Unknown

#### 20. Dataset Storage
- Where will data live?
  - Local SSD (fast, limited space)? → Current: local drives
  - Network storage (slower, more space)? → Unknown
  - Cloud storage (S3, GCS)? → Unknown
  - Need for lazy loading or caching strategies? → Unknown

#### 21. Dependencies & Compatibility
- What matters for your environment?
  - Python version constraints? → Currently Python 3.13
  - PyTorch version (1.x vs 2.x)? → Unknown
  - CUDA version requirements? → Unknown
  - Compatibility with university cluster? → Unknown
  - Avoiding dependency conflicts between toolkits? → Probably important

---

### 📅 Timeline & Priorities

#### 22. Timeline
- When is your thesis due? → Unknown
- How much time can you dedicate to:
  - Implementation/refactoring? → Unknown
  - Running experiments? → Unknown
  - Writing? → Unknown
- What are the hard deadlines (proposal, defense, submission)? → Unknown

#### 23. Milestones
- What do you need working by when?
  - Multi-dataset support for preliminary experiments? → Unknown
  - All model architectures for main experiments? → Unknown
  - Final results for thesis writing? → Unknown
- Which features are must-haves vs. nice-to-haves? → Unknown

#### 24. Risk Management
- What could go wrong?
  - Models don't converge on new datasets? → Unknown
  - Not enough time for all planned experiments? → Unknown
  - Hardware failures or access issues? → Unknown
  - Unexpected poor results requiring pivot? → Unknown
- What are your backup plans? → Unknown

---

### 🎓 Academic Requirements

#### 25. Thesis Committee Expectations
- What does your committee value?
  - Novel contributions vs. thorough empirical work? → Unknown
  - Theoretical depth vs. practical results? → Unknown
  - Publications required before defense? → Unknown
  - Software artifacts as contributions? → Unknown

#### 26. Literature & Positioning
- How will your work fit in the literature?
  - Are you comparing to specific baselines (which ones)? → Unknown
  - Following a specific research thread? → Unknown
  - Addressing gaps identified by others? → Unknown
  - Replicating/extending prior work? → Unknown

#### 27. Publication Plans
- Do you plan to publish during your master's?
  - Conference papers (which venues, deadlines)? → Unknown
  - Workshop papers? → Unknown
  - Journal articles? → Unknown
  - Open-source software releases? → Unknown

---

### 🔍 Edge Cases & Future-Proofing

#### 28. Extensibility
- Beyond datasets and models, might you need:
  - Different loss functions as modules? → Possibly
  - Custom data augmentation strategies? → ✓ Yes (in dataset class)
  - Different input representations (waveform, spectrogram, features)? → Unknown
  - Online learning or continuous adaptation? → Unknown
  - Multi-stage pipelines (preprocessing + separation + postprocessing)? → ✓ (end-to-end vs modular)

#### 29. Collaboration
- Will others use your code?
  - Lab mates for their projects? → Unknown
  - Advisor for demos/papers? → Unknown
  - Future students building on your work? → Unknown
  - Public release with documentation? → Unknown

#### 30. Backwards Compatibility
- How important is it to:
  - Keep existing checkpoints loadable? → Unknown
  - Maintain existing configuration files? → Unknown
  - Support old experiment scripts? → Unknown
  - Or: fresh start acceptable? → Probably acceptable

---

### 🤔 Meta Questions

#### 31. Learning Goals
- What do you want to learn from this project?
  - Deep understanding of separation architectures? → Probably yes
  - Software engineering best practices? → Probably yes
  - Experimental design skills? → Probably yes
  - Specific techniques (transformers, state-space models)? → Unknown

#### 32. Unknowns
- What don't you know yet that matters?
  - Which specific phenomena in PolSESS data to investigate? → ✓ Will clarify with new datasets
  - Whether Polish language has unique challenges? → Unknown
  - How well existing models transfer to your data? → Unknown
  - What your advisor expects? → Unknown

---

## Design Philosophy Summary

### What You Want
1. **Modularity without bloat**
   - Core stays stable
   - Datasets and models as plugins
   - Easy to add new ones without touching core code

2. **Clean and readable code FIRST**
   - Not a commercial product
   - Demonstrates process and ideas
   - Educational value for understanding

3. **Research-focused, not production-focused**
   - Don't over-engineer
   - Don't optimize prematurely
   - Keep it simple

4. **Key constraint**: 8kHz, 2 speakers (simplifies everything)

### What You're Questioning
- Is `trainer_factory` necessary or is it over-engineering?
- Are we building the right abstractions?
- Will this approach work for your thesis needs?

---

## Next Steps

1. Validate architectural approach
2. Design simple, clean plugin system
3. Identify minimal set of abstractions needed
4. Ensure it supports your research goals without bloat

---

**Status**: Needs architectural proposal
**Key Decision**: Simplicity vs. Flexibility balance
