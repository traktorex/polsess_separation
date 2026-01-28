# Todo list

1. new architectures?
    - MossFormer2
    - DPTNet
    - TIGER
    - EDSep
    - coś mapującego
    - coś GAN, diffusion-based
    - Conv-TasNet GAN
    - wavesplit?
    
2. script for running models and saving enhanced/separated audio
    - we don't want to copy the original unseparated files
    - structure should be like this:
        - saved
            - sample_1
                - speaker_1.wav
                - speaker_2.wav
                - metadata containing:
                    - original_file_path
                    - model_name
                    - everything needed to know how the separation was done


3. Option 2: Add Graceful Shutdown (Cleaner)
Add signal handling to your trainer so it exits cleanly when Hyperband stops it:

python
import signal
# In trainer.py, add signal handler
def handle_early_termination(self):
    """Handle Hyperband early termination gracefully."""
    self.logger.warning("Received early termination signal from W&B Hyperband")
    if self.wandb_logger:
        self.wandb_logger.log_metrics({"hyperband_terminated": 1})
    raise SystemExit(0)  # Clean exit
