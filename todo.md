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


