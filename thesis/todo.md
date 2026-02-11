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



SepALM: Audio Language Models Are Error Correctors for Robust Speech Separation
https://www.semanticscholar.org/paper/SepALM%3A-Audio-Language-Models-Are-Error-Correctors-Mu-Yang/9248414ca6036956d41f73c52aeed10ec79d51a3
!!!