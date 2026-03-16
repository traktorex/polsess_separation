"""ASR transcription module using Whisper."""

import torch
import whisper
from pathlib import Path
from typing import Union, List, Optional
import numpy as np


class WhisperTranscriber:
    """Wrapper for Whisper ASR model."""
    
    def __init__(
        self,
        model_name: str = "base.en",
        device: Optional[str] = None,
        language: str = "en"
    ):
        """Initialize Whisper transcriber.
        
        Args:
            model_name: Whisper model size (tiny.en, base.en, small.en, medium.en, large)
            device: Device to run on (cuda/cpu), auto-detect if None
            language: Language code for transcription
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        self.model_name = model_name
        
        print(f"Loading Whisper model '{model_name}' on {self.device}...")
        self.model = whisper.load_model(model_name, device=self.device)
    
    def transcribe(
        self,
        audio: Union[np.ndarray, torch.Tensor, str, Path],
        normalize: bool = True
    ) -> str:
        """Transcribe audio to text.
        
        Args:
            audio: Audio as numpy array, torch tensor, or path to audio file
            normalize: Whether to normalize and uppercase the transcription
            
        Returns:
            Transcription text
        """
        # Convert torch tensor to numpy if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # Transcribe
        result = self.model.transcribe(
            audio,
            language=self.language,
            fp16=(self.device == "cuda")  # Use FP16 only on GPU
        )
        
        text = result["text"]
        
        # Normalize: strip whitespace and uppercase
        if normalize:
            text = text.strip().upper()
        
        return text
    
    def transcribe_batch(
        self,
        audio_list: List[Union[np.ndarray, torch.Tensor]],
        normalize: bool = True
    ) -> List[str]:
        """Transcribe a batch of audio samples.
        
        Args:
            audio_list: List of audio arrays/tensors
            normalize: Whether to normalize transcriptions
            
        Returns:
            List of transcriptions
        """
        return [self.transcribe(audio, normalize=normalize) for audio in audio_list]
