"""
ZipEnhancer Module - Audio Denoising Enhancer

Provides on-demand import ZipEnhancer functionality for audio denoising processing.
Related dependencies are imported only when denoising functionality is needed.
"""

import os
import tempfile
from typing import Optional, Union
import torchaudio
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def load_audio(file_path: str):
    """
    Load audio file with fallback to soundfile if torchcodec is not available.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Tuple of (audio_tensor, sample_rate) where audio_tensor has shape [channels, samples]
    """
    try:
        # Try torchaudio first (requires torchcodec in newer versions)
        audio, sr = torchaudio.load(file_path)
        return audio, sr
    except (ImportError, RuntimeError) as e:
        # Fallback to soundfile if torchcodec is missing
        error_msg = str(e).lower()
        if "torchcodec" in error_msg or "torch codec" in error_msg:
            try:
                import soundfile as sf
                data, sr = sf.read(file_path, dtype='float32')
                # Convert to torch tensor and ensure correct shape [channels, samples]
                if len(data.shape) == 1:
                    # Mono audio: add channel dimension
                    audio = torch.from_numpy(data).unsqueeze(0)
                else:
                    # Multi-channel: transpose from [samples, channels] to [channels, samples]
                    audio = torch.from_numpy(data).T
                return audio, sr
            except ImportError:
                raise ImportError(
                    "Neither torchcodec nor soundfile is available. "
                    "Please install torchcodec or ensure soundfile is installed."
                ) from e
        else:
            # Re-raise if it's not a torchcodec-related error
            raise


class ZipEnhancer:
    """ZipEnhancer Audio Denoising Enhancer"""
    def __init__(self, model_path: str = "iic/speech_zipenhancer_ans_multiloss_16k_base"):
        """
        Initialize ZipEnhancer
        Args:
            model_path: ModelScope model path or local path
        """
        self.model_path = model_path
        self._pipeline = pipeline(
                Tasks.acoustic_noise_suppression,
                model=self.model_path
            )
        
    def _normalize_loudness(self, wav_path: str):
        """
        Audio loudness normalization
        
        Args:
            wav_path: Audio file path
        """
        audio, sr = load_audio(wav_path)
        loudness = torchaudio.functional.loudness(audio, sr)
        normalized_audio = torchaudio.functional.gain(audio, -20-loudness)
        torchaudio.save(wav_path, normalized_audio, sr)
    
    def enhance(self, input_path: str, output_path: Optional[str] = None, 
                normalize_loudness: bool = True) -> str:
        """
        Audio denoising enhancement
        Args:
            input_path: Input audio file path
            output_path: Output audio file path (optional, creates temp file by default)
            normalize_loudness: Whether to perform loudness normalization
        Returns:
            str: Output audio file path
        Raises:
            RuntimeError: If pipeline is not initialized or processing fails
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input audio file does not exist: {input_path}")
        # Create temporary file if no output path is specified
        if output_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                output_path = tmp_file.name
        try:
            # Perform denoising processing
            self._pipeline(input_path, output_path=output_path)
            # Loudness normalization
            if normalize_loudness:
                self._normalize_loudness(output_path)
            return output_path
        except Exception as e:
            # Clean up possibly created temporary files
            if output_path and os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except OSError:
                    pass
            raise RuntimeError(f"Audio denoising processing failed: {e}")