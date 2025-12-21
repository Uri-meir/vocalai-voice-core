import audioop

def mulaw_to_pcm(mulaw_data: bytes) -> bytes:
    """Decodes G.711 mu-law (from Twilio) to PCM16."""
    return audioop.ulaw2lin(mulaw_data, 2)

def pcm_to_mulaw(pcm_data: bytes) -> bytes:
    """Encodes PCM16 to G.711 mu-law (for Twilio)."""
    return audioop.lin2ulaw(pcm_data, 2)

import soxr
import numpy as np

def resample_audio(audio_data: bytes, in_rate: int, out_rate: int) -> bytes:
    """
    Resamples audio using soxr (High Quality).
    audio_data: PCM16 bytes
    """
    if in_rate == out_rate:
        return audio_data
        
    # Convert bytes to numpy array (int16)
    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    
    # Resample
    resampled_np = soxr.resample(audio_np, in_rate, out_rate)
    
    # Convert back to int16 bytes
    # Clip to ensure no overflow artifacts
    resampled_int16 = np.clip(resampled_np, -32768, 32767).astype(np.int16)
    
    return resampled_int16.tobytes()
