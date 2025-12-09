import audioop

def mulaw_to_pcm(mulaw_data: bytes) -> bytes:
    """Decodes G.711 mu-law (from Twilio) to PCM16."""
    return audioop.ulaw2lin(mulaw_data, 2)

def pcm_to_mulaw(pcm_data: bytes) -> bytes:
    """Encodes PCM16 to G.711 mu-law (for Twilio)."""
    return audioop.lin2ulaw(pcm_data, 2)

def resample_audio(audio_data: bytes, in_rate: int, out_rate: int) -> bytes:
    """
    Resamples audio using audioop.ratecv.
    audio_data: PCM16 bytes
    """
    if in_rate == out_rate:
        return audio_data
    
    # audioop.ratecv(fragment, width, nchannels, inrate, outrate, state[, weightA[, weightB]])
    # We use simple linear interpolation (state=None) for efficiency in this POC
    converted, _ = audioop.ratecv(audio_data, 2, 1, in_rate, out_rate, None)
    return converted
