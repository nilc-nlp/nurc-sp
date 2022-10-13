import numpy as np
import librosa
from pydub import AudioSegment

def get_preprocessing_function(model_name, sr):
    if "gain-normalization" in model_name:
        # https://github.com/alefiury/SE-R-2022-SER-Track/blob/main/utils/utils.py
        def pre_proc_audio(audio_path, sr=sr, target_dbfs=-31.187887972911266):
            sound = AudioSegment.from_file(audio_path, format="wav")
            sound = sound.set_channels(1)
            change_in_dBFS = target_dbfs - sound.dBFS
            # Apply normalization
            normalized_sound = sound.apply_gain(change_in_dBFS)
            # Convert array of bytes back to array of samples in the range [-1, 1]
            # This enables to work wih the audio without saving on disk
            norm_audio_samples = np.array(
                normalized_sound.get_array_of_samples()
            ).astype(np.float32, order='C') / 32768.0
            return norm_audio_samples
    else:
        # Skip preprocessing
        def pre_proc_audio(audio_path):
            audio, _ = librosa.load(audio_path, sr=sr)
            return audio

    return pre_proc_audio