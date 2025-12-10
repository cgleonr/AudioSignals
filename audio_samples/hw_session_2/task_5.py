import librosa
import numpy as np
import os


MY_TARGET_PITCH = 2312
MY_TARGET_TIMBRE = 2511.4

# Tolerances (How strict is the security?)
PITCH_TOLERANCE = 30     # +/- 30 Hz
TIMBRE_TOLERANCE = 500   # +/- 500 Hz

def get_voice_features(file_path):
    """
    Analyzes an audio file and returns its Pitch and Timbre stats.
    """

    y, sr = librosa.load(file_path)


    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    

    t_slice = magnitudes > np.median(magnitudes)
    valid_pitches = pitches[t_slice]
    

    valid_pitches = valid_pitches[valid_pitches > 0]
    
    if len(valid_pitches) == 0:
        return 0, 0
        
    avg_pitch = np.mean(valid_pitches)

    centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    avg_centroid = np.mean(centroids)

    return avg_pitch, avg_centroid

def run_security_check(file_path):
    """
    Decides if the file matches the user based on config.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"\n--- ANALYZING: {file_path} ---")
    pitch, timbre = get_voice_features(file_path)
    
    print(f"Detected Pitch:   {pitch:.2f} Hz")
    print(f"Detected Timbre:  {timbre:.2f} Hz")

    pitch_match = (MY_TARGET_PITCH - PITCH_TOLERANCE) <= pitch <= (MY_TARGET_PITCH + PITCH_TOLERANCE)
    timbre_match = (MY_TARGET_TIMBRE - TIMBRE_TOLERANCE) <= timbre <= (MY_TARGET_TIMBRE + TIMBRE_TOLERANCE)

    if pitch_match and timbre_match:
        print(">>> ACCESS GRANTED: Identity Confirmed.")
    else:
        print(">>> ACCESS DENIED: Voice does not match profile.")
        if not pitch_match: print(f"    (Reason: Pitch mismatch. Expected ~{MY_TARGET_PITCH})")
        if not timbre_match: print(f"    (Reason: Timbre mismatch. Expected ~{MY_TARGET_TIMBRE})")


# print("--- CALIBRATION RUN ---")
# print(get_voice_features("audio_samples/hw_session_2/heed.wav")) 


run_security_check("audio_samples/hw_session_2/heed.wav")
run_security_check("audio_samples/hw_session_2/heed_impostor.wav")