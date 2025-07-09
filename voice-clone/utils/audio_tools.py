import librosa
def trim_silence(path):
    y, sr = librosa.load(path, sr=22050)
    y_trim, _ = librosa.effects.trim(y)
    return y_trim, sr
