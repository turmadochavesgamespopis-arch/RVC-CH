import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import ffmpeg
import librosa
import soundfile as sf


def get_audio_duration(file_path: str) -> float:
    """Retorna duração em segundos de qualquer arquivo de áudio/vídeo."""
    try:
        probe = ffmpeg.probe(file_path)
        stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
        if stream and stream.get('duration'):
            return float(stream['duration'])
    except:
        pass
    # fallback
    y, sr = librosa.load(file_path, sr=None)
    return len(y) / sr


def validate_duration(
    files: List[str], min_min: int = 3, max_min: int = 40
) -> Tuple[bool, str, float]:
    total = sum(get_audio_duration(f) for f in files) / 60.0
    if total < min_min:
        return False, f"❌ Duração total {total:.1f} min < mínimo {min_min} min", total
    if total > max_min:
        return False, f"❌ Duração total {total:.1f} min > máximo {max_min} min", total
    return True, f"✅ Duração total válida: {total:.1f} min", total


def extract_audio_from_video(video_path: str, output_wav: str) -> None:
    cmd = [
        "ffmpeg", "-i", video_path,
        "-q:a", "0", "-map", "a",
        "-ac", "1", "-ar", "16000",
        output_wav, "-y"
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def prepare_dataset(files: List[str], job_id: str, base_dir: Path = Path("datasets")) -> Path:
    out_dir = base_dir / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, file_path in enumerate(files):
        out_path = out_dir / f"audio_{i:04d}.wav"
        if file_path.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                extract_audio_from_video(file_path, tmp.name)
                y, sr = librosa.load(tmp.name, sr=16000, mono=True)
                sf.write(out_path, y, sr)
        else:
            y, sr = librosa.load(file_path, sr=16000, mono=True)
            sf.write(out_path, y, sr)
    return out_dir
