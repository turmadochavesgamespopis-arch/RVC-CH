import ffmpeg
import librosa
import soundfile as sf
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple

def get_audio_duration(file_path: str) -> float:
    """Retorna duração em segundos de qualquer arquivo de áudio/vídeo."""
    try:
        probe = ffmpeg.probe(file_path)
        stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
        if stream:
            return float(stream['duration'])
    except:
        pass
    y, sr = librosa.load(file_path, sr=None)
    return len(y) / sr

def validate_duration(files: List[str], min_min: int = 3, max_min: int = 40) -> Tuple[bool, str, float]:
    """Valida se a duração total está dentro dos limites."""
    total_duration = 0.0
    for f in files:
        total_duration += get_audio_duration(f)
    total_min = total_duration / 60.0
    if total_min < min_min:
        return False, f"❌ Duração total muito curta: {total_min:.1f} min (mínimo {min_min} min)", total_min
    if total_min > max_min:
        return False, f"❌ Duração total muito longa: {total_min:.1f} min (máximo {max_min} min)", total_min
    return True, f"✅ Duração total válida: {total_min:.1f} min", total_min

def extract_audio_from_video(video_path: str, output_wav: str) -> None:
    """Extrai o áudio de um arquivo de vídeo para WAV mono 16kHz."""
    cmd = f"ffmpeg -i {video_path} -q:a 0 -map a -ac 1 -ar 16000 {output_wav} -y"
    subprocess.run(cmd, shell=True, check=True)

def prepare_dataset(files: List[str], job_id: str) -> Path:
    """Converte todos os arquivos para WAV mono 16kHz e salva em uma pasta."""
    out_dir = Path(f"datasets/{job_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, file_path in enumerate(files):
        out_path = out_dir / f"audio_{i:04d}.wav"
        if file_path.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
            with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
                extract_audio_from_video(file_path, tmp.name)
                y, sr = librosa.load(tmp.name, sr=16000, mono=True)
                sf.write(out_path, y, sr)
        else:
            y, sr = librosa.load(file_path, sr=16000, mono=True)
            sf.write(out_path, y, sr)
    return out_dir
