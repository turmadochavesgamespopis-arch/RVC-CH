import subprocess
import shutil
from pathlib import Path
from datetime import datetime

import librosa
import soundfile as sf
from demucs_infer import separate  # ou do RVC-CH, use o separador que já existe

from ultimate_rvc.utils.jobs_manager import update_job


def run_conversion(job_id: str, model_name: str, input_file: str, output_dir: Path):
    log_file = Path(f"logs/convert_{job_id}.txt")
    log_file.parent.mkdir(exist_ok=True)
    update_job(job_id, status="processing", started_at=datetime.now().isoformat())

    try:
        # 1. Extrai áudio se for vídeo
        input_wav = output_dir / "entrada.wav"
        if input_file.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
            cmd = f"ffmpeg -i {input_file} -q:a 0 -map a -ac 1 -ar 16000 {input_wav} -y"
            subprocess.run(cmd, shell=True, check=True)
        else:
            shutil.copy(input_file, input_wav)

        # 2. Separa vocais e instrumental
        update_job(job_id, status="separating_vocals")
        sep_dir = output_dir / "separated"
        separate.main(["--out", str(sep_dir), "--two-stems=vocals", str(input_wav)])
        # O demucs_infer gera: sep_dir/demucs_separated/.../vocals.wav e no_vocals.wav
        vocals = list(sep_dir.rglob("vocals.wav"))[0]
        instrumental = list(sep_dir.rglob("no_vocals.wav"))[0]
        shutil.copy(vocals, output_dir / "entrada_acapella.wav")
        shutil.copy(instrumental, output_dir / "entrada_instrumental.wav")

        # 3. Converte os vocais com o modelo RVC
        update_job(job_id, status="converting")
        model_path = Path("rvc_models") / f"{model_name}.pth"
        index_path = Path("rvc_models") / f"{model_name}.index"
        converted_acapella = output_dir / "saida_acapella.wav"
        # Chama o comando de inferência do RVC-CH
        cmd_convert = [
            "python", "-m", "ultimate_rvc.cli", "convert",
            "--model", str(model_path),
            "--index", str(index_path),
            "--input", str(vocals),
            "--output", str(converted_acapella)
        ]
        subprocess.run(cmd_convert, check=True)

        # 4. Mixa com o instrumental para gerar saida.wav
        instrumental_audio, sr = librosa.load(output_dir / "entrada_instrumental.wav", sr=16000)
        converted_audio, _ = librosa.load(converted_acapella, sr=16000)
        min_len = min(len(instrumental_audio), len(converted_audio))
        mixed = instrumental_audio[:min_len] + converted_audio[:min_len]
        sf.write(output_dir / "saida.wav", mixed, sr)

        update_job(job_id, status="completed", completed_at=datetime.now().isoformat())

    except Exception as e:
        update_job(job_id, status="failed", error=str(e))
        with open(log_file, "a") as lf:
            lf.write(f"\nERRO: {e}\n")
