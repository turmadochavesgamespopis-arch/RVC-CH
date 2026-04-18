import os
import json
import time
import hashlib
import shutil
import tempfile
import zipfile
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List

import gradio as gr
import librosa
import soundfile as sf
import ffmpeg

# ========== CONFIGURAÇÕES ==========
BASE_DIR = Path("/app") if os.path.exists("/app") else Path.cwd()
JOBS_DIR = BASE_DIR / "jobs"
MODELS_DIR = BASE_DIR / "rvc_models"
DATASETS_DIR = BASE_DIR / "datasets"
LOGS_DIR = BASE_DIR / "logs"
CONVERSION_DIR = BASE_DIR / "conversion_outputs"

for d in [JOBS_DIR, MODELS_DIR, DATASETS_DIR, LOGS_DIR, CONVERSION_DIR]:
    d.mkdir(exist_ok=True, parents=True)

# ========== FUNÇÕES AUXILIARES ==========
def get_audio_duration(file_path: str) -> float:
    try:
        probe = ffmpeg.probe(file_path)
        stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
        if stream:
            return float(stream['duration'])
    except:
        pass
    y, sr = librosa.load(file_path, sr=None)
    return len(y) / sr

def validate_duration(files: List[str], min_min=3, max_min=40):
    total = 0.0
    for f in files:
        total += get_audio_duration(f)
    total_min = total / 60.0
    if total_min < min_min:
        return False, f"❌ Duração total muito curta: {total_min:.1f} min (mínimo {min_min} min)"
    if total_min > max_min:
        return False, f"❌ Duração total muito longa: {total_min:.1f} min (máximo {max_min} min)"
    return True, f"✅ Duração total válida: {total_min:.1f} min"

def prepare_dataset(files: List[str], job_id: str) -> Path:
    out_dir = DATASETS_DIR / job_id
    out_dir.mkdir(exist_ok=True)
    for i, file_path in enumerate(files):
        out_path = out_dir / f"audio_{i:04d}.wav"
        if file_path.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
            with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
                cmd = f"ffmpeg -i {file_path} -q:a 0 -map a {tmp.name} -y"
                subprocess.run(cmd, shell=True, check=True, capture_output=True)
                y, sr = librosa.load(tmp.name, sr=16000, mono=True)
                sf.write(out_path, y, sr)
        else:
            y, sr = librosa.load(file_path, sr=16000, mono=True)
            sf.write(out_path, y, sr)
    return out_dir

def run_training(job_id: str, model_name: str, dataset_dir: Path, epochs: int, batch_size: int, index_rate: float):
    job_file = JOBS_DIR / f"{job_id}.json"
    log_file = LOGS_DIR / f"{job_id}.txt"
    with open(job_file, 'r') as f:
        job = json.load(f)
    job['status'] = 'training'
    job['started_at'] = datetime.now().isoformat()
    with open(job_file, 'w') as f:
        json.dump(job, f, indent=4)

    try:
        # Comando para treinar usando o Ultimate RVC instalado
        cmd = [
            sys.executable, "-m", "ultimate_rvc", "train",
            "--model_name", model_name,
            "--dataset_path", str(dataset_dir),
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--index_rate", str(index_rate)
        ]
        with open(log_file, 'w') as log_f:
            process = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)
            process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Falha no treinamento (código {process.returncode})")

        # Localiza os arquivos gerados
        pth_path = MODELS_DIR / f"{model_name}.pth"
        index_path = MODELS_DIR / f"{model_name}.index"
        if not pth_path.exists():
            possible = list(Path("/app").rglob(f"{model_name}.pth"))
            if possible:
                pth_path = possible[0]
                index_path = pth_path.parent / f"{model_name}.index"
        if not pth_path.exists() or not index_path.exists():
            raise FileNotFoundError("Arquivos .pth ou .index não encontrados após treino.")

        # Cria o ZIP com os três arquivos
        zip_path = MODELS_DIR / f"{model_name}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(pth_path, f"{model_name}.pth")
            zf.write(index_path, f"{model_name}.index")
            metadata = {
                "name": model_name,
                "description": job.get('description', ''),
                "tags": job.get('tags', []),
                "epochs": epochs,
                "batch_size": batch_size,
                "index_rate": index_rate,
                "created_at": datetime.now().isoformat(),
                "duration_min": job.get('duration_min', 0)
            }
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as mf:
                json.dump(metadata, mf)
                mf.flush()
                zf.write(mf.name, "metadata.json")
        job['status'] = 'completed'
        job['download_url'] = str(zip_path)
        job['completed_at'] = datetime.now().isoformat()
    except Exception as e:
        job['status'] = 'failed'
        job['error'] = str(e)
    finally:
        with open(job_file, 'w') as f:
            json.dump(job, f, indent=4)

def start_training(model_name, description, tags, photo, audio_files, epochs, batch_size, index_rate, progress=gr.Progress()):
    if not model_name.strip():
        raise gr.Error("Nome do modelo é obrigatório.")
    if not description.strip():
        raise gr.Error("Descrição é obrigatória.")
    if not tags.strip():
        raise gr.Error("Tags são obrigatórias.")
    if photo is None:
        raise gr.Error("Foto do modelo é obrigatória.")
    if not audio_files:
        raise gr.Error("Envie pelo menos um arquivo de áudio ou vídeo.")
    valid, msg = validate_duration(audio_files, 3, 40)
    if not valid:
        raise gr.Error(msg)

    job_id = hashlib.md5(f"{model_name}_{time.time()}".encode()).hexdigest()[:8]
    dataset_dir = prepare_dataset(audio_files, job_id)
    photo_path = JOBS_DIR / f"{job_id}_photo{Path(photo.name).suffix}"
    shutil.copy(photo, photo_path)

    job = {
        "job_id": job_id,
        "model_name": model_name,
        "description": description,
        "tags": tags.split(','),
        "photo": str(photo_path),
        "duration_min": sum(get_audio_duration(f) for f in audio_files) / 60.0,
        "epochs": epochs,
        "batch_size": batch_size,
        "index_rate": index_rate,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "type": "training"
    }
    job_file = JOBS_DIR / f"{job_id}.json"
    with open(job_file, 'w') as f:
        json.dump(job, f, indent=4)

    thread = threading.Thread(target=run_training, args=(job_id, model_name, dataset_dir, epochs, batch_size, index_rate))
    thread.daemon = True
    thread.start()
    return f"✅ Treinamento iniciado! Job ID: {job_id}. Acompanhe na aba Jobs."

def separate_vocals(audio_path: str):
    out_dir = tempfile.mkdtemp()
    try:
        from ultimate_rvc.separator import separate
        separate(audio_path, out_dir)
        vocals = os.path.join(out_dir, "vocals.wav")
        instrumental = os.path.join(out_dir, "instrumental.wav")
    except:
        import demucs_infer
        demucs_infer.separate(audio_path, out_dir, two_stems=True)
        vocals = os.path.join(out_dir, "vocals.wav")
        instrumental = os.path.join(out_dir, "no_vocals.wav")
    if not os.path.exists(vocals) or not os.path.exists(instrumental):
        raise RuntimeError("Falha na separação de vocais")
    return vocals, instrumental

def convert_voice(model_name: str, input_audio: str, progress=gr.Progress()):
    model_path = MODELS_DIR / f"{model_name}.pth"
    index_path = MODELS_DIR / f"{model_name}.index"
    if not model_path.exists():
        raise gr.Error(f"Modelo {model_name} não encontrado. Treine primeiro.")
    progress(0.2, desc="Separando vocais...")
    entrada_acapella, entrada_instrumental = separate_vocals(input_audio)
    progress(0.5, desc="Convertendo voz...")
    output_acapella = CONVERSION_DIR / "temp_saida_acapella.wav"
    cmd = [
        sys.executable, "-m", "ultimate_rvc", "convert",
        "--model", str(model_path),
        "--index", str(index_path),
        "--input", entrada_acapella,
        "--output", str(output_acapella),
        "--index_rate", "0.66"
    ]
    subprocess.run(cmd, check=True)
    progress(0.8, desc="Mixando resultado final...")
    instrumental = librosa.load(entrada_instrumental, sr=16000)[0]
    converted = librosa.load(output_acapella, sr=16000)[0]
    min_len = min(len(instrumental), len(converted))
    mixed = instrumental[:min_len] + converted[:min_len]
    saida_wav = CONVERSION_DIR / "saida.wav"
    sf.write(saida_wav, mixed, 16000)
    entrada_wav = CONVERSION_DIR / "entrada.wav"
    shutil.copy(input_audio, entrada_wav)
    shutil.copy(entrada_acapella, CONVERSION_DIR / "entrada_acapella.wav")
    shutil.copy(entrada_instrumental, CONVERSION_DIR / "entrada_instrumental.wav")
    shutil.copy(output_acapella, CONVERSION_DIR / "saida_acapella.wav")
    progress(1, desc="Conversão concluída!")
    return {
        "entrada.wav": str(entrada_wav),
        "entrada_acapella.wav": str(CONVERSION_DIR / "entrada_acapella.wav"),
        "entrada_instrumental.wav": str(CONVERSION_DIR / "entrada_instrumental.wav"),
        "saida_acapella.wav": str(CONVERSION_DIR / "saida_acapella.wav"),
        "saida.wav": str(saida_wav)
    }

def list_jobs():
    jobs = []
    for f in JOBS_DIR.glob("*.json"):
        with open(f, 'r') as fp:
            try:
                jobs.append(json.load(fp))
            except:
                continue
    jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return jobs

def get_job_details(job_id: str):
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        return None, None
    with open(job_file, 'r') as f:
        job = json.load(f)
    log_file = LOGS_DIR / f"{job_id}.txt"
    log_content = ""
    if log_file.exists():
        with open(log_file, 'r') as f:
            log_content = f.read()[-10000:]
    return job, log_content

def download_model(job_id: str):
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        return None
    with open(job_file, 'r') as f:
        job = json.load(f)
    if job.get('status') == 'completed' and job.get('download_url'):
        return job['download_url']
    return None

def refresh_model_list():
    return gr.Dropdown(choices=[p.stem for p in MODELS_DIR.glob("*.pth")])

# ========== INTERFACE GRADIO ==========
css = """
    .gradio-container { max-width: 1300px; margin: auto; }
    button { background: #4f46e5 !important; color: white !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("# 🎙️ Weights.gg - RVC Real com Jobs Persistente")
    with gr.Tabs():
        with gr.TabItem("🎯 Treinar Modelo"):
            with gr.Group():
                model_name = gr.Textbox(label="Nome do modelo")
                description = gr.Textbox(label="Descrição", lines=2)
                tags = gr.Textbox(label="Tags (separadas por vírgula)")
                photo = gr.File(label="Foto do modelo", file_types=["image"])
                audio_files = gr.File(label="Áudios/Vídeos (3‑40 min total)", file_count="multiple")
                with gr.Row():
                    epochs = gr.Slider(label="Épocas", minimum=50, maximum=1000, step=50, value=300)
                    batch_size = gr.Slider(label="Batch size", minimum=4, maximum=32, step=2, value=8)
                    index_rate = gr.Slider(label="Index rate", minimum=0.0, maximum=1.0, step=0.05, value=0.66)
                train_btn = gr.Button("🚀 Iniciar Treinamento", variant="primary")
                train_status = gr.Textbox(label="Status", interactive=False)
                train_btn.click(start_training, inputs=[model_name, description, tags, photo, audio_files, epochs, batch_size, index_rate], outputs=train_status, queue=False)
        with gr.TabItem("📋 Jobs"):
            refresh_btn = gr.Button("🔄 Atualizar lista")
            jobs_table = gr.Dataframe(headers=["Job ID", "Modelo", "Status", "Épocas", "Duração (min)", "Criado em"], datatype=["str", "str", "str", "number", "number", "str"], interactive=False)
            job_id_input = gr.Textbox(label="Job ID (clique na linha)", interactive=True)
            job_details = gr.JSON(label="Detalhes do Job")
            job_log = gr.Textbox(label="Log do treinamento", lines=15, interactive=False)
            download_btn = gr.Button("📦 Baixar modelo (.zip)")
            download_file = gr.File(label="Arquivo ZIP")
            def refresh_jobs():
                jobs = list_jobs()
                data = [[j['job_id'], j['model_name'], j['status'], j['epochs'], round(j['duration_min'],1), j['created_at'][:19]] for j in jobs]
                return data
            def on_select(evt: gr.SelectData, df):
                if df is not None and evt.index[0] < len(df):
                    job_id = df[evt.index[0]][0]
                    details, log = get_job_details(job_id)
                    return job_id, details, log
                return "", None, ""
            def on_download(jid):
                return download_model(jid)
            refresh_btn.click(refresh_jobs, outputs=jobs_table)
            jobs_table.select(on_select, inputs=[jobs_table], outputs=[job_id_input, job_details, job_log])
            download_btn.click(on_download, inputs=job_id_input, outputs=download_file)
        with gr.TabItem("🎤 Conversão"):
            with gr.Group():
                model_list = gr.Dropdown(label="Modelo treinado", choices=[p.stem for p in MODELS_DIR.glob("*.pth")], interactive=True)
                refresh_models_btn = gr.Button("🔄 Atualizar lista")
                input_file = gr.File(label="Áudio ou vídeo para converter", file_types=["audio", "video"])
                convert_btn = gr.Button("🎧 Converter", variant="primary")
                convert_status = gr.Textbox(label="Status", interactive=False)
                with gr.Row():
                    entrada_wav = gr.Audio(label="entrada.wav (original)", type="filepath")
                    entrada_acapella = gr.Audio(label="entrada_acapella.wav (vocais extraídos)", type="filepath")
                    entrada_instrumental = gr.Audio(label="entrada_instrumental.wav (instrumental)", type="filepath")
                with gr.Row():
                    saida_acapella = gr.Audio(label="saida_acapella.wav (voz convertida)", type="filepath")
                    saida_wav = gr.Audio(label="saida.wav (versão completa)", type="filepath")
                def update_model_list():
                    return gr.Dropdown(choices=[p.stem for p in MODELS_DIR.glob("*.pth")])
                refresh_models_btn.click(update_model_list, outputs=model_list)
                def convert_and_output(model, file, progress=gr.Progress()):
                    if not model:
                        raise gr.Error("Selecione um modelo.")
                    if not file:
                        raise gr.Error("Envie um arquivo.")
                    outputs = convert_voice(model, file, progress)
                    return (outputs["entrada.wav"], outputs["entrada_acapella.wav"], outputs["entrada_instrumental.wav"], outputs["saida_acapella.wav"], outputs["saida.wav"])
                convert_btn.click(convert_and_output, inputs=[model_list, input_file], outputs=[entrada_wav, entrada_acapella, entrada_instrumental, saida_acapella, saida_wav], queue=True)
    demo.load(refresh_model_list, outputs=model_list)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", server_port=7860)
