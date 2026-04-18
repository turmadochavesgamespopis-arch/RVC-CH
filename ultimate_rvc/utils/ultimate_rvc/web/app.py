import os
import time
import hashlib
import shutil
import threading
from pathlib import Path
from datetime import datetime

import gradio as gr

from ultimate_rvc.utils.audio_utils import validate_duration, prepare_dataset
from ultimate_rvc.utils.jobs_manager import save_job, load_job, list_jobs, update_job
from ultimate_rvc.training.train_job import run_training
from ultimate_rvc.inference.convert_job import run_conversion

# ========== CONFIG ==========
BASE_DIR = Path(".")
JOBS_DIR = BASE_DIR / "jobs"
MODELS_DIR = BASE_DIR / "rvc_models"
DATASETS_DIR = BASE_DIR / "datasets"
LOGS_DIR = BASE_DIR / "logs"
CONV_OUTPUTS_DIR = BASE_DIR / "conversion_outputs"

for d in [JOBS_DIR, MODELS_DIR, DATASETS_DIR, LOGS_DIR, CONV_OUTPUTS_DIR]:
    d.mkdir(exist_ok=True)

# ========== FUNÇÕES DA UI ==========
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
        raise gr.Error("Envie pelo menos um arquivo de áudio/vídeo.")

    valid, msg, total_min = validate_duration(audio_files, 3, 40)
    if not valid:
        raise gr.Error(msg)

    job_id = hashlib.md5(f"{model_name}_{time.time()}".encode()).hexdigest()[:8]
    dataset_dir = prepare_dataset(audio_files, job_id, DATASETS_DIR)
    photo_path = JOBS_DIR / f"{job_id}_photo{Path(photo.name).suffix}"
    shutil.copy(photo, photo_path)

    job = {
        "job_id": job_id,
        "model_name": model_name,
        "description": description,
        "tags": tags.split(","),
        "photo": str(photo_path),
        "duration_min": total_min,
        "epochs": epochs,
        "batch_size": batch_size,
        "index_rate": index_rate,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "type": "training"
    }
    save_job(job)

    thread = threading.Thread(
        target=run_training,
        args=(job_id, model_name, dataset_dir, epochs, batch_size, index_rate, description, tags.split(","), str(photo_path), total_min)
    )
    thread.daemon = True
    thread.start()

    return f"✅ Treinamento iniciado! Job ID: {job_id} (acompanhe na aba Jobs)"


def start_conversion(model_name, input_file, progress=gr.Progress()):
    if not model_name:
        raise gr.Error("Selecione um modelo.")
    if not input_file:
        raise gr.Error("Envie um arquivo de áudio ou vídeo.")

    job_id = hashlib.md5(f"convert_{model_name}_{time.time()}".encode()).hexdigest()[:8]
    output_dir = CONV_OUTPUTS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    job = {
        "job_id": job_id,
        "model_name": model_name,
        "input_file": input_file,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "type": "conversion"
    }
    save_job(job)

    thread = threading.Thread(target=run_conversion, args=(job_id, model_name, input_file, output_dir))
    thread.daemon = True
    thread.start()

    return f"🔄 Conversão iniciada! Job ID: {job_id} (acompanhe na aba Jobs)"


def refresh_jobs_table():
    jobs = list_jobs()
    data = []
    for j in jobs:
        data.append([
            j["job_id"],
            j.get("model_name", j.get("input_file", "N/A")),
            j["status"],
            j.get("epochs", "-"),
            round(j.get("duration_min", 0), 1) if j.get("duration_min") else "-",
            j["created_at"][:19]
        ])
    return data


def on_job_select(evt: gr.SelectData, df):
    if df is not None and evt.index[0] < len(df):
        job_id = df[evt.index[0]][0]
        job = load_job(job_id)
        log_path = LOGS_DIR / f"{job_id}.txt" if job.get("type") == "training" else LOGS_DIR / f"convert_{job_id}.txt"
        log_content = ""
        if log_path.exists():
            with open(log_path, "r") as f:
                log_content = f.read()[-10000:]
        return job_id, job, log_content
    return "", None, ""


def download_conversion_output(job_id):
    job = load_job(job_id)
    if job and job.get("type") == "conversion" and job.get("status") == "completed":
        output_dir = CONV_OUTPUTS_DIR / job_id
        # Retorna um dicionário com os 5 arquivos
        files = {
            "entrada.wav": str(output_dir / "entrada.wav"),
            "entrada_acapella.wav": str(output_dir / "entrada_acapella.wav"),
            "entrada_instrumental.wav": str(output_dir / "entrada_instrumental.wav"),
            "saida_acapella.wav": str(output_dir / "saida_acapella.wav"),
            "saida.wav": str(output_dir / "saida.wav")
        }
        return files
    return None


def download_model_zip(job_id):
    job = load_job(job_id)
    if job and job.get("type") == "training" and job.get("status") == "completed":
        return job.get("download_url")
    return None


# ========== INTERFACE GRADIO ==========
css = """
    .gradio-container { max-width: 1400px; margin: auto; }
    button { background: #4f46e5 !important; color: white !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("# 🎙️ RVC-CH - Weights.gg Style (Jobs + 5 outputs)")

    with gr.Tabs():
        # --- TREINAMENTO ---
        with gr.TabItem("🎯 Treinar Modelo"):
            with gr.Group():
                model_name = gr.Textbox(label="Nome do modelo", placeholder="ex: minha_voz")
                description = gr.Textbox(label="Descrição", lines=2)
                tags = gr.Textbox(label="Tags (separadas por vírgula)", placeholder="pop, rock, feminino")
                photo = gr.File(label="Foto do modelo", file_types=["image"])
                audio_files = gr.File(label="Áudios/Vídeos (3‑40 min total)", file_count="multiple")
                with gr.Row():
                    epochs = gr.Slider(label="Épocas", minimum=50, maximum=1000, step=50, value=300)
                    batch_size = gr.Slider(label="Batch size", minimum=4, maximum=32, step=2, value=8)
                    index_rate = gr.Slider(label="Index rate", minimum=0.0, maximum=1.0, step=0.05, value=0.66)
                train_btn = gr.Button("🚀 Iniciar Treinamento", variant="primary")
                train_status = gr.Textbox(label="Status", interactive=False)
                train_btn.click(
                    start_training,
                    inputs=[model_name, description, tags, photo, audio_files, epochs, batch_size, index_rate],
                    outputs=train_status,
                    queue=False
                )

        # --- JOBS ---
        with gr.TabItem("📋 Jobs"):
            refresh_btn = gr.Button("🔄 Atualizar lista")
            jobs_table = gr.Dataframe(
                headers=["Job ID", "Nome/Arquivo", "Status", "Épocas", "Duração(min)", "Criado em"],
                datatype=["str", "str", "str", "str", "str", "str"],
                interactive=True
            )
            job_id_input = gr.Textbox(label="Job ID (clique na linha)", interactive=True)
            job_details = gr.JSON(label="Detalhes")
            job_log = gr.Textbox(label="Log", lines=15, interactive=False)
            download_btn = gr.Button("📦 Baixar resultado")
            download_file = gr.File(label="Arquivo(s)")

            refresh_btn.click(refresh_jobs_table, outputs=jobs_table)
            jobs_table.select(on_job_select, inputs=[jobs_table], outputs=[job_id_input, job_details, job_log])
            download_btn.click(
                lambda jid: download_model_zip(jid) or download_conversion_output(jid),
                inputs=job_id_input,
                outputs=download_file
            )

        # --- CONVERSÃO ---
        with gr.TabItem("🎤 Conversão"):
            with gr.Group():
                model_list = gr.Dropdown(label="Modelo treinado", choices=[p.stem for p in MODELS_DIR.glob("*.pth")])
                refresh_models = gr.Button("🔄 Atualizar modelos")
                input_audio = gr.File(label="Áudio ou vídeo para converter", file_types=["audio", "video"])
                convert_btn = gr.Button("🎧 Converter", variant="primary")
                convert_status = gr.Textbox(label="Status", interactive=False)

                gr.Markdown("### 🎵 Resultados (5 arquivos)")
                with gr.Row():
                    out_entrada = gr.Audio(label="entrada.wav", type="filepath")
                    out_entrada_acap = gr.Audio(label="entrada_acapella.wav", type="filepath")
                    out_entrada_inst = gr.Audio(label="entrada_instrumental.wav", type="filepath")
                with gr.Row():
                    out_saida_acap = gr.Audio(label="saida_acapella.wav", type="filepath")
                    out_saida = gr.Audio(label="saida.wav", type="filepath")

                def refresh_models_fn():
                    return gr.Dropdown(choices=[p.stem for p in MODELS_DIR.glob("*.pth")])
                refresh_models.click(refresh_models_fn, outputs=model_list)

                # A conversão aqui é síncrona? O usuário pediu que também apareça na aba Jobs.
                # Vamos fazer assíncrona: ao clicar, cria um job e redireciona para a aba Jobs.
                convert_btn.click(
                    start_conversion,
                    inputs=[model_list, input_audio],
                    outputs=convert_status,
                    queue=False
                )
                # Para exibir os outputs depois, o usuário terá que ir na aba Jobs e baixar.
                # Mas podemos também adicionar um botão "Buscar resultado" que carrega os 5 áudios.
                # Vou adicionar um campo para digitar o job_id da conversão e carregar os outputs.
                gr.Markdown("---\n### 🔍 Buscar resultado da conversão pelo Job ID")
                conv_job_id = gr.Textbox(label="Job ID da conversão")
                load_conv_btn = gr.Button("Carregar áudios")
                load_conv_btn.click(
                    download_conversion_output,
                    inputs=conv_job_id,
                    outputs=[out_entrada, out_entrada_acap, out_entrada_inst, out_saida_acap, out_saida]
                )

    # Carrega modelos e jobs ao iniciar
    demo.load(refresh_models_fn, outputs=model_list)
    demo.load(refresh_jobs_table, outputs=jobs_table)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2)
    demo.launch(server_name="0.0.0.0", server_port=7860)
