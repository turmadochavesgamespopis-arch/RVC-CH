# Adicione estas importações no topo do arquivo
import json
import hashlib
import threading
from datetime import datetime
from ultimate_rvc.utils.audio_utils import validate_duration, prepare_dataset

# ... (código existente) ...

# ========== SISTEMA DE JOBS ==========
JOBS_DIR = Path("jobs")
JOBS_DIR.mkdir(exist_ok=True)

def save_job(job_data: dict):
    """Salva os dados de um job em um arquivo JSON."""
    job_id = job_data["job_id"]
    with open(JOBS_DIR / f"{job_id}.json", "w") as f:
        json.dump(job_data, f, indent=4)

def list_jobs():
    """Retorna uma lista com todos os jobs."""
    jobs = []
    for job_file in JOBS_DIR.glob("*.json"):
        with open(job_file, "r") as f:
            jobs.append(json.load(f))
    jobs.sort(key=lambda x: x["created_at"], reverse=True)
    return jobs

# ========== FUNÇÃO DE TREINAMENTO MODIFICADA ==========
def start_training(model_name, description, tags, photo, audio_files, epochs, batch_size, index_rate, progress=gr.Progress()):
    # Validações
    if not model_name.strip():
        raise gr.Error("❌ Nome do modelo é obrigatório.")
    if not description.strip():
        raise gr.Error("❌ Descrição é obrigatória.")
    if not tags.strip():
        raise gr.Error("❌ Tags são obrigatórias.")
    if photo is None:
        raise gr.Error("❌ Foto do modelo é obrigatória.")
    if not audio_files:
        raise gr.Error("❌ Envie pelo menos um arquivo de áudio ou vídeo.")

    # Valida duração total do áudio
    valid, msg, total_min = validate_duration(audio_files, 3, 40)
    if not valid:
        raise gr.Error(msg)

    # Prepara dataset e salva foto
    job_id = hashlib.md5(f"{model_name}_{time.time()}".encode()).hexdigest()[:8]
    dataset_dir = prepare_dataset(audio_files, job_id)
    photo_path = JOBS_DIR / f"{job_id}_photo{Path(photo.name).suffix}"
    shutil.copy(photo, photo_path)

    # Cria o job
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
        "started_at": None,
        "completed_at": None,
        "download_url": None,
        "error": None
    }
    save_job(job)

    # Inicia o treinamento em uma thread separada (use a função real do RVC-CH aqui)
    thread = threading.Thread(target=run_training, args=(job_id, model_name, dataset_dir, epochs, batch_size, index_rate))
    thread.daemon = True
    thread.start()

    return f"✅ Treinamento iniciado! Job ID: {job_id}. Acompanhe na aba 'Jobs'."

# ========== CONSTRUÇÃO DA INTERFACE ==========
with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    # ... (código existente) ...

    with gr.Tabs():
        # Aba de Treinamento (já existente, mas com novos campos)
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
                train_btn.click(start_training, inputs=[model_name, description, tags, photo, audio_files, epochs, batch_size, index_rate], outputs=train_status, queue=False)

        # Nova Aba "Jobs"
        with gr.TabItem("📋 Jobs"):
            refresh_btn = gr.Button("🔄 Atualizar lista")
            jobs_table = gr.Dataframe(
                headers=["Job ID", "Modelo", "Status", "Épocas", "Duração (min)", "Criado em"],
                datatype=["str", "str", "str", "number", "number", "str"],
                interactive=False
            )
            job_id_input = gr.Textbox(label="Job ID (clique na linha)", interactive=True)
            job_details = gr.JSON(label="Detalhes do Job")
            download_btn = gr.Button("📦 Baixar modelo (.zip)")
            download_file = gr.File(label="Arquivo ZIP")

            def refresh_jobs():
                jobs = list_jobs()
                data = [[j['job_id'], j['model_name'], j['status'], j['epochs'], round(j['duration_min'],1), j['created_at'][:19]] for j in jobs]
                return data

            def on_select(evt: gr.SelectData, df):
                if df is not None and evt.index[0] < len(df):
                    job_id = df[evt.index[0]][0]
                    job_file = JOBS_DIR / f"{job_id}.json"
                    with open(job_file, "r") as f:
                        details = json.load(f)
                    return job_id, details
                return "", None

            refresh_btn.click(refresh_jobs, outputs=jobs_table)
            jobs_table.select(on_select, inputs=[jobs_table], outputs=[job_id_input, job_details])
            download_btn.click(lambda jid: download_model(jid), inputs=job_id_input, outputs=download_file)
