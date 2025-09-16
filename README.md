## RAG Flask App (ID/EN)

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/) [![Flask](https://img.shields.io/badge/Flask-3.x-green)](https://flask.palletsprojects.com/) [![Lint](https://img.shields.io/badge/Lint-ruff-black)](https://github.com/astral-sh/ruff) [![Precommit](https://img.shields.io/badge/pre--commit-enabled-brightgreen)](https://pre-commit.com/)

### Struktur Proyek
```
.
├─ app.py                      # Entrypoint + server
├─ templates/
│  └─ index.html
├─ enterprise_knowledge_base/  # sumber data lokal / sinkron GDrive
├─ requirements.txt
├─ README.md
└─ .env (opsional)
```

### Persiapan
1) Python 3.10+
2) Virtualenv dan install dependency:
```bash
python -m venv .venv
. .venv/Scripts/activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
3) Buat `.env` (opsional tapi disarankan):
```env
# Flask
FLASK_SECRET_KEY=change-me
FLASK_DEBUG=true
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
MAX_CONTENT_LENGTH_MB=50
UPLOAD_FOLDER=temp_uploads

# Knowledge Base
KNOWLEDGE_BASE_PATH=enterprise_knowledge_base

# Embeddings
EMBEDDINGS_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDINGS_DEVICE=cpu
RETRIEVER_K=4

# LLM Provider: salah satu [ollama, groq]
LLM_PROVIDER=ollama
LLM_TEMPERATURE=0.3

# Ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

# Groq
GROQ_API_KEY=
GROQ_MODEL_NAME=llama3-70b-8192

# Google Drive (opsional)
GDRIVE_API_KEY=
GDRIVE_KB_FOLDER_ID=
# atau layanan privat:
# GDRIVE_SERVICE_ACCOUNT_JSON_PATH=C:\\path\\to\\service_account.json
```

### Menjalankan Aplikasi
```bash
python app.py
```
Lalu buka `http://localhost:5000`.

### Endpoint API
- GET `/health`
- POST `/ask` body: `{ "question": "..." }`
- POST `/upload` form-data: `question`, `files[]`
- POST `/sync`
- GET `/conversation`

### Catatan
- KB dibaca dari `enterprise_knowledge_base` (dapat disinkronkan dari Google Drive).
- File upload diproses sementara dan dihapus otomatis.
- Mendukung: PDF, DOC/DOCX, XLS/XLSX, PPT/PPTX, PNG/JPG/JPEG, TXT/CSV/JSON/MD.
- Jawaban mengikuti bahasa pertanyaan (ID/EN).

### Fitur Utama
- RAG multibahasa (ID/EN) dengan grounding ketat pada dokumen.
- Upload file sementara (OCR gambar) tanpa disimpan permanen.
- Sinkronisasi opsional dari Google Drive (rekursif subfolder).
- Batas sumber tampil maksimal 3, unik per file.
- Fallback jawaban: `Tidak paham` jika tidak ada informasi relevan.

### Arsitektur Singkat
- Single Flask app (`app.py`) yang memuat:
  - Loader dokumen + chunking + FAISS vector store
  - LLM (Ollama/Groq) dan prompt terkurasi
  - Endpoint: `/health`, `/ask`, `/upload`, `/sync`, `/conversation`

### Batasan & Keamanan
- Tidak ada penyimpanan permanen untuk file upload.
- Jangan commit `.env`/API keys. Gunakan `.env.example` sebagai template.
- Batasi model/endpoint LLM sesuai kebijakan Anda.

### Lisensi
Proyek ini dirilis di bawah lisensi MIT. Lihat file `LICENSE`.

### Perilaku Jawaban (Penting)
- Asisten hanya menjawab berdasarkan dokumen yang ada atau yang diunggah.
- Jika informasi tidak ditemukan, asisten akan menjawab: `Maaf, saya tidak menemukan jawabannya di dokumen ini`.

