"""Flask RAG application (ID/EN)

This app provides:
- Web UI (`/`) and REST API endpoints (`/health`, `/ask`, `/upload`, `/sync`, `/conversation`).
- RAG pipeline with multilingual support (Indonesian and English), OCR for images, and optional Google Drive sync.

Design notes:
- Keep answers grounded in provided documents. If the information is not available, respond in Indonesian: "Maaf, saya tidak menemukan jawabannya di dokumen ini".
- Detect question language and mirror it for the answer where applicable.
"""

from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import tempfile
import shutil
from pathlib import Path
import time
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv

# Document processing imports
from langchain_community.document_loaders import (
    TextLoader, CSVLoader, JSONLoader,
    PyPDFLoader, UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader, UnstructuredPowerPointLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
## Removed unused ConversationBufferMemory import

# Embeddings and LLM imports
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings

# Multi-language OCR support
import easyocr

# LLM options (Groq or Ollama only)
LLM_TYPE = None
try:
    from langchain_community.llms import Ollama
    LLM_TYPE = LLM_TYPE or "ollama"
except ImportError:
    pass
try:
    from langchain_groq import ChatGroq
    LLM_TYPE = LLM_TYPE or "groq"
except ImportError:
    pass

# Google Drive integration (optional)
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    import io
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False

# Load environment variables
load_dotenv()

# Flask app configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH_MB', '50')) * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'temp_uploads')
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')

# Create temp upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("googleapiclient").setLevel(logging.ERROR)
logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)

class MultiLanguageRAGSystem:
    """RAG System with multi-language support"""
    
    def __init__(self, knowledge_base_path: str = "enterprise_knowledge_base"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.rag_chain = None
        self.conversation_history = []
        self.gdrive_service = None
        self.gdrive_folder_id = os.getenv("GDRIVE_KB_FOLDER_ID")
        self.gdrive_api_key = os.getenv("GDRIVE_API_KEY")
        
        # Multi-language OCR reader
        try:
            self.ocr_reader = easyocr.Reader(['en', 'id'])  # English and Indonesian
            self.ocr_available = True
        except:
            self.ocr_available = False
            logger.warning("OCR not available - image processing disabled")
        
        # Supported file extensions
        self.supported_extensions = {
            '.txt', '.csv', '.json', '.md',  # Text files
            '.pdf',                          # PDF
            '.doc', '.docx',                # Word
            '.xls', '.xlsx',                # Excel
            '.ppt', '.pptx',                # PowerPoint
            '.png', '.jpg', '.jpeg'         # Images
        }
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the RAG system components"""
        try:
            self._setup_embeddings()
            self._setup_llm()
            # Optional: sync Google Drive knowledge base recursively
            self._maybe_setup_gdrive_and_sync()
            self._load_knowledge_base()
            self._create_rag_chain()
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise

    def _maybe_setup_gdrive_and_sync(self):
        """If configured, setup Google Drive client and sync KB folder recursively."""
        if not GDRIVE_AVAILABLE:
            return
        if not self.gdrive_folder_id:
            return
        try:
            # Prefer API key if provided (works for publicly accessible folders/files)
            if self.gdrive_api_key:
                self.gdrive_service = build('drive', 'v3', developerKey=self.gdrive_api_key)
                logger.info("Google Drive client initialized with API key (public access required)")
            else:
                # Fallback to Service Account if configured
                service_account_json = os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON_PATH")
                if not (service_account_json and Path(service_account_json).exists()):
                    logger.info("Google Drive not configured: set GDRIVE_API_KEY for public folders or GDRIVE_SERVICE_ACCOUNT_JSON_PATH for private access")
                    return
                scopes = ["https://www.googleapis.com/auth/drive.readonly"]
                credentials = service_account.Credentials.from_service_account_file(service_account_json, scopes=scopes)
                self.gdrive_service = build('drive', 'v3', credentials=credentials)
                logger.info("Google Drive client initialized with Service Account")

            # Ensure local KB dir exists
            self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
            # Sync
            self._gdrive_sync_folder_recursive(self.gdrive_folder_id, self.knowledge_base_path)
            logger.info("Google Drive sync completed")
        except Exception as e:
            logger.error(f"Google Drive setup/sync failed: {e}")

    def _gdrive_list_children(self, folder_id: str) -> List[Dict[str, Any]]:
        """List children (files and folders) for a Drive folder."""
        items = []
        page_token = None
        while True:
            response = self.gdrive_service.files().list(
                q=f"'{folder_id}' in parents and trashed=false",
                fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                pageToken=page_token
            ).execute()
            items.extend(response.get('files', []))
            page_token = response.get('nextPageToken')
            if not page_token:
                break
        return items

    def _gdrive_sync_folder_recursive(self, folder_id: str, local_dir: Path):
        """Download all files recursively from the Drive folder into local_dir."""
        children = self._gdrive_list_children(folder_id)
        for item in children:
            name = item['name']
            mime = item['mimeType']
            file_id = item['id']
            if mime == 'application/vnd.google-apps.folder':
                subdir = local_dir / name
                subdir.mkdir(parents=True, exist_ok=True)
                self._gdrive_sync_folder_recursive(file_id, subdir)
            else:
                # Determine local file path and export if Google Docs formats
                export_mime = None
                suffix = None
                if mime == 'application/vnd.google-apps.document':
                    export_mime = 'application/pdf'
                    suffix = '.pdf'
                elif mime == 'application/vnd.google-apps.spreadsheet':
                    export_mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    suffix = '.xlsx'
                elif mime == 'application/vnd.google-apps.presentation':
                    export_mime = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
                    suffix = '.pptx'
                # Build local file path
                if export_mime:
                    local_path = local_dir / f"{name}{suffix}"
                    self._gdrive_export_file(file_id, export_mime, local_path)
                else:
                    local_path = local_dir / name
                    self._gdrive_download_file(file_id, local_path)

    def _gdrive_download_file(self, file_id: str, local_path: Path):
        """Download a Drive file by file ID to local_path."""
        try:
            request = self.gdrive_service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(fh.getvalue())
            logger.info(f"Downloaded: {local_path}")
        except Exception as e:
            logger.error(f"Failed to download file {file_id} to {local_path}: {e}")

    def _gdrive_export_file(self, file_id: str, export_mime: str, local_path: Path):
        """Export a Google Doc/Sheet/Slide to a given MIME and save locally."""
        try:
            request = self.gdrive_service.files().export_media(fileId=file_id, mimeType=export_mime)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(fh.getvalue())
            logger.info(f"Exported: {local_path}")
        except Exception as e:
            logger.error(f"Failed to export file {file_id} to {local_path}: {e}")

    def reload_knowledge_base(self) -> int:
        """Re-sync from Drive if configured, then reload vector store and chain. Returns chunk count."""
        if self.gdrive_service and self.gdrive_folder_id:
            try:
                self._gdrive_sync_folder_recursive(self.gdrive_folder_id, self.knowledge_base_path)
            except Exception as e:
                logger.error(f"Drive re-sync failed: {e}")
        documents = self._load_documents_from_path(self.knowledge_base_path)
        if documents:
            chunks = self._create_chunks(documents)
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            self._create_rag_chain()
            return len(chunks)
        else:
            return 0
    
    def _setup_embeddings(self):
        """Setup multilingual embedding model"""
        # Use multilingual model for better Indonesian support
        model_name = os.getenv("EMBEDDINGS_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        device = os.getenv("EMBEDDINGS_DEVICE", "cpu")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"Embeddings model loaded: {model_name}")
    
    def _setup_llm(self):
        """Setup Language Model based on availability"""
        try:
            # Allow env override for provider (only ollama or groq)
            env_provider = os.getenv("LLM_PROVIDER")
            global LLM_TYPE
            if env_provider in {"ollama", "groq"}:
                LLM_TYPE = env_provider

            temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))

            if LLM_TYPE == "ollama":
                ollama_model = os.getenv("OLLAMA_MODEL", "gemma3:1b")
                ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                self.llm = Ollama(model=ollama_model, base_url=ollama_base_url)
                # Test connection
                self.llm.invoke("Hello")
                logger.info("Ollama LLM connected")
            
            elif LLM_TYPE == "groq":
                self.llm = ChatGroq(
                    model_name=os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant"),
                    temperature=temperature,
                    groq_api_key=os.getenv("GROQ_API_KEY")
                )
                logger.info("Groq LLM connected")
            
            else:
                raise Exception("No LLM available - install Ollama or set GROQ_API_KEY")
                
        except Exception as e:
            logger.error(f"LLM setup failed: {e}")
            self.llm = None
    
    def _load_knowledge_base(self):
        """Load documents from knowledge base"""
        if not self.knowledge_base_path.exists():
            logger.warning(f"Knowledge base path not found: {self.knowledge_base_path}")
            self.vector_store = FAISS.from_texts(
                ["Empty knowledge base"], 
                self.embeddings,
                metadatas=[{"source": "empty"}]
            )
            return
        
        documents = self._load_documents_from_path(self.knowledge_base_path)
        
        if documents:
            chunks = self._create_chunks(documents)
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            logger.info(f"Knowledge base loaded: {len(chunks)} chunks")
        else:
            # Create empty vector store
            self.vector_store = FAISS.from_texts(
                ["Empty knowledge base"], 
                self.embeddings,
                metadatas=[{"source": "empty"}]
            )
    
    def _load_documents_from_path(self, path: Path) -> List[Document]:
        """Load all documents from a directory"""
        documents = []
        
        for file_path in path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    docs = self._load_single_file(file_path)
                    documents.extend(docs)
                    logger.info(f"Loaded: {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path.name}: {e}")
        
        return documents
    
    def _load_single_file(self, file_path: Path) -> List[Document]:
        """Load a single file based on its extension"""
        ext = file_path.suffix.lower()
        
        try:
            if ext in ['.txt', '.md']:
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif ext == '.csv':
                loader = CSVLoader(str(file_path))
            elif ext == '.json':
                loader = JSONLoader(str(file_path), jq_schema='.', text_content=False)
            elif ext == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif ext in ['.doc', '.docx']:
                loader = UnstructuredWordDocumentLoader(str(file_path))
            elif ext in ['.xls', '.xlsx']:
                loader = UnstructuredExcelLoader(str(file_path))
            elif ext in ['.ppt', '.pptx']:
                loader = UnstructuredPowerPointLoader(str(file_path))
            elif ext in ['.png', '.jpg', '.jpeg']:
                return self._load_image_file(file_path)
            else:
                loader = TextLoader(str(file_path), encoding='utf-8')
            
            docs = loader.load()
            
            # Add metadata
            for doc in docs:
                doc.metadata.update({
                    "source_file": file_path.name,
                    "file_type": ext,
                    "file_path": str(file_path)
                })
            
            return docs
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []
    
    def _load_image_file(self, file_path: Path) -> List[Document]:
        """Extract text from image using OCR"""
        if not self.ocr_available:
            return []
        
        try:
            # Read image and extract text
            results = self.ocr_reader.readtext(str(file_path))
            
            # Combine all text
            text_content = ' '.join([result[1] for result in results if result[2] > 0.5])
            
            if text_content.strip():
                return [Document(
                    page_content=text_content,
                    metadata={
                        "source_file": file_path.name,
                        "file_type": file_path.suffix.lower(),
                        "extraction_method": "ocr"
                    }
                )]
            
        except Exception as e:
            logger.error(f"OCR extraction failed for {file_path}: {e}")
        
        return []
    
    def _create_chunks(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": f"chunk_{i}",
                "chunk_size": len(chunk.page_content)
            })
        
        return chunks
    
    def _create_rag_chain(self):
        """Create the RAG chain with multilingual prompt"""
        if not self.llm or not self.vector_store:
            logger.error("Cannot create RAG chain - LLM or vector store missing")
            return
        
        # Multilingual prompt template with strict language mirroring and clean formatting
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant that supports Indonesian and English.
            - Detect the language of the QUESTION and answer strictly in that language.
            - If the question is Indonesian, answer in natural Indonesian. If English, answer in English.
            - Be concise and structured: short paragraphs and bullet points where helpful.
            - Do not add closing phrases like "Do you have any other questions?".
            - Jawab hanya berdasarkan dokumen yang ada atau diberikan. Jika tidak ada informasi, jawab 'Tidak paham'.
            - If the QUESTION is in Indonesian, you MUST respond in Indonesian even if the CONTEXT is in another language.
            - Do NOT write meta phrases like "Here's the answer in English:". Respond directly in the required language.

            CONTEXT:
            {context}

            QUESTION:
            {question}

            ANSWER:"""
        )
        
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": int(os.getenv("RETRIEVER_K", "5"))}
            ),
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )
        
        logger.info("RAG chain created successfully")
    
    def process_uploaded_files(self, file_paths: List[Path]) -> Optional[FAISS]:
        """Process uploaded files temporarily"""
        try:
            documents = []
            for file_path in file_paths:
                docs = self._load_single_file(file_path)
                documents.extend(docs)
            
            if not documents:
                return None
            
            chunks = self._create_chunks(documents)
            temp_vector_store = FAISS.from_documents(chunks, self.embeddings)
            
            logger.info(f"Processed {len(chunks)} chunks from uploaded files")
            return temp_vector_store
            
        except Exception as e:
            logger.error(f"Error processing uploaded files: {e}")
            return None
    
    def ask_question(self, question: str, uploaded_vector_store: Optional[FAISS] = None) -> Dict[str, Any]:
        """Ask a question to the RAG system"""
        if not self.rag_chain:
            return {
                "answer": "System not properly initialized. Please contact support.",
                "sources": [],
                "error": "RAG chain not available"
            }
        
        try:
            def is_indonesian(text: str) -> bool:
                indowords = [
                    "yang", "dan", "atau", "tidak", "dengan", "untuk", "pada", "dari", "apa",
                    "bagaimana", "kapan", "dimana", "mengapa", "iya", "tidak", "bukan", "saya",
                    "kami", "kita", "anda", "kamu", "di", "ke", "ada", "adalah", "siapa",
                    "mengapa", "kapan", "dimana", "bagaimana", "apakah", "siapakah", "mengapa",
                    "kapan", "dimana", "bagaimana", "apa", "siapakah", "mengapa", "kapan",
                    "dimana", "bagaimana", "apa", "siapakah", "mengapa", "kapan", "dimana",
                    "bagaimana", "apa", "siapakah", "mengapa", "kapan", "dimana", "bagaimana",
                    "apa", "siapakah", "mengapa", "kapan", "dimana", "bagaimana", "apa",
                    "siapakah", "mengapa", "kapan", "dimana", "bagaimana", "apa", "siapakah"
                ]
                # Lower threshold and add question word bonus
                question_words = ["siapa", "apa", "mengapa", "kapan", "dimana", "bagaimana", "apakah", "siapakah"]
                score = sum(1 for w in indowords if f" {w} " in f" {text.lower()} ")
                question_bonus = sum(1 for w in question_words if text.lower().startswith(w))
                return score >= 1 or question_bonus >= 1

            def has_relevance(query: str, docs: list[Document]) -> bool:
                # Simple keyword overlap: require at least one keyword >=4 chars to appear in context
                import re
                tokens = [t for t in re.split(r"[^\w]+", query.lower()) if len(t) >= 4]
                if not tokens:
                    return True  # if no strong tokens, don't block
                context_text = " \n ".join([getattr(d, "page_content", "").lower() for d in docs])
                return any(tok in context_text for tok in tokens)

            is_id = is_indonesian(question)
            language_instruction = "Jawab dalam bahasa Indonesia." if is_id else "Answer in English."
            question_with_hint = f"{language_instruction}\n{question}"
            start_time = time.time()
            
            # Use uploaded files if available, otherwise use knowledge base
            if uploaded_vector_store:
                # Create temporary RAG chain with uploaded files
                temp_retriever = uploaded_vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": int(os.getenv("RETRIEVER_K", "5"))}
                )
                
                # Only use uploaded documents (do not mix with KB when uploads are present)
                uploaded_docs = temp_retriever.get_relevant_documents(question_with_hint)
                # Guard by relevance
                if not uploaded_docs or not has_relevance(question, uploaded_docs):
                    return {
                        "answer": "Tidak paham",
                        "sources": [],
                        "response_time": time.time() - start_time
                    }
                context = "\n\n".join([doc.page_content for doc in uploaded_docs])
                
                # Direct LLM call with combined context
                required_language = "Indonesian" if is_id else "English"
                prompt = f"""You are a helpful AI assistant that supports Indonesian and English.
                - Detect the language of the QUESTION and answer strictly in that language.
                - If the question is Indonesian, answer in natural Indonesian. If English, answer in English.
                - Be concise and structured: short paragraphs and bullet points where helpful.
                - Do not add closing phrases like 'Do you have any other questions?'.
                - Jawab hanya berdasarkan dokumen yang ada atau diberikan. Jika tidak ada informasi, jawab 'Tidak paham'.
                - If the QUESTION is in Indonesian, you MUST respond in Indonesian even if the CONTEXT is in another language.
                - Do NOT write meta phrases like "Here's the answer in English:". Respond directly in the required language.
                - REQUIRED ANSWER LANGUAGE: {required_language}

                CONTEXT:
                {context}

                QUESTION:
                {question_with_hint}

                ANSWER:"""
                
                response_text = self.llm.invoke(prompt)
                
                response = {
                    "result": response_text,
                    "source_documents": uploaded_docs
                }
            else:
                # Use standard KB retrieval manually for grounding + relevance check
                kb_retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": int(os.getenv("RETRIEVER_K", "5"))}
                )
                kb_docs = kb_retriever.get_relevant_documents(question_with_hint)
                if not kb_docs or not has_relevance(question, kb_docs):
                    return {
                        "answer": "Tidak paham",
                        "sources": [],
                        "response_time": time.time() - start_time
                    }
                context = "\n\n".join([doc.page_content for doc in kb_docs])

                required_language = "Indonesian" if is_id else "English"
                prompt = f"""You are a helpful AI assistant that supports Indonesian and English.
                - Detect the language of the QUESTION and answer strictly in that language.
                - If the question is Indonesian, answer in natural Indonesian. If English, answer in English.
                - Be concise and structured: short paragraphs and bullet points where helpful.
                - Do not add closing phrases like 'Do you have any other questions?'.
                - Jawab hanya berdasarkan dokumen yang ada atau diberikan. Jika tidak ada informasi, jawab 'Tidak paham'.
                - If the QUESTION is in Indonesian, you MUST respond in Indonesian even if the CONTEXT is in another language.
                - Do NOT write meta phrases like "Here's the answer in English:". Respond directly in the required language.
                - REQUIRED ANSWER LANGUAGE: {required_language}

                CONTEXT:
                {context}

                QUESTION:
                {question_with_hint}

                ANSWER:"""

                response_text = self.llm.invoke(prompt)
                response = {
                    "result": response_text,
                    "source_documents": kb_docs
                }

            end_time = time.time()

            # Enforce answer language post-processing if needed
            try:
                raw_answer = response.get("result") or response.get("answer") or ""
                if is_id and raw_answer:
                    # Force rewrite to Indonesian unconditionally when question is Indonesian
                    rewrite_prompt = f"""Ubah jawaban berikut agar sepenuhnya dalam bahasa Indonesia.
                    - Jangan menambah informasi baru.
                    - Pertahankan struktur poin jika ada.
                    - Jangan gunakan bahasa Inggris sama sekali.

                    Jawaban:
                    {raw_answer}
                    """
                    rewritten = self.llm.invoke(rewrite_prompt)
                    if isinstance(rewritten, str) and rewritten.strip():
                        response["result"] = rewritten.strip()
            except Exception as _:
                pass

            # Extract up to 3 unique sources (by file)
            sources = []
            if "source_documents" in response:
                seen_files = set()
                for doc in response["source_documents"]:
                    file_name = doc.metadata.get("source_file", "unknown")
                    if file_name in seen_files:
                        continue
                    seen_files.add(file_name)
                    sources.append({
                        "file": file_name,
                        "type": doc.metadata.get("file_type", "unknown"),
                        "preview": doc.page_content[:150] + "..."
                    })
                    if len(sources) >= 3:
                        break

            # Store conversation
            self.conversation_history.append({
                "question": question,
                "answer": response["result"],
                "timestamp": time.time()
            })

            # Normalize common encoding artifacts
            normalized_answer = response["result"].replace("Â£", "£")

            return {
                "answer": normalized_answer,
                "sources": sources,
                "response_time": end_time - start_time
            }
            
        except Exception as e:
            logger.error(f"Error in ask_question: {e}")
            return {
                "answer": "I apologize, but I encountered an error processing your question. Please try again or contact support.",
                "sources": [],
                "error": str(e)
            }

# Initialize RAG system
rag_system = MultiLanguageRAGSystem(os.getenv("KNOWLEDGE_BASE_PATH", "enterprise_knowledge_base"))

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "llm_available": rag_system.llm is not None,
        "vector_store_ready": rag_system.vector_store is not None,
        "ocr_available": rag_system.ocr_available
    })

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    """Answer question endpoint"""
    try:
        data = request.json
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        # Check for uploaded files in session or temp storage
        # For now, just use knowledge base
        result = rag_system.ask_question(question)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in ask endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/upload', methods=['POST'])
def upload_and_process():
    """Upload files and answer question"""
    try:
        question = request.form.get('question', '').strip()
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        if 'files' not in request.files:
            return jsonify({"error": "No files uploaded"}), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            # No files uploaded, use knowledge base only
            result = rag_system.ask_question(question)
            return jsonify(result)
        
        # Create temporary directory for uploaded files
        temp_dir = tempfile.mkdtemp()
        temp_file_paths = []
        
        try:
            # Save uploaded files temporarily
            for file in files:
                if file.filename and file.filename != '':
                    filename = secure_filename(file.filename)
                    file_ext = Path(filename).suffix.lower()
                    
                    if file_ext not in rag_system.supported_extensions:
                        continue
                    
                    file_path = Path(temp_dir) / filename
                    file.save(str(file_path))
                    temp_file_paths.append(file_path)
            
            if not temp_file_paths:
                return jsonify({"error": "No supported files uploaded"}), 400
            
            # Process uploaded files
            uploaded_vector_store = rag_system.process_uploaded_files(temp_file_paths)
            
            # Answer question using uploaded files
            result = rag_system.ask_question(question, uploaded_vector_store)
            
            return jsonify(result)
            
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        logger.error(f"Error in upload endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/sync', methods=['POST'])
def sync_knowledge_base():
    """Trigger Google Drive sync and KB reload."""
    try:
        chunks = rag_system.reload_knowledge_base()
        return jsonify({"status": "synced", "chunks": chunks})
    except Exception as e:
        logger.error(f"Error in sync endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/conversation')
def get_conversation():
    """Get conversation history"""
    return jsonify({
        "history": rag_system.conversation_history[-10:],  # Last 10 conversations
        "total": len(rag_system.conversation_history)
    })

if __name__ == '__main__':
    debug = os.getenv('FLASK_DEBUG', 'true').lower() == 'true'
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', '5000'))
    app.run(debug=debug, host=host, port=port)