"""
Unified service class that integrates Universal Extractor and Nexy-Rep functionalities.
"""
import os
from datetime import datetime
from typing import Optional, Dict, Any
import logging

# Import Universal Extractor classes
from .extractors.pdf_extractor import PDFExtractor
from .extractors.docx_extractor import DocxExtractor
from .extractors.csv_extractor import CSVExtractor
from .extractors.txt_extractor import TXTExtractor
from .extractors.json_extractor import JSONExtractor
from .extractors.yaml_extractor import YAMLExtractor
from .extractors.toml_extractor import TOMLExtractor
from .extractors.markdown_extractor import MarkdownExtractor

# Import Nexy-Rep configuration (lazy-import other heavy modules at runtime)
from .nexy_rep.config import Config
from .llm.agent_logic import get_query_generator_chain, get_llm, get_prompt
from .github_activity import GitHubUserActivity


class UnifiedService:
    """
    A unified service class that provides access to both document extraction
    and image processing/OCR capabilities.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the unified service.
        
        Args:
            config_path (str, optional): Path to the Nexy-Rep configuration file.
                If not provided, default configuration will be used.
        """
        # Initialize Nexy-Rep configuration
        # Config currently does not accept a path; always instantiate and attach provided path for downstream use.
        self.config = Config()
        if config_path:
            # Attach for consumers that might expect it
            setattr(self.config, "user_config_path", config_path)
        # initialize DB (lazy import to avoid heavy deps during module import)
        try:
            from .nexy_rep.storage import init_db
            init_db(self.config.db_path)
        except Exception:
            # If storage/init_db can't be imported at module import time, defer until runtime.
            logging.debug("nexy_rep.storage.init_db not available at import time; will initialize on first store")
        
        # File type to extractor mapping
        self.extractors = {
            '.pdf': PDFExtractor,
            '.docx': DocxExtractor,
            '.csv': CSVExtractor,
            '.txt': TXTExtractor,
            '.json': JSONExtractor,
            '.yaml': YAMLExtractor,
            '.yml': YAMLExtractor,
            '.toml': TOMLExtractor,
            '.md': MarkdownExtractor
        }
        
        # Keep track of last processed image for similarity comparison
        self._last_embedding = None
    
    def extract_from_file(self, file_path: str) -> str:
        """
        Extract text from a document file.
        
        Args:
            file_path (str): Path to the document file.
        
        Returns:
            str: Extracted text from the document.
            
        Raises:
            ValueError: If file type is not supported.
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self.extractors:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        extractor = self.extractors[file_ext](file_path)
        return extractor.extract_text()
    
    def capture_and_process_screen(self, store: bool = True) -> Dict[str, Any]:
        """
        Capture a screenshot, process it with OCR, and optionally store it.
        
        Args:
            store (bool): Whether to store the processed data. Defaults to True.
        
        Returns:
            dict: Dictionary containing the processed data:
                {
                    'timestamp': datetime object,
                    'image_path': str (path to stored image) or None,
                    'text': str (extracted text),
                    'similarity': float (similarity with previous image)
                }
        """
        # Generate timestamp and paths
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        temp_image_path = os.path.join(self.config.temp_dir, f"temp_{timestamp_str}.png")
        
        # Capture screenshot (lazy import to avoid pyautogui at module import)
        try:
            from .nexy_rep.capture import take_screenshot
            take_screenshot(temp_image_path)
        except Exception:
            # If screenshot capture isn't available, record and return an error-like result
            logging.exception("Screenshot capture not available")
            return {"timestamp": timestamp, "image_path": None, "text": "", "similarity": 0.0, "error": "capture_unavailable"}

        # Extract text using OCR (lazy import)
        try:
            from .nexy_rep.ocr import extract_text_from_image
            text = extract_text_from_image(temp_image_path)
        except Exception:
            logging.exception("OCR not available")
            # Clean up temp image if present
            try:
                os.remove(temp_image_path)
            except Exception:
                pass
            return {"timestamp": timestamp, "image_path": None, "text": "", "similarity": 0.0, "error": "ocr_unavailable"}
        
        result = {
            'timestamp': timestamp,
            'image_path': None,
            'text': text,
            'similarity': 0.0
        }
        
        if not text.strip():
            os.remove(temp_image_path)
            return result
        
        # Get embedding and compute similarity (lazy import)
        try:
            from .nexy_rep.embed import get_embedding
            from .nexy_rep.compare import compute_similarity
            embedding = get_embedding(text)
            if self._last_embedding is not None:
                result['similarity'] = compute_similarity(embedding, self._last_embedding)
        except Exception:
            logging.exception("Embedding/compare not available")
            # If embeddings aren't available, return with zero similarity
            embedding = None
        
        if store and (self._last_embedding is None or 
                     result['similarity'] < self.config.similarity_threshold):
            # Store the image and data
            permanent_image_path = os.path.join(
                self.config.images_dir,
                f"image_{timestamp_str}.png"
            )
            os.rename(temp_image_path, permanent_image_path)
            
            try:
                from .nexy_rep.storage import store_data
                store_data(
                    self.config.db_path,
                    timestamp=timestamp,
                    image_path=permanent_image_path,
                    extracted_text=text
                )
            except Exception:
                logging.exception("Failed to store data to nexy_rep.storage")
            
            result['image_path'] = permanent_image_path
            self._last_embedding = embedding
        else:
            os.remove(temp_image_path)
        
        return result
    
    def process_image(self, image_path: str, store: bool = True) -> Dict[str, Any]:
        """
        Process an existing image file with OCR and optionally store it.
        
        Args:
            image_path (str): Path to the image file.
            store (bool): Whether to store the processed data. Defaults to True.
            
        Returns:
            dict: Dictionary containing the processed data:
                {
                    'timestamp': datetime object,
                    'image_path': str (path to stored image) or None,
                    'text': str (extracted text),
                    'similarity': float (similarity with previous image)
                }
        """
        timestamp = datetime.now()
        
        # Extract text using OCR (lazy import)
        try:
            from .nexy_rep.ocr import extract_text_from_image
            text = extract_text_from_image(image_path)
        except Exception:
            logging.exception("OCR not available for process_image")
            return {"timestamp": timestamp, "image_path": None, "text": "", "similarity": 0.0, "error": "ocr_unavailable"}
        
        result = {
            'timestamp': timestamp,
            'image_path': None,
            'text': text,
            'similarity': 0.0
        }
        
        if not text.strip():
            return result
        
        # Get embedding and compute similarity (lazy import)
        try:
            from .nexy_rep.embed import get_embedding
            from .nexy_rep.compare import compute_similarity
            embedding = get_embedding(text)
            if self._last_embedding is not None:
                result['similarity'] = compute_similarity(embedding, self._last_embedding)
        except Exception:
            logging.exception("Embedding/compare not available for process_image")
            embedding = None
        
        if store and (self._last_embedding is None or 
                     result['similarity'] < self.config.similarity_threshold):
            # Store the data
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            permanent_image_path = os.path.join(
                self.config.images_dir,
                f"image_{timestamp_str}.png"
            )
            
            # Copy the image to permanent storage
            import shutil
            shutil.copy2(image_path, permanent_image_path)
            
            try:
                from .nexy_rep.storage import store_data
                store_data(
                    self.config.db_path,
                    timestamp=timestamp,
                    image_path=permanent_image_path,
                    extracted_text=text
                )
            except Exception:
                logging.exception("Failed to store data to nexy_rep.storage in process_image")
            
            result['image_path'] = permanent_image_path
            self._last_embedding = embedding
        
        return result

    def run_agentic_query(
        self,
        context: str,
        question: str,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Any:
        """
        Run the agentic LLM chain on provided context and question.

        This wraps the existing `get_query_generator_chain` from
        `services/llm/agent_logic.py` and invokes the chain with the
        given inputs. Returns whatever the chain returns (typically a
        parsed Pydantic object or a dict-like result).

        Args:
            context: The textual context to provide to the agent.
            question: The human question or instruction for the agent.
            model_name: Optional model identifier (falls back to Ollama/local behavior if not provided).
            base_url: Optional base URL for remote LLM endpoints (used by some adapters).
            api_key: Optional API key for hosted models (e.g., Gemini).

        Returns:
            Any: The chain invocation result. On error, returns a dict with an 'error' key.
        """
        # Choose a reasonable default model if none provided. We prefer not to force a hosted model here.
        model_to_use = model_name or os.environ.get("NEXA_DEFAULT_MODEL")

        try:
            chain = get_query_generator_chain(model_name=model_to_use or "ollama", base_url=base_url, api_key=api_key)
            # The chain API in this project uses .invoke with a dict carrying context and question
            res = chain.invoke({"context": context, "question": question})
            return res
        except Exception as exc:  # pragma: no cover - surface runtime errors
            # Keep failure mode explicit for callers
            logging.exception("Agentic query failed")
            return {"error": str(exc)}

    def list_prompts(self) -> list:
        """Return available prompt files in `services/llm/assets`.

        Returns a list of absolute file paths. This makes prompt files
        discoverable so callers can pass them to `chat` as the system prompt.
        """
        assets_dir = os.path.join(os.path.dirname(__file__), "llm", "assets")
        prompts = []
        if os.path.isdir(assets_dir):
            for fname in os.listdir(assets_dir):
                if fname.lower().endswith(('.md', '.txt', '.prompt')):
                    prompts.append(os.path.join(assets_dir, fname))
        return sorted(prompts)

    def chat(
        self,
        session_id: str,
        user_message: str,
        system_prompt_file: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        history_limit: int = 20,
    ) -> str:
        """
        Chat entrypoint that manages session state and returns plain-text assistant reply.

        Behavior:
          - If `session_id` has no history in DB, the function starts a new session.
          - Stores the user message and the assistant reply into the DB.
          - On subsequent calls for the same `session_id`, the previous conversation
            is included as context (up to `history_limit` messages).

        Args:
            session_id: Unique session identifier (string).
            user_message: The user's current message.
            system_prompt_file: Path to a system prompt file to use as the system message. If not
                provided, defaults to `services/llm/assets/system_instructions.md`.
            model_name, base_url, api_key: Optional LLM configuration.
            history_limit: How many previous messages to include in the context (most recent first).

        Returns:
            Plain text assistant response.
        """
        # Lazy import storage helpers
        try:
            from .nexy_rep.storage import ensure_conversation_table, store_chat_message, get_chat_history
        except Exception:
            logging.exception("Conversation storage helpers not available")
            raise

        # Ensure conversation table exists
        ensure_conversation_table(self.config.db_path)

        # Prepare system prompt
        if system_prompt_file is None:
            system_prompt_file = os.path.join(os.path.dirname(__file__), "llm", "assets", "system_instructions.md")
        if not os.path.isfile(system_prompt_file):
            # fallback: empty system prompt
            system_prompt = ""
        else:
            system_prompt = get_prompt(system_prompt_file)

        # Fetch prior history and build a plain-text history string
        rows = get_chat_history(self.config.db_path, session_id, limit=history_limit)
        history_lines = []
        for ts, role, msg in rows:
            # role is expected to be 'user' or 'assistant'
            prefix = "User:" if role.lower().startswith('user') else "Assistant:"
            history_lines.append(f"{prefix} {msg}")
        history_text = "\n".join(history_lines)

        # Build prompt template and invoke LLM
        # Use provided model or env/default
        model_to_use = model_name or os.environ.get("NEXA_DEFAULT_MODEL", "ollama")
        llm = get_llm(model_to_use, base_url=base_url, api_key=api_key)

        # Construct a simple chat-style prompt
        from langchain_core.prompts import ChatPromptTemplate

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Conversation so far:\n{history}\n\nUser: {message}\nAssistant:")
        ])

        chain = prompt_template | llm

        try:
            invocation = {"history": history_text, "message": user_message}
            res = chain.invoke(invocation)
            # Normalize the result to a plain string
            if isinstance(res, str):
                assistant_text = res
            else:
                # Some LLM wrappers return an object; attempt common attributes
                assistant_text = getattr(res, "text", None) or getattr(res, "content", None) or str(res)
        except Exception as exc:  # pragma: no cover
            logging.exception("LLM invocation failed")
            assistant_text = f"Error: {exc}"

        # Persist the user message and assistant reply
        try:
            store_chat_message(self.config.db_path, session_id, "user", user_message)
            store_chat_message(self.config.db_path, session_id, "assistant", assistant_text)
        except Exception:
            logging.exception("Failed to persist chat messages")

        # Return plain text only
        return assistant_text

    # ---------------------------------------------------------------
    # GitHub Activity Integration
    # ---------------------------------------------------------------
    def fetch_github_activity(
        self,
        username: str,
        start_date: str,
        end_date: str,
        token: Optional[str] = None,
        repos: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Fetch GitHub activity for a user in the given date range.

        Args:
            username: GitHub username
            start_date: Start date in 'YYYY-MM-DD'
            end_date: End date in 'YYYY-MM-DD'
            token: Optional GitHub token (falls back to GITHUB_TOKEN env var)
            repos: Optional list of repository names to filter

        Returns:
            dict: Summary returned by GitHubUserActivity.get_user_activity()
        """
        tracker = GitHubUserActivity(
            username=username,
            start_date=start_date,
            end_date=end_date,
            token=token,
            repos=repos,
        )

        return tracker.get_user_activity()
