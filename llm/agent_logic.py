import os
import sqlite3
import contextlib
import io
import logging
from typing import List, Optional, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field


# module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

class Task(BaseModel):
    """A model to represent a single task in the output format."""
    task: str = Field(description="The description of the specific task.")
    priority: str = Field(description="The priority level: High, Medium, or Low.")
    etr: str = Field(description="The stated estimated time remaining, e.g., '3 hours' or 'Not Stated'.")
    status_comments: str = Field(description="The status and comments, e.g., 'Status: In Progress (per OCR)'.")

class OutputFormat(BaseModel):
    """The structured output format for the response."""
    todays_focus: str = Field(description="1-2 sentence summary of the day’s primary goal, the most critical item, and the highest-priority blocker.")
    tasks: List[Task] = Field(description="A list of remaining tasks with their details.")
    
def get_prompt(file):
    with open(file, 'r') as f:
        prompt = f.read()
    # Escape curly braces to prevent parsing errors in f-string format
    prompt = prompt.replace("{", "{{").replace("}", "}}")
    return prompt

def get_llm(model_name: str, base_url: Optional[str] = None, api_key: Optional[str] = None):
    """Dynamically loads and returns the appropriate LLM based on the model name."""
    if model_name.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        if api_key is None:
            raise ValueError("API key is required for Gemini models.")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    else:
        # Assume local model via Ollama
        from langchain_ollama import OllamaLLM as Ollama
        if base_url is None:
            base_url = "http://host.docker.internal:11434"  # Default base URL
        return Ollama(model=model_name, base_url=base_url)

def get_query_generator_chain(model_name: str, base_url: Optional[str] = None, api_key: Optional[str] = None):
    """Builds the chain for Agent 1 (Query Generation)."""
    
    llm = get_llm(model_name, base_url=base_url, api_key=api_key)

    parser = PydanticOutputParser(pydantic_object=OutputFormat)

    system_prompt_1 = get_prompt("llm/assets/system_instructions.md")

    format_instructions = parser.get_format_instructions()
    # Escape curly braces in format_instructions
    format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")

    # Append format instructions to the system prompt
    system_prompt_1 += f"\n\n{format_instructions}"

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt_1),
        ("human", "Data Context:\n{context}\n\nUser Question:\n{question}")
    ])
    
    return prompt_template | llm | parser