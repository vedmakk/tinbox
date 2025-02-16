# Tinbox: A CLI Translation Tool

## Overview

**Tinbox** is a command-line tool designed to translate documents using various **Large Language Models (LLMs)**, including both text-only and multimodal models. For PDFs, it leverages multimodal models (like GPT-4V or Claude 3 Sonnet) to directly process page images without OCR. For text-based formats (Word, TXT), it uses text-to-text translation. The tool supports multiple input/output formats and includes two primary translation algorithms—**Page-by-Page with Seam Repair** and **Sliding Window**—with benchmarking capabilities for time and token usage/cost.

---

## Features

1. **Multiple Input Formats**  
   - PDF (processed as images for multimodal models)
   - Word (docx) and TXT (processed as text)
2. **Multiple Output Formats**  
   - Defaults to stdout or `.txt` file
   - Future extensibility for PDF or Word output
3. **Two Translation Algorithms**  
   - **Page-by-Page + Seam Repair** (default for PDF/Word)
   - **Sliding Window** (recommended for long TXT or user-specified)
4. **Model Flexibility**  
   - Multimodal models (GPT-4V, Claude 3 Sonnet) for PDF processing
   - Text-based models for Word/TXT processing
   - Support for both local and cloud providers
5. **Language Support**
   - Flexible source/target language specification using ISO 639-1 codes
   - Common language aliases (e.g., 'en', 'zh', 'es')
6. **Benchmarking**  
   - Track overall translation time and token usage/cost
   - Compare algorithms or model providers side-by-side

---

## Installation & Dependencies

1. **Python 3.9+** (recommended)
2. **Model Dependencies**:
   - `litellm` for unified model interface
   - Local model support via Ollama
   - Appropriate API keys for cloud models
3. **Document Processing**:
   - `pdf2image` for PDF-to-image conversion
   - `python-docx` for `.docx` extraction
4. **(Optional) Additional Tools** for advanced exporting

Installation:

```bash
pip install tinbox

# Optional: Install additional modules for PDF/Word handling
pip install pdf2image python-docx
```

---

## Command-Line Usage

Basic syntax:

```bash
tinbox [OPTIONS] <input-file>
```

### 1. Global Options

| Option                          | Description                                                                                           |
|--------------------------------|-------------------------------------------------------------------------------------------------------|
| `--from, -f <LANG>`           | Source language code (e.g., 'en', 'zh', 'es', 'de'). Defaults to auto-detect.                        |
| `--to, -t <LANG>`             | Target language code. Required.                                                                        |
| `--algorithm, -a <ALGO>`       | Translation algorithm. One of `page`, `sliding-window`. Defaults to `page`.                          |
| `--model <MODEL_NAME>`         | Model/provider (e.g., `gpt4v`, `claude3`, `olama:local-model`).                                      |
| `--output, -o <OUTPUT_FILE>`   | Output file path. If not provided, writes to stdout.                                                 |
| `--benchmark, -b`              | Enable benchmarking mode.                                                                            |

### 2. Supported Language Codes

Common language codes (ISO 639-1):

| Code | Language    | Also Accepts |
|------|-------------|--------------|
| en   | English     | eng          |
| es   | Spanish     | spa          |
| zh   | Chinese     | chi, cmn     |
| fr   | French      | fra          |
| de   | German      | deu, ger     |
| ja   | Japanese    | jpn          |
| ko   | Korean      | kor          |
| ru   | Russian     | rus          |
| ar   | Arabic      | ara          |
| hi   | Hindi       | hin          |

### 3. Algorithm-Specific Options

#### A. Page-by-Page + Seam Repair
(Default for all file types)

| Option                          | Description                                                                                    |
|--------------------------------|------------------------------------------------------------------------------------------------|
| `--page-seam-overlap N`        | Token overlap for seam repair. Default `200` tokens or `25%`.                                 |
| `--max-page-context <TOKENS>`  | Max tokens per chunk. Default `2000`.                                                         |
| `--repair-model <MODEL_NAME>`  | Model for seam repair (can differ from main model).                                           |

**Process Flow**:
1. **Document Processing**:
   - PDF: Convert pages to images
   - Word/TXT: Extract text page-by-page
2. **Translation**:
   - PDF: Send page images to multimodal model
   - Text: Send text chunks to text model
3. **Seam Repair**: Overlap processing for continuity
4. **Assembly**: Concatenate results

#### B. Sliding Window Options

| Option                          | Description                                                                                    |
|--------------------------------|------------------------------------------------------------------------------------------------|
| `--window-size <TOKENS>`       | Size of each translation window. Default `2000`.                                              |
| `--overlap-size <TOKENS>`      | Overlap between windows. Default `200`.                                                       |
| `--split-level <LEVEL>`        | Split text by 'paragraph' or 'sentence'. Default 'paragraph'.                                 |

---

## Example Commands

1. **Basic Translation (English to Spanish)**  
   ```bash
   tinbox --to es document.pdf
   ```
   - Uses default model (e.g., `gpt4v` for PDF)
   - Auto-detects source language
   - Outputs to stdout

2. **Specify Source and Target Languages**  
   ```bash
   tinbox --from zh --to en --model claude3 document.docx
   ```

3. **Use Sliding Window with Custom Parameters**  
   ```bash
   tinbox --from ja --to ko --algorithm sliding-window --window-size 3000 large_text.txt
   ```

4. **Run Benchmarks with Different Models**  
   ```bash
   tinbox --from de --to fr --benchmark --model gpt4v sample.pdf
   tinbox --from de --to fr --benchmark --model claude3 sample.pdf
   ```

## Implementation Details

### 1. Core Classes and Types

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union, Literal
from pathlib import Path
import asyncio

class FileType(Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"

class ModelType(Enum):
    GPT4V = "gpt4v"
    CLAUDE3 = "claude3"
    OLAMA = "olama"

@dataclass
class TranslationConfig:
    source_lang: str
    target_lang: str
    model: ModelType
    algorithm: Literal["page", "sliding-window"]
    input_file: Path
    output_file: Optional[Path]
    benchmark: bool = False
    
    # Algorithm-specific settings
    page_seam_overlap: int = 200
    window_size: int = 2000
    overlap_size: int = 200

@dataclass
class TranslationResult:
    text: str
    tokens_used: int
    cost: float
    time_taken: float
```

### 2. Document Processing

```python
class DocumentProcessor:
    async def load_document(self, file_path: Path) -> Union[List[bytes], List[str]]:
        """Load document and return either list of page images (PDF) or text chunks."""
        file_type = self._get_file_type(file_path)
        
        if file_type == FileType.PDF:
            return await self._process_pdf(file_path)
        elif file_type == FileType.DOCX:
            return await self._process_docx(file_path)
        else:
            return await self._process_txt(file_path)

    async def _process_pdf(self, file_path: Path) -> List[bytes]:
        """Convert PDF pages to images."""
        from pdf2image import convert_from_path
        
        images = []
        pages = convert_from_path(file_path)
        for page in pages:
            with io.BytesIO() as bio:
                page.save(bio, format='PNG')
                images.append(bio.getvalue())
        return images

    async def _process_docx(self, file_path: Path) -> List[str]:
        """Extract text from Word document, preserving page breaks."""
        from docx import Document
        
        doc = Document(file_path)
        pages = []
        current_page = []
        
        for para in doc.paragraphs:
            if self._is_page_break(para):
                pages.append('\n'.join(current_page))
                current_page = []
            else:
                current_page.append(para.text)
                
        if current_page:
            pages.append('\n'.join(current_page))
        return pages
```

### 3. Translation Algorithms

#### A. Page-by-Page Translation

```python
class PageTranslator:
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.model = self._initialize_model()

    async def translate(self, pages: Union[List[bytes], List[str]]) -> TranslationResult:
        start_time = time.time()
        total_tokens = 0
        translated_pages = []

        # Translate each page
        for i, page in enumerate(pages):
            if isinstance(page, bytes):
                # Image-based translation for PDFs
                translation = await self._translate_image(
                    image=page,
                    source_lang=self.config.source_lang,
                    target_lang=self.config.target_lang
                )
            else:
                # Text-based translation
                translation = await self._translate_text(
                    text=page,
                    source_lang=self.config.source_lang,
                    target_lang=self.config.target_lang
                )
            
            translated_pages.append(translation.text)
            total_tokens += translation.tokens_used

        # Perform seam repair
        final_text = await self._repair_seams(translated_pages)
        
        return TranslationResult(
            text=final_text,
            tokens_used=total_tokens,
            cost=self._calculate_cost(total_tokens),
            time_taken=time.time() - start_time
        )

    async def _repair_seams(self, translated_pages: List[str]) -> str:
        """Repair page boundaries for better continuity."""
        final_chunks = []
        
        for i in range(len(translated_pages) - 1):
            # Extract overlap regions
            current_page_end = translated_pages[i][-self.config.page_seam_overlap:]
            next_page_start = translated_pages[i+1][:self.config.page_seam_overlap]
            
            # Request model to smooth the transition
            repaired_seam = await self._smooth_transition(
                current_page_end,
                next_page_start,
                self.config.target_lang
            )
            
            # Replace original text with repaired version
            if i == 0:
                final_chunks.append(translated_pages[i][:-self.config.page_seam_overlap])
            final_chunks.append(repaired_seam)
            
            if i == len(translated_pages) - 2:
                final_chunks.append(translated_pages[i+1][self.config.page_seam_overlap:])
                
        return ''.join(final_chunks)
```

#### B. Sliding Window Translation

```python
class SlidingWindowTranslator:
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.model = self._initialize_model()

    async def translate(self, text: str) -> TranslationResult:
        start_time = time.time()
        total_tokens = 0
        
        # Create overlapping chunks
        chunks = self._create_chunks(
            text,
            window_size=self.config.window_size,
            overlap_size=self.config.overlap_size
        )
        
        # Translate chunks
        translated_chunks = []
        for chunk in chunks:
            translation = await self._translate_chunk(
                text=chunk,
                source_lang=self.config.source_lang,
                target_lang=self.config.target_lang
            )
            translated_chunks.append(translation.text)
            total_tokens += translation.tokens_used
            
        # Merge overlapping translations
        final_text = self._merge_chunks(translated_chunks)
        
        return TranslationResult(
            text=final_text,
            tokens_used=total_tokens,
            cost=self._calculate_cost(total_tokens),
            time_taken=time.time() - start_time
        )

    def _create_chunks(self, text: str, window_size: int, overlap_size: int) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + window_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap_size
            
        return chunks

    def _merge_chunks(self, chunks: List[str]) -> str:
        """Merge translated chunks, handling overlaps."""
        if not chunks:
            return ""
            
        result = [chunks[0]]
        overlap_size = self.config.overlap_size
        
        for i in range(1, len(chunks)):
            current_chunk = chunks[i]
            previous_chunk = chunks[i-1]
            
            # Find best matching point in overlap region
            match_point = self._find_best_merge_point(
                previous_chunk[-overlap_size:],
                current_chunk[:overlap_size]
            )
            
            # Append non-overlapping portion
            result.append(current_chunk[match_point:])
            
        return ''.join(result)
```

### 4. Model Interface using LiteLLM

```python
from litellm import completion
import base64

class ModelInterface:
    """Unified interface for all models using LiteLLM."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    async def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> TranslationResult:
        """Translate text using any text-to-text model."""
        prompt = f"Translate the following text from {source_lang} to {target_lang}. Maintain all formatting and structure:\n\n{text}"
        
        response = await completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3  # Lower temperature for more consistent translations
        )
        
        return TranslationResult(
            text=response.choices[0].message.content,
            tokens_used=response.usage.total_tokens,
            cost=response.usage.cost,  # LiteLLM provides cost directly
            time_taken=response.usage.completion_time  # LiteLLM provides latency
        )

    async def translate_image(
        self,
        image: bytes,
        source_lang: str,
        target_lang: str
    ) -> TranslationResult:
        """Translate content from image using multimodal models."""
        prompt = f"Translate this page from {source_lang} to {target_lang}. Maintain all formatting and structure."
        
        # Prepare image for multimodal models
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64.b64encode(image).decode()}"
            }
        }
        
        response = await completion(
            model=self.model_name,  # Will work with GPT-4V, Claude 3, etc.
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_content
                ]
            }],
            temperature=0.3
        )
        
        return TranslationResult(
            text=response.choices[0].message.content,
            tokens_used=response.usage.total_tokens,
            cost=response.usage.cost,
            time_taken=response.usage.completion_time
        )
```

## Project Structure

```
tinbox/
├── __init__.py
├── cli.py              # Command-line interface
├── config.py           # Configuration handling
├── core/
│   ├── __init__.py
│   ├── types.py       # Core data types
│   ├── processor.py   # Document processing
│   └── translator.py  # Translation algorithms
├── models/
│   ├── __init__.py
│   └── interface.py   # LiteLLM-based model interface
└── utils/
    ├── __init__.py
    ├── language.py    # Language code handling
    └── benchmark.py   # Benchmarking utilities
```

## Future Extensions

1. **Additional Formats**  
   - Output to PDF, Word, or HTML
   - Possibly maintain original layout (requires advanced tooling)
2. **Better Section/Paragraph Detection**  
   - Use AI-based or custom logic to segment large documents into semantically coherent blocks
3. **Shared Terminology / Glossary**  
   - Option to pass a domain glossary or term dictionary for consistency
4. **Human-In-The-Loop**  
   - Provide interactive mode for users to correct partial translations before final output
5. **Model Fine-Tuning**  
   - Connect to specialized or fine-tuned local models for domain-specific documents
