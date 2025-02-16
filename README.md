# Tinbox: A CLI Translation Tool

## Overview

**Tinbox** is a command-line tool designed to translate documents using various **Large Language Models (LLMs)**, including both text-only and multimodal models. For PDFs, it leverages multimodal models (like GPT-4o or Claude 3.5 Sonnet) to directly process page images without OCR. For text-based formats (Word, TXT), it uses text-to-text translation. The tool supports multiple input/output formats and includes two primary translation algorithms—**Page-by-Page with Seam Repair** and **Sliding Window**—with benchmarking capabilities for time and token usage/cost.

---

## Features

1. **Multiple Input Formats**  
   - PDF (processed as images for multimodal models)
   - Word (docx) and TXT (processed as text)
2. **Multiple Output Formats**  
   - Defaults to stdout or `.txt` file
   - Future extensibility for PDF or Word output
3. **Two Translation Algorithms**  
   - **Page-by-Page + Seam Repair** (default for PDF)
   - **Sliding Window** (recommended for long TXT or user-specified)
4. **Model Flexibility**  
   - Multimodal models (GPT-4o, Claude 3.5 Sonnet) for PDF processing
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
pip install tinbox[pdf,docx]
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
| `--to, -t <LANG>`             | Target language code. Defaults to 'en' (English).                                                      |
| `--algorithm, -a <ALGO>`       | Translation algorithm. One of `page`, `sliding-window`. Defaults to `page`.                          |
| `--model <MODEL_NAME>`         | Model/provider (e.g., `gpt-4o`, `claude-3-5-sonnet`, `olama:mistral-small`).                                      |
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

#### B. Sliding Window Options

| Option                          | Description                                                                                    |
|--------------------------------|------------------------------------------------------------------------------------------------|
| `--window-size <TOKENS>`       | Size of each translation window. Default `2000`.                                              |
| `--overlap-size <TOKENS>`      | Overlap between windows. Default `200`.                                                       |
| `--split-level <LEVEL>`        | Split text by 'paragraph' or 'sentence'. Default 'paragraph'.                                 |

### 4. Output Formats

Tinbox supports multiple output formats to suit different needs:

#### A. Plain Text (Default)
Just the translated text, suitable for direct use:
```bash
tinbox translate document.pdf --to es
# Output: The translated text...
```

#### B. JSON
Structured output with metadata, statistics, and results:
```bash
tinbox translate document.pdf --to es --format json
```

Example JSON output:
```json
{
  "metadata": {
    "source_lang": "en",
    "target_lang": "es",
    "model": "claude-3-sonnet",
    "algorithm": "page",
    "input_file": "document.pdf",
    "input_file_type": "pdf",
    "timestamp": "2024-03-21T14:30:00"
  },
  "result": {
    "text": "El texto traducido...",
    "tokens_used": 1500,
    "cost": 0.045,
    "time_taken": 12.5
  },
  "warnings": [
    "Large document detected"
  ],
  "errors": []
}
```

#### C. Markdown
Human-readable report with all details:
```bash
tinbox translate document.pdf --to es --format markdown
```

Example Markdown output:
```markdown
# Translation Results

## Metadata
- Source Language: en
- Target Language: es
- Model: claude-3-sonnet
- Algorithm: page
- Input File: document.pdf
- File Type: pdf
- Timestamp: 2024-03-21T14:30:00

## Translation
```text
El texto traducido...
```

## Statistics
- Tokens Used: 1,500
- Cost: $0.0450
- Time Taken: 12.5s

## Warnings
- Large document detected

## Errors
[None]
```

The output format can be specified with the `--format` option:
- `--format text` (default): Just the translated text
- `--format json`: Structured JSON output
- `--format markdown`: Human-readable report

Use with `--output` to save to a file:
```bash
# Save as JSON
tinbox translate document.pdf --to es --format json --output translation.json

# Save as Markdown report
tinbox translate document.pdf --to es --format markdown --output report.md
```

**Process Flow**:
1. **Document Processing**:
   - PDF: Convert pages to images
   - Word/TXT: Extract text page-by-page
2. **Translation**:
   - PDF: Send page images to multimodal model
   - Text: Send text chunks to text model
3. **Seam Repair**: Overlap processing for continuity
4. **Assembly**: Concatenate results

---

## Example Commands

1. **Basic Translation (English to Spanish)**  
   ```bash
   tinbox --to es document.pdf
   ```
   - Uses default model (e.g., `gpt-4o` for PDF)
   - Auto-detects source language
   - Outputs to stdout

2. **Specify Source and Target Languages**  
   ```bash
   tinbox --from zh --to en --model claude-3.5-latest document.docx
   ```

3. **Use Sliding Window with Custom Parameters**  
   ```bash
   tinbox --from ja --to ko --algorithm sliding-window --window-size 3000 large_text.txt
   ```

4. **Run Benchmarks with Different Models**  
   ```bash
   tinbox --from de --to fr --benchmark --model gpt-4o sample.pdf
   tinbox --from de --to fr --benchmark --model claude-3.5-latest sample.pdf
   ```

## Implementation Details

### 1. Core Types and Models

```python
from enum import Enum
from typing import List, Optional, Union, Literal
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
import asyncio

class FileType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"

class ModelType(str, Enum):
    gpt-4o = "gpt-4o"
    claude-3.5-latest = "claude-3.5-latest"
    OLAMA = "olama"

class TranslationConfig(BaseModel):
    source_lang: str
    target_lang: str
    model: ModelType
    algorithm: Literal["page", "sliding-window"]
    input_file: Path
    output_file: Optional[Path] = None
    benchmark: bool = False
    
    # Algorithm-specific settings
    page_seam_overlap: int = Field(default=200, gt=0)
    window_size: int = Field(default=2000, gt=0)
    overlap_size: int = Field(default=200, gt=0)

    model_config = ConfigDict(frozen=True)  # Make config immutable

class TranslationResult(BaseModel):
    text: str
    tokens_used: int = Field(ge=0)
    cost: float = Field(ge=0.0)
    time_taken: float = Field(ge=0.0)

    model_config = ConfigDict(frozen=True)
```

### 2. Document Processing

```python
from typing import Protocol
import io
from pathlib import Path

class DocumentContent(BaseModel):
    """Represents processed document content"""
    pages: List[Union[bytes, str]]
    file_type: FileType
    metadata: dict = Field(default_factory=dict)

async def load_document(file_path: Path) -> DocumentContent:
    """Load document and return either list of page images (PDF) or text chunks."""
    file_type = _get_file_type(file_path)
    
    processors = {
        FileType.PDF: _process_pdf,
        FileType.DOCX: _process_docx,
        FileType.TXT: _process_txt
    }
    
    processor = processors.get(file_type)
    if not processor:
        raise ValueError(f"Unsupported file type: {file_type}")
        
    pages = await processor(file_path)
    return DocumentContent(pages=pages, file_type=file_type)

async def _process_pdf(file_path: Path) -> List[bytes]:
    """Convert PDF pages to images."""
    from pdf2image import convert_from_path
    
    images = []
    pages = convert_from_path(file_path)
    for page in pages:
        with io.BytesIO() as bio:
            page.save(bio, format='PNG')
            images.append(bio.getvalue())
    return images

async def _process_docx(file_path: Path) -> List[str]:
    """Extract text from Word document, preserving page breaks."""
    from docx import Document
    
    doc = Document(file_path)
    pages = []
    current_page = []
    
    for para in doc.paragraphs:
        if _is_page_break(para):
            pages.append('\n'.join(current_page))
            current_page = []
        else:
            current_page.append(para.text)
            
    if current_page:
        pages.append('\n'.join(current_page))
    return pages
```

### 3. Translation Functions

```python
async def translate_document(
    content: DocumentContent,
    config: TranslationConfig
) -> TranslationResult:
    """Main translation function that delegates to appropriate algorithm"""
    if config.algorithm == "page":
        return await translate_page_by_page(content, config)
    else:
        return await translate_sliding_window(content, config)

async def translate_page_by_page(
    content: DocumentContent,
    config: TranslationConfig
) -> TranslationResult:
    """Page-by-page translation with seam repair"""
    start_time = time.time()
    total_tokens = 0
    translated_pages = []
    
    model = initialize_model(config.model)
    
    # Translate each page
    for page in content.pages:
        translation = await (
            translate_image(page, config)
            if isinstance(page, bytes)
            else translate_text(page, config)
        )
        
        translated_pages.append(translation.text)
        total_tokens += translation.tokens_used
    
    # Perform seam repair
    final_text = await repair_seams(translated_pages, config)
    
    return TranslationResult(
        text=final_text,
        tokens_used=total_tokens,
        cost=calculate_cost(total_tokens, config.model),
        time_taken=time.time() - start_time
    )

async def translate_sliding_window(
    content: DocumentContent,
    config: TranslationConfig
) -> TranslationResult:
    """Sliding window translation for long documents"""
    start_time = time.time()
    total_tokens = 0
    
    # Join all pages into single text for sliding window
    text = "\n\n".join(content.pages) if isinstance(content.pages[0], str) else ""
    
    # Create overlapping chunks
    chunks = create_chunks(
        text,
        window_size=config.window_size,
        overlap_size=config.overlap_size
    )
    
    # Translate chunks
    translated_chunks = []
    for chunk in chunks:
        translation = await translate_text(chunk, config)
        translated_chunks.append(translation.text)
        total_tokens += translation.tokens_used
        
    # Merge overlapping translations
    final_text = merge_chunks(translated_chunks, config.overlap_size)
    
    return TranslationResult(
        text=final_text,
        tokens_used=total_tokens,
        cost=calculate_cost(total_tokens, config.model),
        time_taken=time.time() - start_time
    )
```

### 4. Model Interface

```python
from litellm import completion
import base64

class TranslationResponse(BaseModel):
    """Structured response from translation model"""
    text: str
    tokens_used: int
    cost: float
    time_taken: float

async def translate_text(
    text: str,
    config: TranslationConfig
) -> TranslationResponse:
    """Translate text using any text-to-text model."""
    prompt = f"Translate the following text from {config.source_lang} to {config.target_lang}. Maintain all formatting and structure:\n\n{text}"
    
    response = await completion(
        model=config.model.value,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    return TranslationResponse(
        text=response.choices[0].message.content,
        tokens_used=response.usage.total_tokens,
        cost=response.usage.cost,
        time_taken=response.usage.completion_time
    )

async def translate_image(
    image: bytes,
    config: TranslationConfig
) -> TranslationResponse:
    """Translate content from image using multimodal models."""
    prompt = f"Translate this page from {config.source_lang} to {config.target_lang}. Maintain all formatting and structure."
    
    image_content = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64.b64encode(image).decode()}"
        }
    }
    
    response = await completion(
        model=config.model.value,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                image_content
            ]
        }],
        temperature=0.3
    )
    
    return TranslationResponse(
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
