# Tinbox: A CLI Translation Tool

**Tinbox** is a command-line tool designed to translate documents using various
Large Language Models (LLMs), including both text-only and multimodal models.
For PDFs, it leverages multimodal models (like GPT-4o or Claude 3.5 Sonnet) to
directly process page images without OCR. For text-based formats (Word, TXT), it
uses text-to-text translation. The tool supports multiple input/output formats
and includes two primary translation algorithms—**Page-by-Page with Seam
Repair** and **Sliding Window**—with benchmarking capabilities for time and
token usage/cost. It also supports local model inference using Ollama.

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

## Installation & Dependencies

1. **Python 3.9+** (recommended)
2. **Model Dependencies**:
   - `litellm` for unified model interface
   - Local model support via Ollama
   - Appropriate API keys for cloud models

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
| `--format, -F <FORMAT>`        | Output format (text, json, or markdown). Defaults to text.                                           |
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

## Project Structure

```
tinbox/
├── src
│   └── tinbox
│       ├── cli.py
│       ├── core
│       │   ├── cost.py
│       │   ├── __init__.py
│       │   ├── output.py
│       │   ├── processor
│       │   │   ├── docx.py
│       │   │   ├── __init__.py
│       │   │   ├── pdf.py
│       │   │   └── text.py
│       │   ├── progress.py
│       │   ├── translation
│       │   │   ├── algorithms.py
│       │   │   ├── checkpoint.py
│       │   │   ├── __init__.py
│       │   │   ├── interface.py
│       │   │   ├── litellm.py
│       │   │   └── types.py
│       ├── __init__.py
│       └── utils
│           ├── language.py
│           ├── logging.py
└── tests
    ├── data
    ├── test_cli.py
    ├── test_core
    │   ├── test_cost.py
    │   ├── test_output.py
    │   ├── test_processor.py
    │   ├── test_processors
    │   │   ├── test_pdf_processor.py
    │   │   ├── test_text_processor.py
    │   │   └── test_word_processor.py
    │   ├── test_progress.py
    │   └── test_translation
    │       ├── test_algorithms.py
    │       └── test_litellm.py
    └── test_utils
        └── test_language.py
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
