# 🔄 Tinbox: Your Ultimate Translation Tool

**Tinbox** is a robust command-line tool designed to tackle the challenges of translating large documents, especially PDFs, using Large Language Models (LLMs). Unlike other tools, Tinbox excels in handling extensive document sizes and navigates around model limitations related to size and copyright issues, ensuring seamless and efficient translations.

**Why Choose Tinbox?**

- **Handles Large Documents**: Efficiently processes large PDFs and other document types.
- **Overcomes Model Limitations**: Bypasses common model refusals due to size or copyright concerns.
- **No OCR Needed**: Directly translates PDFs using advanced multimodal models.
- **Smart Algorithms**: Achieve optimal translation results with our intelligent algorithms.
- **Local and Cloud Support**: Use models locally or in the cloud, depending on your preference.

**Quick Start Example:**

```bash
# Set your API key first
export OPENAI_API_KEY="your-api-key-here"
# or
export ANTHROPIC_API_KEY="your-api-key-here"

# Translate a document
tinbox translate --to es document.pdf
```

## 🎯 The Problems Tinbox Solves

1. **PDF Translation Challenges**

   - Most tools require OCR, leading to formatting loss and errors
   - Tinbox uses multimodal models to directly understand PDFs as images

2. **Large Document Limitations**

   - Traditional tools often fail with large documents
   - Models frequently refuse or timeout on big files
   - Tinbox smartly splits and processes documents while maintaining context

3. **Model Refusal Issues**

   - Many models refuse translation tasks due to:
     - Copyright concerns
     - Document size limitations
     - Rate limiting
   - Tinbox's algorithms work around these limitations intelligently

4. **Quality and Consistency**
   - **Context-Aware Algorithm**: Provides translation context from previous chunks for better consistency
   - Smart text splitting at natural boundaries (paragraphs, sentences, clauses)
   - No duplicate content issues (eliminates overlapping translation problems)
   - Maintains context between pages and segments

![Tinbox Workflow](link_to_diagram.png)

🔍 **Key Highlights:**

- Translate PDFs without OCR using advanced AI models
- Handle documents of any size with smart splitting algorithms
- Work around common model limitations and refusals
- Track costs and performance with built-in benchmarking

## ✨ Features

### 📄 Smart Document Handling

- **PDFs**: Processed directly as images - no OCR needed!
- **Word (docx)**: Preserves formatting while translating
- **Text files**: Efficient processing for large files

### 🧠 Intelligent Translation

- **Smart Algorithms**:
  - **Context-Aware** (recommended default): Smart text splitting with translation context
  - **Page-by-Page** with Seam Repair: Best for PDFs
  - **Sliding Window**: For long text documents
  - Automatic context preservation between sections

### 📚 Glossary Support

- **Consistent Terminology**: Maintain consistent translation of technical terms across documents
- **Term Learning**: Automatically discover and reuse domain-specific vocabulary
- **Persistent Glossaries**: Save and load glossaries across translation sessions
- **Algorithm Integration**: Works with all translation algorithms (page-by-page, sliding-window, context-aware)

### 🤖 Flexible Model Support

- Use powerful cloud models (GPT-4V, Claude 3.5 Sonnet)
- Run translations locally with Ollama
- Mix and match models for different tasks

### 🌐 Language Support

- Flexible source/target language specification using ISO 639-1 codes
- Common language aliases (e.g., "en", "zh", "es")

### 📊 Benchmarking

- Track overall translation time and token usage/cost
- Compare algorithms or model providers side-by-side

## 🚀 Getting Started

### Quick Install

```bash
# Recommended: Install with all features (includes PDF, DOCX, and image processing)
pip install "tinbox[all]"

# Or install base package (you'll need to install Pillow separately)
pip install tinbox
pip install Pillow  # Required for PDF processing

# For PDF support specifically
pip install "tinbox[pdf]"

# For Word document support
pip install "tinbox[docx]"
```

### System Dependencies

For PDF processing, Tinbox requires `poppler-utils` to be installed on your system:

#### macOS

```bash
brew install poppler
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install poppler-utils
```

#### Windows

1. Download poppler for Windows from [here](https://github.com/oschwartz10612/poppler-windows/releases/)
2. Extract the files and add the `bin` directory to your system PATH
3. Alternatively, use Chocolatey: `choco install poppler`

**Why is this needed?** Tinbox processes PDFs as images (no OCR required) using the `pdf2image` Python library, which depends on poppler's `pdfinfo` and `pdftoppm` utilities for PDF manipulation.

### 🔑 API Key Setup

Before using cloud models, set up your API keys:

```bash
# For OpenAI models (GPT-4o, GPT-5, etc.)
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic models (Claude)
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# For Google models (Gemini)
export GOOGLE_API_KEY="your-google-api-key"

# Or set them in your shell profile (.bashrc, .zshrc, etc.)
echo 'export OPENAI_API_KEY="your-key"' >> ~/.zshrc
```

### Basic Usage

1. **Translate a PDF to Spanish**

   ```bash
   tinbox translate --to es document.pdf
   ```

2. **Translate a Word document from Chinese to English**

   ```bash
   tinbox translate --from zh --to en document.docx
   ```

3. **Handle a large text file with context-aware algorithm**

   ```bash
   tinbox translate --to fr --context-size 1500 large_document.txt
   ```

4. **Using OpenAI models**

   ```bash
   tinbox translate --model openai:gpt-4o --to spanish document.pdf
   ```

5. **Using different output formats**

   ```bash
   tinbox translate --model openai:gpt-4o --to es --format json --output result.json document.pdf
   ```

6. **Enable checkpointing for large documents**
   ```bash
   tinbox translate --to es --checkpoint-dir ./checkpoints large_document.pdf
   ```

### 💡 Tips for Best Results

1. **For Most Documents**

   - Use the context-aware algorithm (default): `--algorithm context-aware`
   - Adjust chunk size if needed: `--context-size 2000`
   - Use custom split tokens for structured documents: `--split-token "|||"`

2. **For Large Documents**

   - Context-aware algorithm handles large documents intelligently
   - Alternative: Use sliding window: `--algorithm sliding-window`

3. **For PDFs**

   - Page-by-page algorithm works best for PDFs: `--algorithm page`
   - No OCR needed - just point to your PDF!

4. **For Best Performance**

   - Use local models via Ollama for no API costs
   - Cloud models (OpenAI, Anthropic, Google) for highest quality and faster processing

5. **For Long-Running Translations**

   - Enable checkpointing to resume interrupted translations: `--checkpoint-dir ./checkpoints`
   - Adjust checkpoint frequency for very large documents: `--checkpoint-frequency 10`
   - Checkpoints automatically resume from where you left off if translation is interrupted

6. **For Consistent Terminology**
   - Enable glossary for technical documents: `--glossary`
   - Build domain-specific glossaries: `--save-glossary terms.json`
   - Share glossaries across projects: `--glossary-file shared_terms.json`
   - Glossary works with all algorithms and improves translation consistency

## 📖 Detailed Documentation

### Command-Line Options

#### Core Options

| Option         | Description                                    | Example                   |
| -------------- | ---------------------------------------------- | ------------------------- |
| `--from, -f`   | Source language (auto-detect if not specified) | `--from zh`               |
| `--to, -t`     | Target language (default: English)             | `--to es`                 |
| `--model`      | Model to use for translation                   | `--model openai:gpt-4o`   |
| `--output, -o` | Output file (default: print to console)        | `--output translated.txt` |

#### Algorithm Options

| Option            | Description                                                       | Default         |
| ----------------- | ----------------------------------------------------------------- | --------------- |
| `--algorithm, -a` | Translation algorithm (`context-aware`, `page`, `sliding-window`) | `context-aware` |
| `--context-size`  | Target chunk size for context-aware algorithm (characters)        | 2000            |
| `--split-token`   | Custom token to split text on (context-aware only)                | None            |
| `--window-size`   | Size of translation window (sliding-window only)                  | 2000 tokens     |
| `--overlap-size`  | Overlap between windows (sliding-window only)                     | 200 tokens      |

#### Checkpoint Options

| Option                   | Description                                             | Default |
| ------------------------ | ------------------------------------------------------- | ------- |
| `--checkpoint-dir`       | Directory to store translation checkpoints for resuming | None    |
| `--checkpoint-frequency` | Save checkpoint every N pages/chunks                    | 1       |

#### Glossary Options

| Option            | Description                                      | Example                      |
| ----------------- | ------------------------------------------------ | ---------------------------- |
| `--glossary`      | Enable glossary for consistent term translations | `--glossary`                 |
| `--glossary-file` | Path to existing glossary file (JSON format)     | `--glossary-file terms.json` |
| `--save-glossary` | Path to save updated glossary after translation  | `--save-glossary terms.json` |

#### Output Format Options

| Option            | Description                          | Example Output          |
| ----------------- | ------------------------------------ | ----------------------- |
| `--format, -F`    | Output format (text, json, markdown) | See examples below      |
| `--benchmark, -b` | Include performance metrics          | Translation time, costs |

### Supported Languages

Common language codes (ISO 639-1):

| Code | Language | Also Accepts |
| ---- | -------- | ------------ |
| en   | English  | eng          |
| es   | Spanish  | spa          |
| zh   | Chinese  | chi, cmn     |
| fr   | French   | fra          |
| de   | German   | deu, ger     |
| ja   | Japanese | jpn          |
| ko   | Korean   | kor          |
| ru   | Russian  | rus          |
| ar   | Arabic   | ara          |
| hi   | Hindi    | hin          |

### Output Format Examples

#### 1. Plain Text (Default)

```bash
tinbox translate --to es document.pdf
# Output: Translated text...
```

#### 2. JSON Output

```bash
tinbox translate --to es --format json document.pdf
```

Example response:

```json
{
  "metadata": {
    "source_lang": "en",
    "target_lang": "es",
    "model": "claude-3-sonnet",
    "algorithm": "page"
  },
  "result": {
    "text": "Translated text...",
    "tokens_used": 1500,
    "cost": 0.045,
    "time_taken": 12.5
  }
}
```

#### 3. Markdown Report

```bash
tinbox translate --to es --format markdown document.pdf
```

### Advanced Usage

1. **Context-Aware Algorithm (Recommended)**

   ```bash
   # Use default context-aware with custom chunk size
   tinbox translate --to es --context-size 1500 document.txt

   # Split text on custom tokens (great for structured documents)
   tinbox translate --to fr --split-token "---" structured_document.txt
   ```

2. **Handling Very Large Documents**

   ```bash
   # Context-aware handles large documents intelligently (recommended)
   tinbox translate --to es --context-size 3000 large_document.txt

   # Alternative: sliding window algorithm
   tinbox translate --to es --algorithm sliding-window \
          --window-size 3000 --overlap-size 300 \
          large_document.pdf
   ```

3. **Using Local Models**

   ```bash
   tinbox translate --to fr --model ollama:mistral-small document.txt
   ```

4. **Cost Control and Dry Runs**

   ```bash
   # Check costs before translating
   tinbox translate --to de --dry-run --model openai:gpt-4o document.pdf

   # Set maximum cost limit
   tinbox translate --to de --max-cost 5.00 --model openai:gpt-4o document.pdf
   ```

5. **Checkpoint and Resume Support**

   ```bash
   # Enable checkpointing for large documents (saves progress every page/chunk)
   tinbox translate --to es --checkpoint-dir ./checkpoints document.pdf

   # Custom checkpoint frequency (save every 5 pages/chunks)
   tinbox translate --to fr --checkpoint-dir ./checkpoints --checkpoint-frequency 5 large_document.txt

   # Resume interrupted translation (automatically detects and resumes from checkpoint)
   tinbox translate --to es --checkpoint-dir ./checkpoints document.pdf
   ```

6. **Different Model Providers**

   ```bash
   # OpenAI GPT-4
   tinbox translate --to es --model openai:gpt-4o document.pdf

   # Anthropic Claude
   tinbox translate --to es --model anthropic:claude-3-sonnet document.pdf

   # Google Gemini
   tinbox translate --to es --model gemini:gemini-pro document.pdf
   ```

7. **Benchmarking Different Models**

   ```bash
   tinbox translate --to de --benchmark --model openai:gpt-4o document.pdf
   ```

8. **Glossary Support for Consistent Terminology**

   ```bash
   # Enable glossary with automatic term discovery
   tinbox translate --to es --glossary --save-glossary medical_terms.json medical_document.pdf

   # Load existing glossary and extend it
   tinbox translate --to fr --glossary-file existing_terms.json --save-glossary updated_terms.json technical_doc.pdf

   # Use glossary without saving updates
   tinbox translate --to de --glossary-file company_terms.json document.docx
   ```

### Glossary File Format

Glossary files are stored in JSON format:

```json
{
  "entries": {
    "CPU": "Processeur",
    "GPU": "Carte graphique",
    "SSD": "Disque SSD"
  }
}
```

## 🛠 Project Structure

```
tinbox/
├── src/
│   └── tinbox/
│       ├── cli.py                 # Command-line interface
│       ├── core/                  # Core functionality
│       │   ├── cost.py           # Cost tracking
│       │   ├── processor/        # Document processors
│       │   └── translation/      # Translation algorithms
│       └── utils/                # Utilities
└── tests/                        # Test suite
```

## 🧪 Running Tests

Here's how to run the tests:

### Prerequisites

Make sure you have the development dependencies installed:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Or if you're using the project locally
pip install pytest pytest-asyncio pytest-cov
```

### Running All Tests

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run tests with coverage report
pytest --cov=tinbox --cov-report=html
```

### Running Specific Test Categories

```bash
# Run only cost estimation tests
pytest tests/test_core/test_cost.py -v

# Run only CLI tests
pytest tests/test_cli.py -v

# Run all core functionality tests
pytest tests/test_core/ -v

# Run processor tests (PDF, DOCX, text)
pytest tests/test_core/test_processors/ -v

# Run context-aware algorithm tests
pytest tests/test_core/test_translation/test_context_aware.py -v
```

### Running Tests Without Coverage

If you encounter coverage-related issues, you can run tests without coverage:

```bash
pytest tests/ --override-ini="addopts=-ra -q"
```

### Test Structure

The test suite is organized as follows:

```
tests/
├── data/                     # Sample test documents
│   ├── sample_ar.pdf        # Arabic PDF for testing
│   └── sample_ar.docx       # Arabic DOCX for testing
├── test_cli.py              # CLI interface tests
├── test_core/               # Core functionality tests
│   ├── test_cost.py         # Cost estimation tests
│   ├── test_output.py       # Output formatting tests
│   ├── test_progress.py     # Progress tracking tests
│   ├── test_processors/     # Document processor tests
│   │   ├── test_pdf_processor.py
│   │   ├── test_text_processor.py
│   │   └── test_word_processor.py
│   └── test_translation/    # Translation algorithm tests
│       ├── test_algorithms.py
│       ├── test_context_aware.py  # Context-aware algorithm tests
│       └── test_litellm.py
└── test_utils/              # Utility function tests
    └── test_language.py
```

### Key Test Areas

1. **Cost Estimation** (`test_cost.py`)

   - Token counting for different file types
   - Input/output token cost calculations
   - Cost threshold warnings
   - Model pricing verification

2. **Document Processing** (`test_processors/`)

   - PDF text extraction and page handling
   - DOCX word counting and text extraction
   - Text file processing

3. **CLI Interface** (`test_cli.py`)

   - Command-line argument parsing
   - Dry-run functionality
   - Output formatting options
   - Error handling

4. **Translation Algorithms** (`test_translation/`)
   - Context-aware algorithm testing (smart splitting, context building)
   - Page-by-page algorithm testing
   - Sliding window algorithm testing
   - Model interface testing

### Continuous Integration

The project uses pytest with the following configuration (in `pyproject.toml`):

```toml
[tool.pytest.ini_options]
minversion = "8.0"
addopts = "-ra -q --cov=tinbox"
testpaths = ["tests"]
asyncio_mode = "auto"
```

### Writing New Tests

When contributing new features, please:

1. Add tests in the appropriate directory
2. Follow the existing naming convention (`test_*.py`)
3. Use descriptive test function names
4. Include docstrings explaining what each test does
5. Mock external dependencies (APIs, file system operations)

Example test structure:

```python
def test_feature_name():
    """Test description of what this test verifies."""
    # Arrange
    input_data = create_test_data()

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result.expected_property == expected_value
```

## 🔜 Future Plans

1. **Enhanced Output Formats**

   - PDF output with original formatting
   - Word document export
   - HTML with parallel text

2. **Advanced Features**

   - AI-powered section detection
   - Custom terminology support
   - Interactive translation review
   - Domain-specific model fine-tuning

3. **Performance Improvements**
   - Parallel processing
   - Better caching
   - Reduced API costs
