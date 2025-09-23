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
   - Smart algorithms ensure consistent translations across document sections
   - Maintains context between pages and segments
   - Repairs potential inconsistencies at section boundaries

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
  - Page-by-Page with Seam Repair (default for PDF)
  - Sliding Window for long text documents
  - Automatic context preservation between sections

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

3. **Handle a large text file with custom settings**

   ```bash
   tinbox translate --to fr --algorithm sliding-window large_document.txt
   ```

4. **Using OpenAI models**

   ```bash
   tinbox translate --model openai:gpt-4o --to spanish document.pdf
   ```

5. **Using different output formats**
   ```bash
   tinbox translate --model openai:gpt-4o --to es --format json --output result.json document.pdf
   ```

### 💡 Tips for Best Results

1. **For Large Documents**

   - Use the sliding window algorithm: `--algorithm sliding-window`
   - Adjust window size if needed: `--window-size 3000`

2. **For PDFs**

   - The default page-by-page algorithm works best
   - No OCR needed - just point to your PDF!

3. **For Best Performance**
   - Use local models via Ollama for no API costs
   - Cloud models (OpenAI, Anthropic, Google) for highest quality and faster processing

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

| Option            | Description                                        | Default        |
| ----------------- | -------------------------------------------------- | -------------- |
| `--algorithm, -a` | Translation algorithm (`page` or `sliding-window`) | `page` for PDF |
| `--window-size`   | Size of translation window                         | 2000 tokens    |
| `--overlap-size`  | Overlap between windows                            | 200 tokens     |

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

1. **Handling Very Large Documents**

   ```bash
   tinbox translate --to es --algorithm sliding-window \
          --window-size 3000 --overlap-size 300 \
          large_document.pdf
   ```

2. **Using Local Models**

   ```bash
   tinbox translate --to fr --model ollama:mistral-small document.txt
   ```

3. **Cost Control and Dry Runs**

   ```bash
   # Check costs before translating
   tinbox translate --to de --dry-run --model openai:gpt-4o document.pdf

   # Set maximum cost limit
   tinbox translate --to de --max-cost 5.00 --model openai:gpt-4o document.pdf
   ```

4. **Different Model Providers**

   ```bash
   # OpenAI GPT-4
   tinbox translate --to es --model openai:gpt-4o document.pdf

   # Anthropic Claude
   tinbox translate --to es --model anthropic:claude-3-sonnet document.pdf

   # Google Gemini
   tinbox translate --to es --model gemini:gemini-pro document.pdf
   ```

5. **Benchmarking Different Models**
   ```bash
   tinbox translate --to de --benchmark --model openai:gpt-4o document.pdf
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
