# Notes for Claude

## Environment Setup

- Use `uv pip install` instead of regular pip for package installation
- For local development installation: `uv pip install -e .`

## Common Commands

Use `tinbox --help` to see all available options. A few useful commands:

- `tinbox --to es document.pdf` — translate a PDF to Spanish
- `tinbox --from zh --to en document.docx` — translate a Word document from Chinese to English

## Project-Specific Information

- Pillow is required for image processing in the PDF processor and LiteLLM translator
