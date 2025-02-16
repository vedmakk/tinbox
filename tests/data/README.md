# Test Data

This directory contains sample documents used for testing the document processors.

## Files

### PDF Documents
- `sample_ar.pdf`: Multi-page PDF containing Arabic text
  - Used for testing PDF processing
  - Tests multi-page handling
  - Tests right-to-left text support
  - Tests non-Latin character handling

### Word Documents
- `sample_ar.docx`: Multi-page Word document containing Arabic text
  - Used for testing DOCX processing
  - Tests multi-page handling
  - Tests right-to-left text support
  - Tests non-Latin character handling

## Usage

These files are used in the test suite, particularly in:
- `tests/test_core/test_processors/test_pdf_processor.py`
- `tests/test_core/test_processors/test_word_processor.py`

Please ensure these files remain in version control as they are essential for testing. 
