"""Document processor implementations."""

# These will be implemented in stages
__all__ = [
    "PDFProcessor",
    "WordProcessor",
    "TextProcessor",
]

# Import implemented processors
from tinbox.core.processors.pdf import PDFProcessor

# Placeholders will raise NotImplementedError
from tinbox.core.processors.word import WordProcessor
from tinbox.core.processors.text import TextProcessor 
