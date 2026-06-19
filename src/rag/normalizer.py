import re
import tokenize
from io import StringIO

class CodeTextNormalizer:
    """
    Handles preprocessing, cleanup, and normalization of programming code 
    and natural text to optimize semantic chunking and index querying.
    """
    
    @staticmethod
    def remove_python_comments_and_docstrings(source: str) -> str:
        """
        Removes all comments (#) and docstrings (''' or \"\"\") from python code.
        """
        if not source:
            return ""
        
        io_obj = StringIO(source)
        out = []
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        
        try:
            tokens = tokenize.generate_tokens(io_obj.readline)
            for tok in tokens:
                token_type = tok[0]
                token_string = tok[1]
                start_line, start_col = tok[2]
                end_line, end_col = tok[3]
                
                # Suppress docstrings
                if token_type == tokenize.STRING:
                    if prev_toktype == tokenize.INDENT or prev_toktype == tokenize.NEWLINE:
                        # Likely a docstring
                        continue
                
                # Suppress comments
                if token_type == tokenize.COMMENT:
                    continue
                
                if start_line > last_lineno:
                    last_col = 0
                if start_col > last_col:
                    out.append(" " * (start_col - last_col))
                
                out.append(token_string)
                last_lineno = end_line
                last_col = end_col
                prev_toktype = token_type
                
            return "".join(out).strip()
        except Exception:
            # Fallback regex-based clean in case tokenization fails due to syntax errors
            # Remove multiline docstrings
            source = re.sub(r'\"\"\"[\s\S]*?\"\"\"', '', source)
            source = re.sub(r"\'\'\'[\s\S]*?\'\'\'", '', source)
            # Remove single-line comments
            source = re.sub(r'#.*', '', source)
            # Remove blank lines
            return "\n".join([line for line in source.splitlines() if line.strip()]).strip()

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Normalizes multiple whitespaces, tabs, and newlines into a consistent format.
        """
        if not text:
            return ""
        # Replace multiple spaces with a single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    @staticmethod
    def clean_extracted_ocr(text: str) -> str:
        """
        Cleans typical raw OCR artifacts (e.g., weird characters, broken indentation).
        """
        if not text:
            return ""
        
        # Replace common OCR misreads of indentation (like multiple dots or underscores at start of lines)
        lines = []
        for line in text.splitlines():
            # If line starts with weird OCR characters indicating leading spaces
            line_cleaned = re.sub(r'^[\.\-\_\:\~\s]+', lambda m: ' ' * len(m.group(0)), line)
            lines.append(line_cleaned)
            
        cleaned = "\n".join(lines)
        return cleaned.strip()

code_normalizer = CodeTextNormalizer()
