import re
from typing import List, Dict

class SemanticCodeChunker:
    """
    Splits coding documents and natural text syllabus components 
    using logical boundaries (like functions, classes, or paragraph structures) 
    with customizable sliding overlaps.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_python_code(self, code: str) -> List[Dict[str, any]]:
        """
        Parses python code and groups it into logical chunks based on 
        class and function definitions, maintaining indentation context.
        """
        if not code:
            return []
            
        lines = code.splitlines()
        chunks = []
        current_chunk = []
        current_size = 0
        context_stack = [] # Tracks class or function header lines
        
        class_func_pattern = re.compile(r'^\s*(class|def)\s+(\w+)')
        
        for line_idx, line in enumerate(lines):
            match = class_func_pattern.match(line)
            
            # If we find a new block and our current chunk has exceeded overlap limits, commit it
            if match and current_chunk:
                # Keep parent header context
                header = match.group(0)
                if len(current_chunk) > 5 or current_size >= self.chunk_size:
                    chunks.append({
                        "text": "\n".join(current_chunk),
                        "start_line": line_idx - len(current_chunk) + 1,
                        "end_line": line_idx,
                        "context": list(context_stack)
                    })
                    
                    # Retain last few lines for overlap
                    overlap_lines = current_chunk[-max(1, int(self.chunk_overlap/30)):]
                    current_chunk = list(overlap_lines)
                    current_size = sum(len(l) for l in current_chunk)
            
            if match:
                # Add context (like the class or def signature)
                context_stack.append(line.strip())
                if len(context_stack) > 3:
                    context_stack.pop(0)
                    
            current_chunk.append(line)
            current_size += len(line)
            
            if current_size >= self.chunk_size:
                chunks.append({
                    "text": "\n".join(current_chunk),
                    "start_line": line_idx - len(current_chunk) + 2,
                    "end_line": line_idx + 1,
                    "context": list(context_stack)
                })
                current_chunk = []
                current_size = 0
                
        if current_chunk:
            chunks.append({
                "text": "\n".join(current_chunk),
                "start_line": len(lines) - len(current_chunk) + 1,
                "end_line": len(lines),
                "context": list(context_stack)
            })
            
        return chunks

    def chunk_text(self, text: str) -> List[str]:
        """
        Standard sliding window token/character chunker for natural language.
        """
        if not text:
            return []
            
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]
            
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = self.chunk_size // 2
            
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append(" ".join(chunk_words))
            if i + self.chunk_size >= len(words):
                break
                
        return chunks

code_chunker = SemanticCodeChunker()
