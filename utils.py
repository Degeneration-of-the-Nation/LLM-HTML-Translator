"""
Utility functions for Multilingual HTML Translator
Original site: https://hitdarderut-haaretz.org
Translated versions: https://degeneration-of-nation.org

Contains helper functions for text processing, validation, and error handling
"""

import re
from collections import Counter
from typing import List, Tuple
from config import LANGUAGE_CONFIG

def remove_formatting(s: str) -> str:
    """Cleans and standardizes HTML formatting while preserving structure"""
    s = re.sub(r'\s+', ' ', s.replace("\n", " ").replace("\t", " ").replace("\r", " "))
    s = s.replace("> <", "><").replace(" <br>", "<br>").replace("<br> ", "<br>")
    return s

def truncate_str(s: str, length: int = 30) -> str:
    """Truncates string with ellipsis in middle if too long"""
    return f"{s[:length]}...{s[-length:]}" if len(s) > length * 2 else s

def abnormal_repetitions(text: str, asian: bool, window_size=200) -> List[str]:
    """
    Detects abnormal word repetitions in translated text
    Used to catch potential translation errors or hallucinations
    """
    words = remove_formatting(re.sub(r'[^\w\s]', ' ', re.sub(r'<[^>]+>', '', text.replace('&nbsp;', " ")))).lower().strip()
    words = list(words.replace(" ", "")) if asian else words.split()
    
    for i in range(0, len(words) - window_size + 1, window_size // 2):
        word_counts = Counter(words[i:i+window_size])
        repeated_words = [word for word, count in word_counts.items() 
                         if count >= (35 if asian else 20)]
        if len(repeated_words) > 2:
            return repeated_words
    return []

def extract_test(answer: str) -> tuple[str, str]:
    """Extracts test tags from translator response"""
    start_index = answer.find("<test")
    if start_index == -1:
        return "", answer
    return (answer[start_index + len("<test "):answer.rfind("></test>")].strip(), 
            answer[:start_index])

def contains_hebrew(text: str) -> List[int]:
    """
    Checks for Hebrew characters in text
    Returns list of positions where Hebrew characters were found
    """
    hebrew_chars = []
    for i, char in enumerate(text):
        for start, end in LANGUAGE_CONFIG['hebrew_chars']['ranges']:
            if start <= ord(char) <= end:
                hebrew_chars.append(i)
                if len(hebrew_chars) > 15:  # Early return if too many found
                    return hebrew_chars
    return hebrew_chars

def adjust_paths_after_translation(content: str, lang: str, path_mappings: dict) -> str:
    """
    Adjusts paths and URLs in translated content
    Handles relative paths, domains, and section mappings
    """
    # Set HTML lang and dir attributes
    content = re.sub(r'<html[^>]*>', f'<html lang="{lang}" dir="ltr">', content)
    
    # Replace img src paths
    content = content.replace('url(img/', 'url(../img/')
    
    # Update domain references
    content = content.replace('hitdarderut-haaretz.org', f'degeneration-of-nation.org/{lang}')
    
    # Update relative href paths
    def href_replace(match):
        quote_type = match.group(1)
        href = match.group(2)
        if not href.startswith('http') and not href.endswith(('.html', '.js')) and href != '/':
            href = '../' + href
        if href == '/':
            href = './index.html'
        return f'href={quote_type}{href}{quote_type}'
    
    content = re.sub(r'href=(["\'])((?![^"\']*[$}])[^"\']*)\1', href_replace, content)
    
    # Replace section paths
    for old_path, new_path in path_mappings.items():
        content = content.replace(old_path, new_path)
    
    return content