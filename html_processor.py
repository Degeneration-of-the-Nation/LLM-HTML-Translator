"""
HTML Processing module for Multilingual HTML Translator
Original site: https://hitdarderut-haaretz.org
Translated versions: https://degeneration-of-nation.org

Handles HTML structure preservation, validation, and reconstruction
while maintaining original formatting and structure
"""

import re
from typing import List, Tuple
import logging
from utils import truncate_str

logger = logging.getLogger('website_translator')

def extract_html_structure(content: str) -> List[List]:
    """
    Extracts HTML structure from content
    Returns list of [element, is_html_tag, position] for each content part
    """
    structure = []
    pattern = re.compile(r'(<[^>]+>|[^<]+)')
    for match in pattern.finditer(content):
        item = match.group(1)
        end_position = match.end()
        if item.strip():
            structure.append([item, item.startswith('<') and item.endswith('>'), end_position])        
    return structure

def validate_html_structure(original_structure: List[str], translated_content: str, 
                          start_position: int, leftover: str) -> Tuple[int, str]:
    """
    Validates translated content against original HTML structure
    Returns updated position and leftover content
    """
    translated_structure = extract_html_structure(leftover + translated_content)
    
    for i, (orig, trans) in enumerate(zip(original_structure[start_position:], translated_structure)):
        if i == (len(translated_structure)-1) and not trans[1]:
            if len(translated_structure) == 1:
                logger.info("\n\n!Answer without HTML inside!\n")
                leftover = leftover + translated_content
            else:
                leftover = translated_content[translated_content.rfind(translated_structure[-2][0]) 
                                           + len(translated_structure[-2][0]):]
                if leftover.lstrip().startswith('<'):
                    continue 

        if orig[1] != trans[1] or (orig[1] and re.split(r'[\s>]', orig[0])[0] != 
                                  re.split(r'[\s>]', trans[0])[0]):
            logger.error(f"<><><>< Structure mismatch ><><><> Original Positions {start_position}-{start_position +i}"
                        f" / {len(original_structure)-1} | Translated 0-{i} / {len(translated_structure)-1}")
            
            error = ValueError("Structure Mismatch")
            error.ratio = min(1, ((len("".join(map(str, [s[0] for s in translated_structure[0:i+1]])))
                                 - len(leftover)) / len(translated_content)) + 0.05)
            error.orig = orig
            error.trans = trans            
            error.original_context = "".join(map(str, [truncate_str(s[0], 50) for s in 
                                    original_structure[max(start_position, start_position + i - 4):start_position + i+1]]))
            error.translated_context = "".join(map(str, [truncate_str(s[0], 50) for s in 
                                             translated_structure[max(0, i - 4):i+1]]))
            raise error
    
    end_position = start_position + len(translated_structure)
    leftover = "" if translated_structure[-1][1] else leftover
    if not translated_structure[-1][1]:
        end_position -= 1
    
    logger.info(f"@-->--Validated--<--@ Original Positions {start_position}-{start_position +i} /"
                f" {len(original_structure)-1} {truncate_str(str(orig), 50)} | Translated 0-{i} /"
                f" {len(translated_structure)-1} {truncate_str(str(trans), 50)} end_position={end_position}"
                f" ...leftover={truncate_str(leftover)}\n")
    
    return end_position, leftover

def reconstruct_html_from_structure(original_structure: List[str], translated_content: str) -> str:
    """
    Reconstructs HTML content while preserving original structure
    Returns complete HTML with preserved formatting
    """
    result = translated_content
    search_start = 0
    
    for item in original_structure:
        if item[1]:  # is HTML tag
            tag_to_find = re.split(r'[\s>]', item[0])[0] + '>'  
            tag_position = result.find(tag_to_find, search_start)
            next_tag_position = result.find("<", search_start)
            
            if tag_position != -1 and tag_position == next_tag_position:
                result = result[:tag_position] + item[0] + result[tag_position + len(tag_to_find):]
                search_start = tag_position + len(item[0])
            else:
                if result[next_tag_position:next_tag_position + 24] == "<incomplete-translation>":
                    break
                raise ValueError(f"Error: Could not find a match for {truncate_str(str(item))} after position"
                               f" {search_start} starting at {result[search_start:search_start+50]}..."
                               f" tag_to_find: {tag_to_find}")
    
    return result