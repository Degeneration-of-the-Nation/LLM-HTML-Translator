"""
Main Translation module for Multilingual HTML Translator
Original site: https://hitdarderut-haaretz.org
Translated versions: https://degeneration-of-nation.org

Handles the core translation process, including chunking, recovery,
and orchestration of the entire translation pipeline
"""

import os
import time
import json
import subprocess
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import subprocess
import re
import logging
from anthropic import RateLimitError, BadRequestError
logger = logging.getLogger('website_translator')

from html_processor import (extract_html_structure, validate_html_structure, 
                          reconstruct_html_from_structure)
from utils import (truncate_str, remove_formatting, abnormal_repetitions, 
                  extract_test, contains_hebrew, adjust_paths_after_translation)
from api_client import TranslationAPIClient
from logger import setup_logger, set_verbose_mode
from config import (TRANSLATION_CONFIG, LANGUAGE_CONFIG, 
                   PATH_MAPPING_CONFIG, EXECUTION_CONFIG)
MODE = EXECUTION_CONFIG['mode']
TEST_FILE_COUNT = EXECUTION_CONFIG['test_file_count']



def translate_chunk(chunk: list, language: str, lang_config: dict, 
                   previous_context: str, hebrew_context: str,
                   destination_path: str, html_structure: List[str], 
                   api_client: TranslationAPIClient,
                   problematic: bool, new_model: bool) -> Tuple[str, bool]:
    """
    Translates a single chunk of HTML content while maintaining structure
    Includes recovery mechanisms and validation
    """
    system_text = f"""<task>Translate Hebrew culture website into {language} website. Translate the provided HTML content, while preserving all HTML structure</task>
<role>You are an expert literary translator specializing in translating from Hebrew to {language}:
    <literary-quality>
        <goal>Create high-quality translation that could be published as respected literary or intellectual work in {language}</goal>
        <process>Exercise deep consideration in every translation decision. Pay special attention to the artistic elements</process>
    </literary-quality>
    <readability>Produce natural, fluent, and compelling text in {language}, avoiding word-for-word translation</readability>
    <adaptation>Adapt cultural references, idioms, metaphors, wordplay, puns, rhythm, rhyme, and literary devices for the {language}-reading audience</adaptation>
    <untranslatable-terms>For terms unfamiliar to {language} readers (including biblical/religious allusions and Israeli/Jewish cultural or historical references):
        <format>Provide a concise explanation in square brackets [like this] immediately after the term, always within the same HTML tag. If elaboration is needed, use sparingly [{lang_config['translator_note']}: explanation]. Avoid in titles/short elements</format>
    </untranslatable-terms>
    <website-name>Translate "{LANGUAGE_CONFIG['source_text']['website_name']}" as "{lang_config['title']}"</website-name>
</role>
<output-format>
    <hebrew-only-translation>Translate all and only Hebrew content to {language}. No Hebrew allowed in output</hebrew-only-translation>
    <html-structure>Preserve all HTML structure, including all tags and HTML entities. Purpose: To replicate my site in {language}</html-structure>
    <br-tags>Preserve all '<br>' tags exactly. Do not replace them with line breaks or omit them. Never use line-breaks</br-tags>
    <nbsp-entities>Preserve all '&nbsp;'. Do not convert them to spaces</nbsp-entities>
    <html-format-example>
Example input: <div>שלום&nbsp;עולם<br><b>כותרת</b></div>
Example output: <div>{lang_config['example']}</div>
    </html-format-example>
</output-format>
<validation>Violating these rules will cause automatic failure of the translation task:
    <html-preservation>Ensure all HTML elements are preserved EXACTLY as in the original, including '<br>', '&nbsp;' and '<b>'. Do not omit ANY of them</html-preservation>
    <no-omissions>Ensure that ABSOLUTELY no text is omitted from within the translation. Translate all content SEQUENTIALLY without skipping or jumping ahead</no-omissions>
    <no-stops>If more text remains, continue translating and hit the token limit. Only stop answering at the end of the entire original text. NEVER interrupt the translation to ask or update anything. NEVER clip it or break into parts</no-stops>"""

    if new_model:
        system_text += """
    <no-additions>Output ONLY the translated HTML content. NO meta-text, status updates, questions, or comments about the translation process</no-additions>
    <no-final-comment>Do not add any sentence after the translated content</no-final-comment>"""

    system_text += f"""
</validation>
<summary>
    <content>For Hebrew content: Ensure high-quality, natural translation</content>
    <structure>For HTML structure: Maintain precise technical replication</structure>
</summary>

HTML text to translate into {language}:
{hebrew_context}{chunk[1]}"""

    constant_instruction = "Re-read <output-format> and <validation> requirements and verify strict compliance. Do not omit ANY content or HTML element, NOR replace <br> with line break. Do not stop mid-translation. NEVER truncate your answer before the end of the entire original text"
    constant_instruction += ", or end it with ANY comment." if new_model else "."
# Setup for translation state management
    assistant_overlap = TRANSLATION_CONFIG['chunk_overlap'] // TRANSLATION_CONFIG['overlap_parting']
    max_tokens = TRANSLATION_CONFIG['max_tokens']
    saved_data = {'chunks': [], 'translations': []}
    answer_start = f"<body><div><a>{lang_config['title']}</a></div><"
    answer_end = "</body>"
    chunk_translation, warning_message, leftover = "", "", ""
    conversation = []
    token_count, chunk_parts, current_position, attempt = 0, 0, 0, -1

    # Check for existing partial translation
    json_file = f"{destination_path}.json"
    if os.path.exists(json_file):
        answer_start = ""
        with open(json_file, 'r') as f:
            saved_data = json.load(f)
            current_position = saved_data['current_position']
            leftover = saved_data['leftover']
        
        # If chunk was already translated (except last chunk)
        if chunk in saved_data['chunks']:
            chunk_translation = saved_data['translations'][int(chunk[0])]
            if chunk != saved_data['chunks'][-1]:
                return chunk_translation, problematic        
            chunk_end_position = saved_data['chunk_end_position']
            previous_context = (previous_context + chunk_translation)[-TRANSLATION_CONFIG['chunk_overlap']:]
    
    # Set chunk end position for new or continuing translation
    if not chunk_translation:
        chunk_end_position = len(extract_html_structure(chunk[1])) + current_position
    
    # Determine whether to use cache based on chunk size and position
    use_cache = problematic
    logger.info(f"Chunk: {truncate_str(chunk[1],100)} | chunk_position_length={chunk_end_position - current_position} Cache={use_cache}")

    # Main translation loop
    while attempt < TRANSLATION_CONFIG['max_retries'] - 1:
        chunk_parts += 1
        max_tokens = (TRANSLATION_CONFIG['max_tokens'] + max_tokens) // 2

        # Prepare context for continuing translation
        if previous_context:
            assistant_answer = previous_context[-assistant_overlap:]
            prompt_context = previous_context[:assistant_overlap * (TRANSLATION_CONFIG['overlap_parting'] - 1)]
            current_length = int(((len(chunk_translation) - len(assistant_answer)) 
                                if chunk_translation else len(prompt_context)) / lang_config.get('multiplier', 1))
            if chunk_translation:
                current_length += max(len(hebrew_context) - 3, 0)
            prompt = f"""About {max(current_length, 0)} characters were translated. End of previous translation: <context>{prompt_context}</context> Continue the translation exactly from where you left off. """
        else:
            prompt = ""
            assistant_answer = answer_start

        # Translation attempt loop
        for attempt in range(TRANSLATION_CONFIG['max_retries']):
            try:
                # Prepare conversation for API
                new_conversation = [
                    {"role": "user", 
                     "content": [{"type": "text", "text": prompt + constant_instruction + warning_message}]},
                    {"role": "assistant", 
                     "content": [{"type": "text", "text": assistant_answer}]}
                ]

                # Check for conversation loop
                if new_conversation == conversation:
                    raise RuntimeError("Stuck in a loop with LLM")
                else:
                    conversation = new_conversation

                # Calculate temperature based on attempt and conditions
                temperature = (int(bool(attempt)) if problematic else 
                             min(TRANSLATION_CONFIG['default_temperature'] + 0.3 * attempt, 1))

                # Log translation attempt details
                logger.info(f"""Chunk {chunk[0]+1} Part {chunk_parts} Attempt {attempt+1} 
                           {'New' if new_model else 'Old'} Model. 
                           Translation size/Chunk size={len(chunk_translation)}/{len(chunk[1])}
                           System text size: {len(system_text)} Temp={temperature} 
                           Max_tokens={max_tokens}""")

				# Make API call and handle response
                if TRANSLATION_CONFIG['sleep_time']:
                    logger.info(f".......Waiting {TRANSLATION_CONFIG['sleep_time']} seconds before API call")
                    time.sleep(TRANSLATION_CONFIG['sleep_time'])
                    logger.info(">>====> API Call >>====>")
                
                # Send translation request to API
                message = api_client.create_translation_message(
                    system_text=system_text,
                    prompt=prompt + constant_instruction + warning_message,
                    assistant_answer=assistant_answer,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    use_cache=use_cache,
                    new_model=new_model
                )

                # Log API response details
                with open(destination_path + ".txt", mode='a', encoding='utf-8') as f:
                    f.write(f"{time.strftime('%H:%M')} Conversation: {conversation}\n\nMessage: {message}\n\n\n")
                
                # Process API response
                if message.content:
                    answer = message.content[0].text
                    test, answer = extract_test(answer)
                    answer = answer.rstrip().lstrip('\n')
                    
                    if test:
                        logger.info(f"0v0 Test: {test}")
                    
                    # Remove any trailing comments
                    if (match := re.search(r'\s*(?:\[[^\]]+\][\s]*)+$', answer)):
                        logger.error("\n\n[!COMMENT!]: ..." + answer[max(0, match.start() - 50):] + "\n\n")
                        subprocess.run(["espeak", "Comment Comment Comment"])
                        answer = answer[:match.start()].rstrip()

                # Validate response
                if not message.content or not answer:
                    warning_message = f"Last time, you returned an empty answer. Do not repeat this mistake. DO NOT stop before the very last character of this HTML text"
                    if html_structure[chunk_end_position - 1][1]:
                        warning_message += ", NOR forget to include the last HTML tags in your answer. Your translation must end with " + html_structure[chunk_end_position - 1][0]
                    raise ValueError("Empty Answer")

                # Log translation progress
                logger.info(f"{(f'+++ Continued Translation: {previous_context[-50:]}+++' if previous_context else 'Answer: ')}{answer[:100]}...")
                logger.info(f"---] Answer End ({len(answer)}):...{answer[-100:]}")

                # Process and validate answer
                answer = ("" if previous_context else answer_start) + answer
                if message.stop_reason == 'stop_sequence':
                    answer += answer_end

                # Check for Hebrew characters in translation
                hebrew_chars = contains_hebrew(answer)
                if hebrew_chars:
                    hebrew = answer[hebrew_chars[0]-20:hebrew_chars[0]+20]
                    warning_message = f"""Last time, there was Hebrew inside your answer, instead of only {language}: "...{hebrew}...". NEVER use hebrew letters in your answer. DO NOT repeat this mistake</warning>"""
                    raise ValueError("Hebrew in translation: ..." + hebrew)

                # Check for abnormal repetitions
                repeating_words = abnormal_repetitions(answer, lang_config['html'] in "ja,zh,hi")
                if repeating_words:
                    warning_message = f"""Content hallucination was detected in your previous answer. Within ~200 words, these appeared over 20 times each: {", ".join(map(str, repeating_words))}. This indicates severe repetition and loss of context. DO NOT repeat this mistake</warning>"""
                    raise ValueError("abnormal_repetition detected")

                # Validate HTML structure
                if not re.search(r'<[^>]+>', answer) and len(answer) < len(chunk[1]) * lang_config.get('multiplier', 1) / 3:
                    warning_message = "Last time you returned only text without HTML structure. ONLY stop answering at the end of the entire original text</warning>"
                    raise ValueError("Cannot validate: Only text, No HTML")

# Validate and update HTML structure
                current_position, leftover = validate_html_structure(
                    html_structure, answer, current_position, leftover)
                
                # Handle remaining structure elements
                structure_leftover = html_structure[current_position: chunk_end_position]
                if structure_leftover and all([item[1] for item in structure_leftover]) and not leftover:
                    last_tags = re.sub(
                        r'<([^/>\s]+)[^>]*>', 
                        r'<\1>', 
                        ''.join(map(str, [item[0] for item in structure_leftover]))
                    )
                    answer += last_tags
                    current_position = chunk_end_position
                    logger.info("\n\n(8-)-<--<  Adding </Body> or last <tags>: " + last_tags)

                # Save partial translation
                with open(f"{destination_path}.partial.html", 'a') as f:
                    f.write(answer)
                    
                chunk_translation += remove_formatting(answer)
                token_count += message.usage.output_tokens

                # Update saved state
                saved_data['current_position'] = current_position
                saved_data['leftover'] = leftover
                saved_data['chunk_end_position'] = chunk_end_position
                if chunk not in saved_data['chunks']:
                    saved_data['chunks'] = saved_data['chunks'][:-1]
                    saved_data['chunks'].append(chunk)
                    saved_data['translations'].append(chunk_translation)
                else:
                    saved_data['translations'][chunk[0]] = chunk_translation

                # Check if chunk is complete
                if (message.stop_reason == "stop_sequence" or 
                    (current_position == chunk_end_position and not leftover) or 
                    (current_position == chunk_end_position - 1 and leftover)):
                    
                    logger.info(f"(-: Chunk Complete :-) Tokens: {token_count}. "
                              f"Multiplier: {(token_count / len(chunk[1])):.2f}")
                    
                    # Save final state
                    if os.path.exists(json_file):
                        os.rename(json_file, json_file + ".old")
                    with open(json_file, 'w') as f:
                        saved_data['chunks'].append("Chunk End")
                        json.dump(saved_data, f)
                    
                    return chunk_translation, False

                # Prepare for next iteration
                previous_context = chunk_translation[-TRANSLATION_CONFIG['chunk_overlap']:]            
                answer_start, warning_message = "", ""
                attempt = -1
                break

            except Exception as e:
                logger.error(f"((0)) Error in translate_chunk for {destination_path}: {e} \n\n")
                
                # Handle critical errors
                if (isinstance(e, RateLimitError) or 
                    isinstance(e, BadRequestError)):
                    subprocess.run(["espeak", str(e)])
                    raise RuntimeError("!Stop!")
                
                # Handle recovery scenarios
                if (attempt >= TRANSLATION_CONFIG['max_retries'] - 1 or 
                    (isinstance(e, ValueError) and str(e) == "Empty Answer") or 
                    (isinstance(e, RuntimeError) and str(e) == "Stuck in a loop with LLM")):
                    
                    if TRANSLATION_CONFIG['recovery_mode'] and (os.path.exists(json_file) or chunk_translation):
                        name = os.path.basename(destination_path)
                        return f"""{'' if os.path.exists(json_file) else chunk_translation} 
                        <incomplete-translation></incomplete-translation>{"</div>"*20}
                        <div style="text-align:center;margin:0 auto;">
                        {lang_config['future']}<br><br>
                        <a style="text-decoration:underline;" href="/en/{name}.html">
                        Read complete version in English</a></div><br><br>
                        <div style="font-size:13px;text-align:center;border-top:1px solid black;padding:4px;margin-bottom:2px;">
                        <a style="text-decoration:underline;" href="https://degeneration-of-nation.org/index.html">
                        Our Site in Multiple Languages</a></div></body>""", True
                    else:
                        raise

                # Handle structure mismatch
                if isinstance(e, ValueError) and str(e) == "Structure Mismatch":
                    max_tokens = int(max(message.usage.output_tokens * e.ratio, 
                                      TRANSLATION_CONFIG['min_tokens']))
                    use_cache = True
                    orig = re.sub(r'<([^/>\s]+)[^>]*>', r'<\1>', e.orig[0]) if e.orig[1] else "text"
                    trans = e.trans[0] if e.trans[1] else "text"
                    warning_message = f"""An HTML mismatch was detected in your previous answer, due to earlier omissions or alterations. Instead of {orig} there was {trans}. Context up to and including mismatch: <original>{e.original_context}</original> <your-translation>{e.translated_context}</your-translation></warning> DO NOT repeat this mistake, NOR misplace any {orig} or {trans}."""
                
                if warning_message and "<warning>" not in warning_message:
                    warning_message = f" <warning>ATTENTION: This is your last attempt to translate this section into {language}. " + warning_message

    return "", True

def adjust_chunks(chunks: List[List]) -> List[List]:
    """
    Adjusts chunk boundaries to maintain coherent content
    Uses configured separators to find natural break points
    """
    separators = LANGUAGE_CONFIG['text_separators']
    
    for chunk in chunks[:-1]:  # Process all chunks except the last
        next_chunk = chunks[chunk[0]+1]
        
        # Try to find natural break points
        for separator in separators:
            next_cut = next_chunk[1].find(separator, 0, len(next_chunk[1])//7)
            cut = chunk[1].rfind(separator, -len(chunk[1])//7)
            
            # Adjust chunk boundaries if better break point found
            if -1 < next_cut < len(chunk[1]) - cut - 2 * len(separator):
                next_cut += len(separator)
                chunk[1] += next_chunk[1][:next_cut]
                next_chunk[1] = next_chunk[1][next_cut:]
                break
                
            if cut != -1:
                cut += len(separator)
                next_chunk[1] = chunk[1][cut:] + next_chunk[1]
                chunk[1] = chunk[1][:cut]
                break
        
        # Ensure HTML tag completeness
        for item in extract_html_structure(next_chunk[1]):
            if item[1]:  # is HTML tag
                adjusted_cut = next_chunk[1].find(item[0]) + len(item[0])
                chunk[1] += next_chunk[1][:adjusted_cut]
                next_chunk[1] = next_chunk[1][adjusted_cut:]
            else:
                break
                
        logger.info(f"--><-- Cut chunks {chunk[0]+1} ({len(chunk[1])}, {len(next_chunk[1])} chars): "
                   f"{chunk[1][-40:]}+ + +{next_chunk[1][:40]}")
    
    return chunks

def translate_html_file(content: str, language: str, lang_config: dict, 
                       destination_path: str, api_client: TranslationAPIClient) -> str:
    """
    Translates complete HTML file while preserving structure
    Handles splitting content, translation, and reconstruction
    """
    # Split HTML into header and body
    split_index = content.index('<div class="mainheadline">')
    first_part = content[:split_index]
    second_part = "<body>" + content[split_index:content.rfind("</html>")]

    # Extract and prepare HTML structure
    html_structure = extract_html_structure(second_part)
    translation_content = remove_formatting(
        re.sub(r'<([^/>\s]+)[^>]*>', r'<\1>', second_part)
    )

    # Split content into manageable chunks
    chunks_num = len(translation_content) // TRANSLATION_CONFIG['chunk_max_size'] + 1
    chunk_size = len(translation_content) // chunks_num + 1
    chunks = adjust_chunks([
        [i, translation_content[i*chunk_size:(i+1)*chunk_size]] 
        for i in range(0, chunks_num)
    ])

    # Translate content
    translated_content = ""
    previous_context = ""
    hebrew_context = ""
    problematic = os.path.exists(f"{destination_path}.txt")

    try:
        for chunk in chunks:
            if len(chunks) > 1:
                logger.info(f"]---[]---[ Translating Chunk {chunk[0]+1}/{len(chunks)} ]---[]---[")
            
            # Attempt translation with retry
            response, problematic = translate_chunk(
                chunk, language, lang_config, previous_context, hebrew_context,
                destination_path, html_structure, api_client, problematic, True
            )
            
            if not response:
                response, problematic = translate_chunk(
                    chunk, language, lang_config, previous_context, hebrew_context,
                    destination_path, html_structure, api_client, problematic, True
                )
                if not response:
                    raise ValueError("(((((((0))))))) Failure after chunk retry")

            translated_content += response
            previous_context = translated_content[-TRANSLATION_CONFIG['chunk_overlap']:]
            hebrew_context = "..." + chunk[1][-int(TRANSLATION_CONFIG['chunk_overlap'] / 
                                                 lang_config.get('multiplier', 1)):]

            if TRANSLATION_CONFIG['recovery_mode'] and "<incomplete-translation>" in translated_content:
                break

        # Post-process translated content
        translated_content = reconstruct_html_from_structure(
            html_structure, translated_content + "</html>"
        )
        translated_content = translated_content.replace("<body>", "", 1)
        full_content = first_part + translated_content

        # Update title
        article_title = re.search(
            r'<div class="row">.*?<div>(.*?)</div>', full_content, re.DOTALL
        )
        new_title = (f"{article_title.group(1).strip()} - {lang_config['title']}" 
                    if article_title else lang_config['title'])
        full_content = re.sub(
            r'<title>.*?</title>', 
            f'<title>{new_title}</title>', 
            full_content, 
            flags=re.DOTALL
        )

        # Clean up headlines
        pattern = r'(<(?:div class="row">.*?<div|(?:div|span)[^>]*class="(?:headline|editor)"[^>]*>))(.*?)(</(?:div|span)>)'
        full_content = re.sub(
            pattern,
            lambda m: m.group(1) + re.sub(r'\s*\[.*?\]', '', m.group(2), flags=re.DOTALL) + m.group(3),
            full_content,
            flags=re.DOTALL
        )

        # Adjust paths and structure
        full_content = adjust_paths_after_translation(
            full_content, lang_config['html'], PATH_MAPPING_CONFIG['source_paths']
        )

        return full_content

    except Exception as error:
        raise

def translate_website(language: str, api_key: str) -> None:
    """
    Main function for translating entire website to target language
    Handles file processing, logging, and error recovery
    """
    # Get language configuration
    lang_config = LANGUAGE_CONFIG['languages'].get(language)
    if not lang_config:
        raise ValueError(f"Unsupported language: {language}")

    # Setup logging
    logger = setup_logger(lang_config['html_code'])
    set_verbose_mode(logger, TRANSLATION_CONFIG['verbose'])
    
    # Initialize API client
    api_client = TranslationAPIClient(api_key)

    # Setup target directory
    translated_website_dir = os.path.join(
        TRANSLATION_CONFIG['source_dir'], 
        lang_config['html_code']
    )
    os.makedirs(translated_website_dir, exist_ok=True)

    # Get and sort files for translation
    files = [f for f in os.listdir(TRANSLATION_CONFIG['source_dir']) 
             if f.endswith(('.html', 'add.js'))]
    files_without_numbers = [f for f in files 
                           if not any(char.isdigit() for char in f)]
    files = files_without_numbers + [f for f in files 
                                   if f not in files_without_numbers]

    logger.info(f"Translating {TEST_FILE_COUNT if MODE == 'test' else len(files)} files to {language}. "
                f"RECOVERY: {TRANSLATION_CONFIG['recovery_mode']}")
    
    # Initialize progress bar
    pbar = tqdm(total=len(files), ncols=70)

    files_processed, error_count = 0, 0
    
    for file in files:
        source_file_path = os.path.join(TRANSLATION_CONFIG['source_dir'], file)
        
        # Apply path replacements to file names            
        for old_path, new_path in PATH_MAPPING_CONFIG['source_paths'].items():
            file = file.replace(old_path, new_path)
        
        target_file = os.path.join(translated_website_dir, file)

        # Skip already translated files
        if (os.path.exists(target_file) and 
            os.path.getsize(target_file) > 0 and 
            not os.path.exists(f"{target_file.removesuffix('.html')}.partial.html")):
            pbar.update(1)
            continue

        try:
            logger.info(f"\n\n===== {target_file} =====")
            
            with open(source_file_path, mode='r', encoding='utf-8') as f:
                content = f.read()

            if str(target_file).endswith('.html'):
                # Translate HTML file
                destination_path = target_file.removesuffix(".html")
                translated_content = translate_html_file(
                    content, language, lang_config, destination_path, api_client
                )
                
                # Clean up temporary files if translation complete
                if not TRANSLATION_CONFIG['recovery_mode'] or "<incomplete-translation>" not in translated_content:
                    cleanup_files = [
                        f"{destination_path}.json",
                        f"{destination_path}.json.old",
                        f"{destination_path}.partial.html"
                    ]
                    for cleanup_file in cleanup_files:
                        if os.path.exists(cleanup_file):
                            os.remove(cleanup_file)
                else:
                    msg = f"Recovery in file {file}: Incomplete Translation"
                    logger.error(f"{msg} - - - \n...{translated_content[-800:translated_content.rfind('</incomplete-translation>')]}+\n\n")
                    subprocess.run(["espeak", msg])
                    error_count += 1
                    
            elif str(target_file).endswith('add.js'):
                # Process JavaScript file
                translated_content = adjust_paths_after_translation(
                    content.replace(LANGUAGE_CONFIG['source_text']['website_name'], 
                                  lang_config['title'])
                          .replace(LANGUAGE_CONFIG['source_text']['more_text'], 
                                 lang_config['more']),
                    lang_config['html_code'],
                    PATH_MAPPING_CONFIG['source_paths']
                )

            # Save translated content
            with open(target_file, mode='w', encoding='utf-8') as f:
                f.write(translated_content)
            pbar.update(1)

        except Exception as error:
            if "!Stop!" in str(error):
                raise
            msg = f"Failure processing file {file}: {str(error)}"
            logger.error(f")-; {msg} )-;\n\n")
            subprocess.run(["espeak", msg])
            error_count += 1
            
            # Stop if too many errors
            if error_count > max(files_processed // 8, 4):
                raise RuntimeError('O-: Too many translation errors: ' + str(error_count))
            
        finally:
            files_processed += 1
            if MODE == 'test' and files_processed >= TEST_FILE_COUNT:
                break

    pbar.close()
    msg = f'Finished {files_processed} files in {language} with {error_count} failures'
    logger.info("\n\n" + msg + "\n")
    subprocess.run(["espeak", msg])

def main():
    """Entry point for the translation process"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Website Translator')
    parser.add_argument('api_key', help='Anthropic API key')
    parser.add_argument('language', help='Target language code (e.g., "en", "fr")')
    args = parser.parse_args()

    try:
        translate_website(args.language, args.api_key)
    except Exception as error:
        msg = f'Error in main function: {str(error)}'
        logging.error(msg)
        subprocess.run(["espeak", msg])

if __name__ == "__main__":
    main()