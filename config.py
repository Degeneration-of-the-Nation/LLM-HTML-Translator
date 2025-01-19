"""
Configuration file for Multilingual HTML Translator
Original site: https://hitdarderut-haaretz.org
Translated versions: https://degeneration-of-nation.org

This file contains all configurable parameters for the translation system,
including language settings, API configurations, and parsing parameters.
"""

# API Configuration
API_CONFIG = {
    'model': 'claude-3-5-sonnet-20241022',
    'old_model': 'claude-3-5-sonnet-20240620',
    'headers': {"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15,prompt-caching-2024-07-31"}
}

# Translation Process Configuration
TRANSLATION_CONFIG = {
    'chunk_max_size': 15000,
    'chunk_overlap': 1200,
    'overlap_parting': 2,
    'max_retries': 3,
    'sleep_time': 5,
    'min_tokens': 2000,
    'max_tokens': 8192,
    'default_temperature': 0.3,
    'recovery_mode': False,
    'verbose': True,
    'source_dir': 'website2',
    'target_dir_template': 'website2/{lang}'  # {lang} will be replaced with language code
}

# Language Settings
LANGUAGE_CONFIG = {
    # Basic language properties
    'default_multiplier': 1.2,
    'rtl_languages': ['he', 'ar'],
    'asian_languages': ['ja', 'ko', 'zh'],
    
    # Unicode ranges for validation
    'hebrew_chars': {
        'ranges': [
            (0x05D0, 0x05EA),  # Hebrew letters
            (0x05F0, 0x05F2)   # Hebrew ligatures
        ]
    },
    
    # Source language text replacements
    'source_text': {
        'website_name': "התדרדרות הארץ",
        'more_text': "עוד",
        'about_text': "אודות"
    },
    
    # Text splitting configuration for chunking
    'text_separators': [
        "<br><br><div>",
        "<br><div>",
        "<div>",
        "<br><br><b>",
        "<br><b>",
        "<br><br><br>",
        "<br><br>",
        "<br>",
        ". ",
        "? ",
        ", ",
        " "
    ],

    # Language configurations
    'languages': {
        'en': {
            'name': "English",
            'html_code': "en",
            'title': "The Degeneration of the Nation",
            'more': "More",
            'about': "About",
            'multiplier': 1.2,
            'future': "Translation will be completed in the future",
            'translator_note': "Translator's note",
            'example': "Hello&nbsp;world<br><b>Title"
        }
        # Add other languages as needed
    }
}

# Path mapping configuration
PATH_MAPPING_CONFIG = {
    'source_paths': {
        'source_section': 'translated_section',  # Example: 'blog': 'en-blog'
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'format': '%(asctime)s %(levelname)s %(message)s',
    'datefmt': '%H:%M',
    'level': 'INFO',
    'log_file_suffix': 'translation_log'
}

EXECUTION_CONFIG = {
    'mode': 'full',  # or 'test'
    'test_file_count': 1
}