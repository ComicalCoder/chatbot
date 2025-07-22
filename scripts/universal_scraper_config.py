from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import re
import unicodedata # Added for normalize_text

# --- Helper Function (Moved from website_parse.py to avoid circular dependency) ---
def normalize_text(text: str) -> str:
    """Normalize text: fix whitespace and Unicode issues."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text

class WebsiteConfig(ABC):
    """Abstract base class for website-specific configurations"""
    
    @property
    @abstractmethod
    def base_url(self) -> str:
        pass
        
    @property
    @abstractmethod
    def name(self) -> str:
        pass
        
    @property
    def boilerplate_selectors(self) -> List[str]:
        """CSS selectors for elements to remove (navigation, ads, etc.)"""
        return [
            'nav', 'header nav', 'footer nav',
            '.advertisement', '.ads', '.sidebar',
            'script', 'style', 'noscript'
        ]
        
    @property
    def boilerplate_patterns(self) -> List[str]:
        """Regex patterns for text to remove"""
        return [
            r'Copyright.*?All rights reserved',
            r'Privacy Policy.*?Terms of Service',
            r'Follow us on.*?social media'
        ]
        
    @property
    def header_selectors(self) -> List[str]:
        return ['header', 'h1', '.page-title', '.main-title']
        
    @property
    def footer_selectors(self) -> List[str]:
        return ['footer', '.footer', '.page-footer']
        
    @property
    def main_content_selectors(self) -> List[str]:
        return ['main', '.main-content', '.content', 'article', '.article']
        
    @property
    def dropdown_selectors(self) -> List[str]:
        return [
            'details summary', '.accordion-toggle', '.dropdown-toggle',
            '.collapsible', '[data-toggle="collapse"]'
        ]
        
    @property
    def external_domains_to_skip(self) -> List[str]:
        return ['facebook.com', 'twitter.com', 'instagram.com', 'youtube.com']
        
    @property
    def ignored_extensions(self) -> tuple:
        return (
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.zip', '.rar', '.7z', '.tar', '.gz', '.jpg', '.jpeg', '.png',
            '.gif', '.bmp', '.svg', '.mp3', '.mp4', '.avi', '.mov', '.webp'
        )
        
    def extract_contact_info(self, page) -> Dict[str, str]:
        """Override this method for site-specific contact extraction"""
        return {}
        
    def extract_office_hours(self, page) -> Dict[str, str]:
        """Override this method for site-specific office hours extraction"""
        return {}
        
    def custom_content_extractors(self) -> List[callable]:
        """Return list of custom extraction functions"""
        return []

class SabahGovConfig(WebsiteConfig):
    """Configuration for Sabah Government website"""
    
    @property
    def base_url(self) -> str:
        return "https://jpkn.sabah.gov.my/"
        
    @property
    def name(self) -> str:
        return "Sabah Government"
        
    @property
    def boilerplate_selectors(self) -> List[str]:
        return super().boilerplate_selectors + [
            '.et_pb_menu', '.et_pb_social_media_follow',
            '.et_pb_login', '.et_pb_search',
            ".et_pb_section_0", # Specific Divi structural divs that often contain non-content
            ".et_pb_row_0",
            ".et_pb_text_11", ".et_pb_text_12", ".et_pb_text_14", ".et_pb_text_15", # Specific Divi text modules that contained links/JS
            "#google_translate_element", # Google Translate widget
            ".gtranslate_wrapper",
            ".glt-toolbar",
            ".ea11y-widget-container", # Accessibility widget
            "[class*='vanta']", # Vanta.js related elements (3D globe)
            "[class*='firebase']", # Firebase related content (visitor stats scripts)
            "[class*='wp-']", # WordPress related dynamic/meta content that isn't core text
            "[id*='wp-']",
        ]
        
    @property
    def boilerplate_patterns(self) -> List[str]:
        return super().boilerplate_patterns + [
            r'(?i)Info Korporat(?:.*?)(?:Takwim JPKN|Pelan Strategik|Sistem Pengurusan Kualiti)', # Large navigation blocks
            r'(?i)Tender dan Sebutharga(?:.*?)(?:Terbaharu|terkini|Pemenang Sebutharga)',
            r'(?i)Promosi Dan Hebahan(?:.*?)(?:Hebahan ICT|Sistem Aplikasi|Buletin JPKN|Pemakluman Pertukaran Pegawai)',
            r'(?i)Hubungi Kami(?:.*?)(?:Direktori|Maklum Balas|Alamat Cawangan|Pejabat Pasukan Inovasi Digital|Permohonan Latihan Industri|Aduan / Maklum balas|Sistem PAKSi|Helpdesk|Menara Kinabalu|Kajian Maklum Balas Perkhidmatan JPKN)',
            r'(?i)Select Page(?:.*?)(?:JPKN|Info Korporat)', # Common menu/header phrases
            r'(?i)Designed by Elegant Themes \| Powered by WordPress', # Footer credits
            r'(?i)NotificationsOriginal textRate this translationYour feedback will be used to help improve Google Translate', # Google Translate UI text
            r'(?i)Powered by Translatefunction GoogleLanguageTranslatorInit', # Google Translate JS init
            r'(?i)TranslateSelect Language(?:.*?)(?:Arabic|Chinese|English|French|German|Indonesian|Japanese|Javanese|Korean|Tamil)', # Google Translate language list
            r'(?i)\+6088 – 368 833 \(Helpdesk\)', # Specific phone numbers that appear as standalone boilerplate
            r'(?i)jpkn@sabah.gov.my', # Specific email that appears as standalone boilerplate
            r'(?i)PAKSi Sabah E-Circular', # Specific system name
            r'(?i)apiKey: \"AIzaSyAqpGN30mhbzGfy7oaNcpb0WNcWjgUUHhc\"', # Firebase API Key fragments
            r'const \w+ = .*?;', # General JS variable declarations
            r'var \w+ = .*?;', # General JS variable declarations
            r'function \w+\(.*?\) \{.*?\}', # General JS function definitions
            r'\( function\(\) \{.*?\} \)\(\);', # Immediately invoked function expressions
            r'\[.+?\];', # Array assignments
            r'\{.+?\};', # Object assignments
            r'\/\* <\!\[CDATA\[ \*\/.*?\/\* \]\]> \*\/', # CDATA blocks
            r'moment\.updateLocale\(\s*\'en_US\'.*?\);', # Specific JS library calls (moment.js)
            r'wp\.apiFetch\.use\(.*?\);', # WordPress API fetch setup
            r'wp\.date\.setSettings\(.*?\);', # WordPress date settings
            r'var et_animation_data = .*?;', # Divi theme animation data
            r'var et_link_options_data = .*?;', # Divi theme link options
            r'VANTA\.GLOBE\(.*?\)', # Vanta.js initialization
            r'©Powered By BayuCentric', # Specific footer text
            r'wp\.i18n\.setLocaleData\(.*?\);', # WordPress internationalization data
            r'var _wpUtilSettings = .*?;', # WordPress utility settings
            r'var ea11yWidget = .*?;', # Accessibility widget settings
            r'var DIVI = .*?;', # Divi theme global variables
            r'var et_builder_utils_params = .*?;', # Divi builder params
            r'var et_frontend_scripts = .*?;', # Divi frontend scripts
            r'var et_pb_custom = .*?;', # Divi custom settings
            r'var et_pb_box_shadow_elements = .*?;' # Divi box shadow elements
        ]
        
    def extract_contact_info(self, page) -> Dict[str, str]:
        """Sabah-specific contact information extraction"""
        contact_info = {}
        # Use the normalize_text function defined at the top of this file
        
        contact_block = page.locator('div.et_pb_text_inner:has-text("Contact Information")')
                
        if contact_block.count() > 0:
            block_text = normalize_text(contact_block.first.inner_text())
                        
            patterns = {
                "Alamat": r'Alamat: (.+?)(?:No\. Telefon|Emel|No\. Faks|$)',
                "No. Telefon": r'No\. Telefon: (.+?)(?:Emel|No\. Faks|$)',
                "Emel": r'Emel: (.+?)(?:No\. Faks|$)',
                "No. Faks": r'No\. Faks: (.+)'
            }
                        
            for key, pattern in patterns.items():
                match = re.search(pattern, block_text, re.DOTALL)
                if match:
                    contact_info[key] = normalize_text(match.group(1))
        return contact_info

    # Office hours are now static in agency_config.py, not scraped here
    def extract_office_hours(self, page) -> Dict[str, str]:
        """Office hours are handled by static data in agency_config.py, not scraped here."""
        return {}

class GenericConfig(WebsiteConfig):
    """Generic configuration for unknown websites"""
    
    def __init__(self, base_url: str, name: str = "Generic Site"):
        self._base_url = base_url
        self._name = name
        
    @property
    def base_url(self) -> str:
        return self._base_url
        
    @property
    def name(self) -> str:
        return self._name
        
    def extract_contact_info(self, page) -> Dict[str, str]:
        """Generic contact extraction using common patterns"""
        contact_info = {}
        # Use the normalize_text function defined at the top of this file
        
        # Look for common contact patterns        
        text_content = normalize_text(page.text_content())
                
        # Email patterns        
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text_content)
        if emails:
            contact_info['email'] = emails[0]
                
        # Phone patterns        
        phones = re.findall(r'(\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})', text_content)
        if phones:
            contact_info['phone'] = phones[0]
                
        return contact_info

# Factory function to get appropriate config
def get_website_config(url: str) -> WebsiteConfig:
    """Factory function to return appropriate configuration based on URL."""
    if 'jpkn.sabah.gov.my' in url: # Use the specific JPKN domain
        return SabahGovConfig()
    else:
        return GenericConfig(url)
