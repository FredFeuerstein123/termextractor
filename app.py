import streamlit as st
import spacy
import requests
import json
import re
from collections import Counter
import pandas as pd

# App Configuration
st.set_page_config(
    page_title="Terminology Extractor",
    page_icon="🔍",
    layout="wide"
)

# Initialize Session State
if 'extracted_terms' not in st.session_state:
    st.session_state.extracted_terms = []

@st.cache_resource
def load_spacy_model():
    """Loads the spaCy model (cached for better performance)"""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy model 'en_core_web_sm' not found. Please install it with: python -m spacy download en_core_web_sm")
        return None

def extract_terminology(text, nlp):
    """Extracts potential technical terms from the text"""
    doc = nlp(text)
    
    # Different strategies for terminology extraction
    terms = set()
    
    # 1. Named Entities (Organizations, Products, etc.)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "EVENT", "LAW", "LANGUAGE"]:
            terms.add(ent.text.strip())
    
    # 2. Noun Phrases
    for chunk in doc.noun_chunks:
        # Filter out short and common words
        if len(chunk.text.split()) >= 2 and len(chunk.text) > 3:
            # Remove articles and pronouns at the beginning
            cleaned = re.sub(r'^(the|a|an|this|that|these|those)\s+', '', chunk.text.lower())
            if cleaned and len(cleaned) > 3:
                terms.add(chunk.text.strip())
    
    # 3. Compound Terms
    compound_patterns = [
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',  # Multiple capitalized words
        r'\b[a-z]+(?:-[a-z]+)+\b',  # Hyphenated compounds
        r'\b[A-Z]{2,}\b'  # Acronyms
    ]
    
    for pattern in compound_patterns:
        matches = re.findall(pattern, text)
        terms.update(matches)
    
    return list(terms)

def filter_against_glossary(terms, glossary_text):
    """Filters terms that are already present in the glossary"""
    if not glossary_text.strip():
        return terms
    
    # Glossary in lowercase for comparison
    glossary_lower = glossary_text.lower()
    glossary_terms = set()
    
    # Extract terms from glossary (line by line)
    for line in glossary_text.split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):  # Ignore comments
            # Extract first term before colon or tab
            term = re.split(r'[:\t]', line)[0].strip()
            if term:
                glossary_terms.add(term.lower())
    
    # Filter already existing terms
    new_terms = []
    for term in terms:
        if term.lower() not in glossary_terms:
            # Additional check for substrings
            is_in_glossary = any(term.lower() in existing.lower() or existing.lower() in term.lower() 
                               for existing in glossary_terms)
            if not is_in_glossary:
                new_terms.append(term)
    
    return new_terms

def translate_with_deepl(terms, api_key, target_lang='DE'):
    """Translates terms using DeepL API"""
    if not api_key or not terms:
        return {}
    
    translations = {}
    base_url = "https://api-free.deepl.com/v2/translate"
    
    # DeepL API supports batch translation
    try:
        headers = {
            "Authorization": f"DeepL-Auth-Key {api_key}",
            "Content-Type": "application/json"
        }
        
        # Split large lists (DeepL limit)
        batch_size = 50
        for i in range(0, len(terms), batch_size):
            batch = terms[i:i+batch_size]
            
            data = {
                "text": batch,
                "target_lang": target_lang,
                "source_lang": "EN"
            }
            
            response = requests.post(base_url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                for j, translation in enumerate(result['translations']):
                    if i+j < len(terms):
                        translations[terms[i+j]] = translation['text']
            else:
                st.error(f"DeepL API Error: {response.status_code}")
                break
                
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
    
    return translations

# Main Interface
st.title("🔍 Terminology Extractor")
st.markdown("Extracts technical terms from English texts and compares them with a glossary.")

# Sidebar for Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # DeepL API Key
    deepl_api_key = st.text_input(
        "DeepL API Key (optional)",
        type="password",
        help="For automatic translation of extracted terms"
    )
    
    target_language = st.selectbox(
        "Target language for translation",
        ["DE", "FR", "ES", "IT", "NL", "PL", "RU"],
        help="Language for DeepL translation"
    )
    
    min_term_length = st.slider(
        "Minimum term length",
        min_value=2,
        max_value=10,
        value=4,
        help="Minimum number of characters for extracted terms"
    )

# Main Area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📄 Technical Text")
    input_text = st.text_area(
        "Enter English technical text:",
        height=300,
        placeholder="Insert your English technical text here..."
    )

with col2:
    st.header("📚 Glossary")
    glossary_text = st.text_area(
        "Existing glossary (one term per line):",
        height=300,
        placeholder="Term 1\nTerm 2: Definition\nTerm 3\t Definition..."
    )

# Processing
if st.button("🚀 Extract Terms", type="primary"):
    if not input_text.strip():
        st.error("Please enter some text.")
    else:
        # Load spaCy model
        nlp = load_spacy_model()
        if nlp is None:
            st.stop()
        
        with st.spinner("Extracting terminology..."):
            # Step 1: Extract terms
            all_terms = extract_terminology(input_text, nlp)
            
            # Step 2: Filter by minimum length
            filtered_terms = [term for term in all_terms if len(term) >= min_term_length]
            
            # Step 3: Compare against glossary
            new_terms = filter_against_glossary(filtered_terms, glossary_text)
            
            # Step 4: Remove duplicates and sort
            unique_terms = sorted(list(set(new_terms)))
            
            st.session_state.extracted_terms = unique_terms

# Display Results
if st.session_state.extracted_terms:
    st.header("📊 Extracted Terms")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("New terms found", len(st.session_state.extracted_terms))
    with col2:
        total_terms = len(extract_terminology(input_text, load_spacy_model())) if input_text else 0
        st.metric("Total terms", total_terms)
    with col3:
        glossary_count = len([line for line in glossary_text.split('\n') if line.strip() and not line.startswith('#')])
        st.metric("Terms in glossary", glossary_count)
    
    # Terms as table
    df_terms = pd.DataFrame({
        'Term': st.session_state.extracted_terms,
        'Length': [len(term) for term in st.session_state.extracted_terms]
    })
    
    # Optional: Translation
    if deepl_api_key and st.button("🌐 Translate with DeepL"):
        with st.spinner("Translating terms..."):
            translations = translate_with_deepl(
                st.session_state.extracted_terms, 
                deepl_api_key, 
                target_language
            )
            
            if translations:
                df_terms['Translation'] = [
                    translations.get(term, "Not translated") 
                    for term in st.session_state.extracted_terms
                ]
    
    # Display table
    st.dataframe(df_terms, use_container_width=True)
    
    # Download options
    st.header("💾 Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV Download
        csv = df_terms.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="extracted_terms.csv",
            mime="text/csv"
        )
    
    with col2:
        # Text format for glossary
        glossary_format = "\n".join([f"{term}:\t" for term in st.session_state.extracted_terms])
        st.download_button(
            label="📝 As Glossary Template",
            data=glossary_format,
            file_name="new_glossary_terms.txt",
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.markdown("""
**Notes:**
- Install the spaCy model: `python -m spacy download en_core_web_sm`
- Get a free DeepL API key at [deepl.com](https://www.deepl.com/api)
- The app extracts Named Entities, Noun Phrases, and compound terms
""")
