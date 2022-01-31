"""
DeepFinNER: Streamlit Web Application
Advanced Financial Document Analysis with NLP and NER

This Streamlit app provides a user-friendly interface for:
- Financial document upload and analysis
- Named Entity Recognition visualization
- Sentiment analysis and text statistics
- Financial entity extraction
- Interactive document processing
"""

import streamlit as st
import spacy_streamlit
import spacy
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our custom modules
from nlp_processor import NLPProcessor
from entity_extractor import FinancialEntityExtractor

# Page configuration
st.set_page_config(
    page_title="DeepFinNER - Financial NLP Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .financial-entity {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize processors
@st.cache_resource
def load_processors():
    """Load NLP and entity extraction processors."""
    try:
        nlp_processor = NLPProcessor()
        entity_extractor = FinancialEntityExtractor()
        return nlp_processor, entity_extractor
    except Exception as e:
        st.error(f"Error loading processors: {e}")
        return None, None

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    """Load spaCy model for visualization."""
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.error("spaCy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm")
        return None

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 DeepFinNER</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">Financial NLP & Named Entity Recognition</h2>', unsafe_allow_html=True)
    
    # Load processors
    nlp_processor, entity_extractor = load_processors()
    spacy_model = load_spacy_model()
    
    if nlp_processor is None or entity_extractor is None or spacy_model is None:
        st.error("Failed to load required models. Please check the installation.")
        return
    
    # Sidebar
    st.sidebar.title("📋 Analysis Options")
    
    # Input method selection
    input_method = st.sidebar.selectbox(
        "Choose input method:",
        ["Text Input", "File Upload", "Sample Financial Text"]
    )
    
    # Text input
    if input_method == "Text Input":
        text_input = st.text_area(
            "Enter financial text for analysis:",
            height=200,
            placeholder="Paste your financial document text here..."
        )
    
    # File upload
    elif input_method == "File Upload":
        uploaded_file = st.file_uploader(
            "Upload a financial document:",
            type=["txt", "pdf", "docx"],
            help="Supported formats: TXT, PDF, DOCX"
        )
        
        if uploaded_file is not None:
            if uploaded_file.type == "text/plain":
                text_input = uploaded_file.getvalue().decode("utf-8")
            else:
                st.error("Please upload a text file for now. PDF and DOCX support coming soon!")
                text_input = ""
        else:
            text_input = ""
    
    # Sample text
    else:
        sample_text = """
        Apple Inc. reported revenue of $394.3 billion for fiscal year 2022, 
        representing an 8% increase compared to the previous year. The company's 
        net income reached $99.8 billion, driven by strong iPhone sales and 
        services growth. However, supply chain challenges and macroeconomic 
        uncertainty pose risks to future performance. The stock price closed at 
        $150.23 per share, with a PE ratio of 25.4.
        
        Sundar Pichai is the CEO of Google, and Tim Cook serves as Apple's CEO.
        """
        text_input = st.text_area(
            "Sample financial text:",
            value=sample_text,
            height=200
        )
    
    # Analysis section
    if text_input and text_input.strip():
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "🏷️ Named Entities", 
            "📈 Financial Analysis", 
            "📝 Text Statistics", 
            "🎯 Sentiment Analysis"
        ])
        
        with tab1:
            st.header("📊 Document Overview")
            
            # Basic statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Word Count", len(text_input.split()))
            
            with col2:
                st.metric("Character Count", len(text_input))
            
            with col3:
                st.metric("Sentence Count", len(text_input.split('.')))
            
            with col4:
                st.metric("Paragraph Count", len([p for p in text_input.split('\n\n') if p.strip()]))
            
            # Document preview
            st.subheader("📄 Document Preview")
            st.text_area("Preview:", text_input[:500] + "..." if len(text_input) > 500 else text_input, height=150)
        
        with tab2:
            st.header("🏷️ Named Entity Recognition")

            # spaCy visualization first
            st.subheader("spaCy NER Visualization")
            models = ["en_core_web_sm"]
            spacy_streamlit.visualize(models, text_input)

            # ---- Custom Extraction + Interactive Explorer ----
            st.subheader("💰 Financial Entity Extraction & Explorer")

            if st.button("Extract & Explore Entities"):
                with st.spinner("Processing entities ..."):
                    entities = entity_extractor.comprehensive_entity_extraction(text_input)

                # ---------- Entity Frequency Overview ----------
                st.markdown("### 📊 Entity Overview & Frequency")

                entity_records = []
                for ent_type, ent_list in entities.get('pattern_entities', {}).items():
                    for ent in ent_list:
                        count = len([m for m in ent_list if m == ent]) if isinstance(ent_list, list) else 1
                        entity_records.append({
                            'Entity': ent,
                            'Type': ent_type.replace('_', ' ').title(),
                            'Count': count
                        })

                if entity_records:
                    df_entities = pd.DataFrame(entity_records).drop_duplicates().sort_values('Count', ascending=False)
                    st.dataframe(df_entities, use_container_width=True)

                    # Bar chart for top 20
                    fig_ent = px.bar(df_entities.head(20), x='Entity', y='Count', color='Type', title='Top Entities')
                    fig_ent.update_layout(xaxis_tickangle=-45, height=400)
                    st.plotly_chart(fig_ent, use_container_width=True)

                # ---------- Risk Indicator Overview ----------
                st.markdown("### ⚠️ Risk Indicator Overview")
                risk_records = []
                for risk_type, terms in entities.get('risk_entities', {}).items():
                    for term in terms:
                        risk_records.append({
                            'Risk Term': term,
                            'Risk Type': risk_type.replace('_', ' ').title()
                        })

                if risk_records:
                    df_risk = pd.DataFrame(risk_records).drop_duplicates()
                    st.dataframe(df_risk, use_container_width=True)
                    if not df_risk.empty:
                        fig_risk = px.bar(df_risk.head(20), x='Risk Term', y=[1]*len(df_risk.head(20)), color='Risk Type', title='Risk Terms')
                        fig_risk.update_layout(yaxis_title='Presence', xaxis_tickangle=-45, height=300)
                        st.plotly_chart(fig_risk, use_container_width=True)

                # ---------- Quick-search Tags ----------
                st.markdown("### 🔍 Quick Search Tags")
                top_entities = df_entities.head(12).to_dict('records') if entity_records else []
                cols = st.columns(4)
                for i, ent_row in enumerate(top_entities):
                    col_idx = i % 4
                    with cols[col_idx]:
                        if st.button(ent_row['Entity'], key=f"quick_tag_{ent_row['Entity']}"):
                            st.session_state['search_term'] = ent_row['Entity']
        
        with tab3:
            st.header("📈 Financial Analysis")
            
            # Process document with NLP processor
            if st.button("Run Financial Analysis"):
                with st.spinner("Analyzing financial content..."):
                    results = nlp_processor.process_financial_document(text_input)
                    
                    # Financial metrics visualization
                    st.subheader("📊 Financial Metrics Distribution")
                    
                    # Create word frequency chart for financial terms
                    financial_words = [word for word, count in results['word_frequency'] 
                                     if any(term in word.lower() for term in ['revenue', 'profit', 'income', 'asset', 'debt', 'cash'])]
                    
                    if financial_words:
                        word_counts = Counter(financial_words)
                        fig = px.bar(
                            x=list(word_counts.keys()),
                            y=list(word_counts.values()),
                            title="Financial Terms Frequency",
                            labels={'x': 'Financial Terms', 'y': 'Frequency'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Key phrases
                    st.subheader("🔑 Key Financial Phrases")
                    if results['key_phrases']:
                        phrases_df = pd.DataFrame(results['key_phrases'][:10], columns=['Phrase', 'Score'])
                        st.dataframe(phrases_df, use_container_width=True)
        
        with tab4:
            st.header("📝 Text Statistics")
            
            if st.button("Calculate Text Statistics"):
                with st.spinner("Calculating text statistics..."):
                    results = nlp_processor.process_financial_document(text_input)
                    stats = results['text_statistics']
                    
                    # Display statistics in cards
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f'<div class="metric-card">📊 <strong>Word Count:</strong> {stats["word_count"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-card">📝 <strong>Sentence Count:</strong> {stats["sentence_count"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-card">📏 <strong>Avg Sentence Length:</strong> {stats["avg_sentence_length"]:.1f} words</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f'<div class="metric-card">🔤 <strong>Unique Words:</strong> {stats["unique_words"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-card">📊 <strong>Lexical Diversity:</strong> {stats["lexical_diversity"]:.3f}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-card">📏 <strong>Avg Word Length:</strong> {stats["avg_word_length"]:.1f} characters</div>', unsafe_allow_html=True)
        
        with tab5:
            st.header("🎯 Sentiment Analysis")
            
            if st.button("Analyze Sentiment"):
                with st.spinner("Analyzing sentiment..."):
                    results = nlp_processor.process_financial_document(text_input)
                    sentiment = results['sentiment_analysis']
                    
                    # Sentiment visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Sentiment Scores")
                        
                        # Polarity gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=sentiment['polarity'],
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Polarity"},
                            delta={'reference': 0},
                            gauge={
                                'axis': {'range': [-1, 1]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [-1, -0.1], 'color': "lightgray"},
                                    {'range': [-0.1, 0.1], 'color': "yellow"},
                                    {'range': [0.1, 1], 'color': "lightgreen"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 0
                                }
                            }
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("📈 Sentiment Details")
                        st.metric("Polarity", f"{sentiment['polarity']:.3f}")
                        st.metric("Subjectivity", f"{sentiment['subjectivity']:.3f}")
                        st.metric("Category", sentiment['sentiment_category'].title())
                        
                        # Sentiment interpretation
                        if sentiment['polarity'] > 0.1:
                            st.success("✅ Positive sentiment detected")
                        elif sentiment['polarity'] < -0.1:
                            st.error("❌ Negative sentiment detected")
                        else:
                            st.info("⚖️ Neutral sentiment detected")
    
    else:
        st.info("👆 Please enter some text or upload a file to begin analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🤖 DeepFinNER - Advanced Financial NLP & NER Pipeline</p>
        <p>Built with Streamlit, spaCy, and custom AI/ML techniques</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 