#!/bin/bash
echo "🚀 Setting up PikaPlexity environment..."
pip install -r requirements.txt
echo "✅ Dependencies installed!"
echo "🌐 Starting Streamlit app..."
streamlit run app.py
