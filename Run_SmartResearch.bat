@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
title SmartResearch Chatbot AI
echo ========================================================
echo        Starting SmartResearch Chatbot (Agentic RAG)...
echo ========================================================
echo.

cd /d "%~dp0"

echo Opening browser at http://localhost:8501 ...
start "" http://localhost:8501

streamlit run app.py --server.headless true --server.maxUploadSize 5

pause
