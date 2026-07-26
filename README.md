# AI Crime Investigator

An AI-powered crime investigation assistant that leverages multimodal artificial intelligence to analyze witness statements and case files, generating structured preliminary investigation reports.

## Overview

AI Crime Investigator is a multimodal AI application that combines automatic speech recognition with a large language model to assist in preliminary crime investigations. The system transcribes witness audio, compares it with the provided case information, evaluates evidence consistency, identifies potential contradictions, and generates a structured investigation report with an investigation confidence score.

## Problem

Crime investigations involve analyzing witness statements alongside case information to support informed investigative decisions. Efficiently organizing these inputs into a clear, structured report helps streamline the preliminary investigation process and provides investigators with an overview of the available evidence.

## Solution

AI Crime Investigator automates this workflow by combining speech recognition with artificial intelligence. The application converts witness audio into text, compares it with the case file, evaluates evidence consistency, identifies possible contradictions, and generates a professional preliminary investigation report together with a confidence score.

## Features

- Speech-to-text transcription using Whisper Small.
- AI-powered case analysis using Qwen2.5.
- Evidence consistency evaluation.
- Contradiction detection.
- Structured investigation report generation.
- Investigation confidence score.
- Automatic Case ID generation.
- Interactive Gradio web interface.
- Built-in example cases for quick testing.

## AI Models

- **Whisper Small** – Speech-to-Text Transcription
- **Qwen2.5-1.5B-Instruct** – Investigation Analysis and Report Generation

## Workflow

1. Upload a witness audio recording.
2. Enter the case details.
3. Transcribe the witness statement.
4. Compare the transcript with the case file.
5. Generate a structured investigation report and confidence score.

## Project Structure

```text
app.py
requirements.txt
README.md
examples/
├── witness1.wav
└── witness2.wav
```

## Installation

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

## Technologies Used

- Python
- Gradio
- PyTorch
- Hugging Face Transformers
- Whisper
- Qwen2.5
- SoundFile

## Future Enhancements

- Image evidence analysis.
- Multi-witness comparison.
- Timeline reconstruction.
- Relationship mapping between suspects and evidence.
- Retrieval-Augmented Generation (RAG).

## Disclaimer

This project was developed for educational and research purposes to demonstrate the application of multimodal artificial intelligence in preliminary crime investigation. The generated reports are intended to assist analysis and should not be considered legal, forensic, or judicial conclusions.

## Developed By

**Shahad Alshaibani**
