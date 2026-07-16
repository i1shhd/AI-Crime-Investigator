# AI-Crime-Investigator
An AI-powered crime investigation assistant that analyzes witness statements and case files to generate a structured preliminary investigation report.

## Overview

AI Crime Investigator is a multimodal AI application that combines speech recognition and natural language understanding to assist in preliminary crime investigations. The system transcribes witness audio, compares it with the provided case file, detects potential inconsistencies, and generates a professional investigation report.

## Features

- Speech-to-text transcription of witness statements.
- AI-powered case file analysis.
- Evidence consistency and contradiction detection.
- Structured preliminary investigation report.
- Investigation confidence score.
- Automatically generated Case ID.
- Interactive Gradio web interface.
- Built-in example cases for quick testing.

## AI Models

- **Whisper Small** – Speech Recognition
- **Qwen2.5-1.5B-Instruct** – Investigation Report Generation

## Workflow

1. Upload a witness audio recording.
2. Enter the case details.
3. Transcribe the witness statement using Whisper.
4. Compare the transcript with the case file.
5. Generate a structured investigation report and confidence score.

## Project Structure

```
app.py
requirements.txt
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

## Future Improvements

- Image evidence analysis.
- Multi-witness comparison.
- Timeline reconstruction.
- Relationship mapping between suspects and evidence.
- Retrieval-Augmented Generation (RAG) for external case knowledge.

## Disclaimer

This project is developed for educational and research purposes to demonstrate the use of multimodal AI in preliminary crime investigation workflows. The generated reports are intended to assist analysis and should not be considered legal or forensic conclusions.
