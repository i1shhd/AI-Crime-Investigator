import spaces
import warnings
warnings.filterwarnings("ignore")

import random
from datetime import datetime

import torch
from transformers import pipeline
import gradio as gr

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging

logging.set_verbosity_error()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LLM_ID = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(LLM_ID)

llm = AutoModelForCausalLM.from_pretrained(
    LLM_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
)

@spaces.GPU
def transcribe(audio):
    whisper_pipe.model.to("cuda")
    result = whisper_pipe(audio)
    whisper_pipe.model.to("cpu")
    return result["text"]
    

def chat(prompt, system=None, max_new_tokens=200):
    messages = []

    if system:
        messages.append({"role": "system", "content": system})

    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(llm.device)

    outputs = llm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    response = (
        response.replace("**", "")
        .replace("###", "")
        .replace("##", "")
        .replace("CASE SUMMARY", "CASE SUMMARY\n----------------------------------------")
        .replace("EVIDENCE MATCH", "EVIDENCE MATCH\n----------------------------------------")
        .replace("POSSIBLE CONTRADICTIONS", "POSSIBLE CONTRADICTIONS\n----------------------------------------")
        .replace("RECOMMENDED NEXT EVIDENCE", "RECOMMENDED NEXT EVIDENCE\n----------------------------------------")
        .replace("INITIAL INVESTIGATION INSIGHT", "INITIAL INVESTIGATION INSIGHT\n----------------------------------------")
    )

    return response.strip()


def speech_to_text(audio):
    if audio is None:
        return "No audio uploaded."

    text =transcribe(audio)
    
    if len(text.strip()) < 5:
        return "Audio quality is too low."

    return text


def create_case_id():
    number = random.randint(1000, 9999)
    return f"INV-{datetime.now().year}-{number}"


def analyze_case(case_notes, witness_text):

    if not case_notes or len(case_notes.strip()) < 10:
        return None, "Missing case details."

    prompt = f"""
You are an AI crime investigation assistant.
Case File:
{case_notes}
Witness Statement:
{witness_text}
Analyze ONLY the information provided.
Generate a professional preliminary investigation report in the EXACT format below.
> CASE SUMMARY
==================================================
Write a concise summary of the case.
> EVIDENCE MATCH
==================================================
List the main agreements between the witness statement and the case file.
> POSSIBLE CONTRADICTIONS
==================================================
List any inconsistencies between the witness's statement and the case file.
If none exist, write:
No significant contradictions were identified.
> RECOMMENDED NEXT EVIDENCE
==================================================
Recommend 2–3 additional investigation steps or pieces of evidence that would strengthen the case.
> INITIAL INVESTIGATION INSIGHT
==================================================
Provide a brief professional conclusion based only on the available evidence.
Rules:
- Compare the witness statement directly with the case file.
- If the witness denies involvement, location, actions, or contact with evidence while the case file indicates otherwise, mark it as a contradiction.
- Do not say "No significant contradictions were identified" if there is a conflict between the witness statement and case evidence.
- Lower the evidence match confidence when the witness statement conflicts with available evidence.
- Do not invent facts.
- Keep the report concise.
- Use bullet points where appropriate.
- Do not use Markdown (#, ##, **, or *).
"""

    analysis = chat(prompt)

    confidence_prompt = f"""
Based on this case analysis:
{analysis}
Give a confidence percentage from 0-100.
Consider contradictions between witness statements and case evidence.
A strong contradiction should significantly reduce the confidence score.
Return only the number.
"""

    confidence = chat(
        confidence_prompt,
        max_new_tokens=10
    )

    return analysis, confidence

@spaces.GPU
def run_investigation(audio, notes):

    if audio is None:
        return (
            "",
            "⚠️ No audio provided.",
            "",
            ""
        )

    case_id = create_case_id()

    transcript = speech_to_text(audio)

    analysis, confidence = analyze_case(
        notes,
        transcript
    )

    if analysis is None:
        return (
            "",
            transcript,
            confidence,
            ""
        )

    confidence = confidence.replace("%", "").strip()

    return (
        case_id,
        transcript,
        analysis,
        f"{confidence}%"
    )


CUSTOM_CSS = """
/* ===========================
   AI Crime Investigator Theme
   =========================== */
/* Background */
.gradio-container{
    background: linear-gradient(135deg, #050505, #220000, #050505) !important;
    color:white !important;
    font-family:"Segoe UI", sans-serif;
}
/* Main Title */
h1{
    color:white !important;
    text-align:center !important;
    font-size:42px !important;
    font-weight:800 !important;
    text-shadow:0 0 15px rgba(255,0,0,.5);
}
h2,h3{
    color:#ffcccc !important;
    font-weight:700 !important;
}
/* Cards */
.block{
    background:#181818 !important;
    border:2px solid #8B0000 !important;
    border-radius:18px !important;
    padding:16px !important;
    box-shadow:0 0 18px rgba(180,0,0,.35) !important;
}
/* New Investigation + Report Cards */
.investigation-card,
.report-card{
    background:#181818 !important;
    border:2px solid #8B0000 !important;
    border-radius:20px !important;
    padding:20px !important;
    box-shadow:0 0 20px rgba(180,0,0,.35) !important;
}
/* Inputs */
textarea,
input{
    background:#111111 !important;
    color:white !important;
    border:2px solid #8B0000 !important;
    border-radius:14px !important;
}
textarea:focus,
input:focus{
    border:2px solid #ff3333 !important;
    box-shadow:0 0 15px rgba(255,0,0,.5) !important;
}
/* Audio */
.upload-container,
.file-preview{
    background:#111111 !important;
    border:2px dashed #B30000 !important;
    border-radius:15px !important;
}
/* Buttons */
button{
    background:#B30000 !important;
    color:white !important;
    border:none !important;
    border-radius:14px !important;
    font-size:17px !important;
    font-weight:bold !important;
    transition:.25s !important;
}
button:hover{
    background:#E00000 !important;
    transform:scale(1.04);
    box-shadow:0 0 20px rgba(255,0,0,.6) !important;
}
/* Labels */
label{
    color:white !important;
    font-weight:700 !important;
}
/* Examples */
.examples{
    background:#181818 !important;
    border:2px solid #8B0000 !important;
    border-radius:15px !important;
}
/* Markdown */
hr{
    border:1px solid #8B0000 !important;
}
/* Hover glow */
.block:hover{
    box-shadow:0 0 20px rgba(255,0,0,.25) !important;
}
"""


with gr.Blocks(
    css=CUSTOM_CSS,
    theme=gr.themes.Base(),
    title="AI Crime Investigator"
) as demo:


    gr.Markdown(
"""
# 🕵🏻‍♀️ AI Crime Investigator
### AI-powered Case Analysis Assistant
Analyze . Connect . Discover
An AI assistant that analyzes witness statements and case details
to uncover connections, detect inconsistencies, and provide preliminary investigation insights.
---
"""
)


    with gr.Group(elem_classes="investigation-card"):

        gr.Markdown(
"""
## 🔍 New Investigation
Start analyzing a new case.
"""
)


        case_notes=gr.Textbox(
            label="📄 Incident Notes",
            placeholder="Enter case details..."
        )


        audio=gr.Audio(
            type="filepath",
            label="🎙 Witness Audio"
        )


        gr.Examples(
            examples=[
              [
                  "examples/witness1.wav",
                  "A white sedan was stolen from a public parking lot near Najran Library at 9:15 PM."
              ],
              [
                  "examples/witness2.wav",
                  "A pharmacy in Najran was robbed. Surveillance cameras captured a witness entering the pharmacy and found his fingerprints, making him the prime suspect."
              ]
          ],
          inputs=[audio, case_notes]
        )


        analyze_btn=gr.Button(
            "🔴 Analyze Case",
            variant="primary"
        )


    with gr.Group(elem_classes="report-card"):

        gr.Markdown("# 📁 Investigation Report")


        case_id_box = gr.Textbox(
            label="📁 Case ID",
            interactive=False
        )


        transcript=gr.Textbox(
            label="🎙 Witness Transcript",
            lines=5
        )


        analysis=gr.Textbox(
            label="🕵🏻‍♀️ Investigation Report",
            lines=10,
            interactive=False
        )


        confidence=gr.Textbox(
            label="📊 Evidence Confidence Score",
            lines=3
        )


    analyze_btn.click(
        run_investigation,
        inputs=[
            audio,
            case_notes
        ],
        outputs=[
            case_id_box,
            transcript,
            analysis,
            confidence
        ]
    )


demo.launch()
