import os
import json
import base64
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader

# ------------------ ENV SETUP ------------------
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY must be set in .env")

# ------------------ LLM INIT ------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1
)

# ------------------ FILE TYPE ------------------
def get_file_type(file_path: str) -> str:
    ext = file_path.lower().split(".")[-1]
    if ext == "pdf":
        return "pdf"
    if ext in ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]:
        return "image"
    raise ValueError(f"Unsupported file type: .{ext}")

# ------------------ PDF TEXT ------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return "\n\n".join(page.page_content for page in pages)

# ------------------ IMAGE OCR (VISION) ------------------
def extract_text_from_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    vision_prompt = """
You are analyzing a veterinary EMR or lab report image for DOGS or CATS.

Rules:
- Extract ONLY abnormal findings explicitly visible
- Ignore ALL normal values
- Abnormal indicators include: High, Low, H, L, *, flags, out-of-range values
- Preserve names and values exactly as written
- Do NOT infer, diagnose, or summarize
- Output ONLY plain text (no headers, no explanations)
"""

    response = llm.invoke([
        {
            "role": "user",
            "content": [
                {"type": "text", "text": vision_prompt},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            ]
        }
    ])

    return response.content.strip()

# ------------------ FILE PROCESSOR ------------------
def process_file(file_path: str) -> str:
    file_type = get_file_type(file_path)
    if file_type == "pdf":
        return extract_text_from_pdf(file_path)
    return extract_text_from_image(file_path)

# ------------------ EMR ANALYSIS ------------------
def analyze_emr_file(file_path: str) -> dict:
    raw_text = process_file(file_path)

    prompt = f"""
You are an AI system analyzing veterinary EMR or lab reports
for DOGS and CATS only.

Your task:
- Identify ONLY abnormal findings explicitly present
- Extract clinical notes if written
- Assess triage risk ONLY from provided text
- Do NOT hallucinate
- Use null when information is missing
- Return VALID JSON ONLY (no markdown, no comments)

Return JSON with EXACTLY this schema:
{{
  "abnormal_findings": [
    {{
      "parameter": string,
      "issue": string
    }}
  ],
  "triage_signals": {{
    "potential_emergency": boolean,
    "reasons": [string]
  }},
  "overall_summary": string | null
}}

RAW EMR / LAB TEXT:
{raw_text}
"""

    response = llm.invoke(prompt)
    text = response.content

    # Defensive JSON extraction
    if "```" in text:
        text = text.split("```")[1]

    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end == -1:
        raise ValueError("No JSON object returned by model")

    return json.loads(text[start:end])

# ------------------ CLI / DEBUG ------------------
if __name__ == "__main__":
    test_file = "sample_dog_emr.pdf"  # change for local testing
    result = analyze_emr_file(test_file)
    print(json.dumps(result, indent=2))

