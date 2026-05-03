import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE ----------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    layout="centered"
)

st.title("📄 AI Resume Analyzer")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- SKILL DATABASE ----------------
SKILL_DB = {
    "software": [
        "python", "java", "c++", "sql",
        "html", "css", "javascript",
        "react", "nodejs", "django",
        "flask", "mongodb", "git"
    ],

    "data_science": [
        "python", "machine learning",
        "deep learning", "tensorflow",
        "pandas", "numpy", "sql",
        "tableau", "power bi",
        "statistics", "data analysis"
    ],

    "ece": [
        "embedded systems", "iot",
        "vlsi", "verilog",
        "matlab", "arduino",
        "microcontrollers"
    ]
}

# ---------------- PDF TEXT EXTRACTION ----------------
def extract_text_from_pdf(uploaded_file):

    text = ""

    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)

        for page in pdf_reader.pages:

            extracted_text = page.extract_text()

            if extracted_text:
                text += extracted_text + " "

    except Exception as e:
        st.error(f"Error reading PDF: {e}")

    return text.lower()

# ---------------- DOMAIN DETECTION ----------------
def detect_domain(text):

    scores = {}

    for domain, skills in SKILL_DB.items():

        score = sum(
            1 for skill in skills
            if skill in text
        )

        scores[domain] = score

    return max(scores, key=scores.get)

# ---------------- SKILL EXTRACTION ----------------
def extract_skills(text, domain):

    text = text.lower()

    skills = SKILL_DB[domain]

    matched = [
        skill for skill in skills
        if skill in text
    ]

    missing = [
        skill for skill in skills
        if skill not in text
    ]

    return matched, missing

# ---------------- MATCH SCORE ----------------
def calculate_similarity(resume, job):

    embeddings = model.encode([resume, job])

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return round(similarity * 100, 2)

# ---------------- ANALYSIS ----------------
def analyze_resume(resume_text, job_text):

    combined_text = resume_text + " " + job_text

    domain = detect_domain(combined_text)

    matched_skills, missing_skills = extract_skills(
        resume_text,
        domain
    )

    semantic_score = calculate_similarity(
        resume_text,
        job_text
    )

    total_skills = len(SKILL_DB[domain])

    skill_score = (
        len(matched_skills) / total_skills
    ) * 100

    final_score = round(
        (semantic_score * 0.7) +
        (skill_score * 0.3),
        2
    )

    resume_rating = round(
        final_score / 10,
        1
    )

    return {
        "domain": domain,
        "match_score": final_score,
        "resume_rating": resume_rating,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    }

# ---------------- UI ----------------
resume_pdf = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description"
)

# ---------------- BUTTON ----------------
if st.button("Analyze Resume"):

    if resume_pdf is not None and job_description:

        with st.spinner("Analyzing Resume..."):

            # Extract resume text
            resume_text = extract_text_from_pdf(
                resume_pdf
            )

            if resume_text.strip() == "":

                st.error(
                    "Could not extract text from PDF."
                )

            else:

                # Analyze
                result = analyze_resume(
                    resume_text,
                    job_description.lower()
                )

                # ---------------- OUTPUT ----------------
                st.success(
                    "Resume Analysis Completed ✅"
                )

                st.subheader("📊 Analysis Result")

                st.write(
                    f"### 🎯 Match Percentage: {result['match_score']}%"
                )

                st.write(
                    f"### ⭐ Resume Score: {result['resume_rating']} / 10"
                )

                st.write(
                    f"### 🧠 Detected Domain: {result['domain']}"
                )

                st.subheader("✅ Matched Skills")
                st.write(
                    result["matched_skills"]
                )

                st.subheader("❌ Missing Skills")
                st.write(
                    result["missing_skills"]
                )

                st.subheader("💡 Suggested Skills To Add")

                if result["missing_skills"]:

                    for skill in result["missing_skills"]:
                        st.write(f"• {skill}")

                else:
                    st.write(
                        "No major skills missing."
                    )

    else:
        st.warning(
            "Please upload resume PDF and enter job description."
        )