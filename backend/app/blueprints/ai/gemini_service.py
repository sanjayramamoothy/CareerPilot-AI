import os
import json
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

# Retrieve API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
is_gemini_mock = not GEMINI_API_KEY or GEMINI_API_KEY.startswith("your-") or GEMINI_API_KEY.startswith("dummy")

if not is_gemini_mock:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Google Gemini SDK initialized successfully using GEMINI_API_KEY in gemini_service.")
    except Exception as e:
        logger.error(f"Failed to configure Google Gemini SDK: {e}")
        is_gemini_mock = True
else:
    logger.warning("GEMINI_API_KEY is not configured or is placeholder. Resume analysis will run in Mock AI Mode.")

REQUIRED_KEYS = [
    "resume_summary",
    "technical_skills",
    "soft_skills",
    "strengths",
    "weaknesses",
    "missing_skills",
    "improvements",
    "recommended_roles",
    "career_recommendations"
]

MOCK_ANALYSIS_RESULTS = {
    "resume_summary": "Highly motivated and results-oriented Software Engineer with 2+ years of experience building modern web applications. Proficient in Python, Flask, JavaScript, and React, with a strong focus on clean architecture, performance optimization, and scalable backend design.",
    "technical_skills": ["Python", "Flask", "JavaScript", "HTML5", "CSS3", "React", "SQL", "Git", "RESTful APIs"],
    "soft_skills": ["Problem-solving", "Collaboration", "Technical Communication", "Adaptability", "Teamwork"],
    "strengths": [
        "Strong foundation in web application development using Python and Flask.",
        "Demonstrated experience with modern front-end technologies (React, JavaScript).",
        "Proper repository structure and clean code practices implemented."
    ],
    "weaknesses": [
        "Lacks representation of automated testing frameworks (e.g. pytest, Jest).",
        "Limited exposure to containerization technologies (e.g. Docker, Kubernetes).",
        "No mention of CI/CD pipeline automation or cloud deployment models (AWS, GCP)."
    ],
    "missing_skills": ["TypeScript", "Docker", "pytest", "Jest", "CI/CD Pipelines", "AWS/GCP Cloud Deployments"],
    "improvements": [
        "Incorporate a dedicated 'Testing' subsection in technical skills and list unit testing libraries like pytest or Jest.",
        "Include cloud deployment tools (e.g., Docker, AWS) in your technical stack to demonstrate modern DevOps readiness.",
        "Rephrase project bullet points using the STAR methodology, focusing on quantifiable metrics (e.g., 'improved query performance by 25%')."
    ],
    "recommended_roles": ["Full-Stack Software Engineer", "Backend Developer", "Junior DevOps Specialist", "Application Developer"],
    "career_recommendations": [
        "Gain hands-on experience with Docker containerization and deploy a project to AWS or GCP to build cloud competency.",
        "Learn TypeScript to complement JavaScript skills and increase eligibility for senior Full-Stack roles.",
        "Contribute to open-source projects or build a portfolio project that implements a full CI/CD deployment pipeline."
    ]
}

def validate_analysis_json(data):
    """Validates if the dictionary contains all required keys and correct types."""
    if not isinstance(data, dict):
        return False
    for key in REQUIRED_KEYS:
        if key not in data:
            logger.warning(f"Validation failed: missing key '{key}'")
            return False
    # Check types
    if not isinstance(data["resume_summary"], str):
        logger.warning("Validation failed: 'resume_summary' is not a string")
        return False
    for k in REQUIRED_KEYS[1:]:
        if not isinstance(data[k], list):
            logger.warning(f"Validation failed: '{k}' is not a list")
            return False
        # Ensure all elements in the lists are strings
        if not all(isinstance(item, str) for item in data[k]):
            logger.warning(f"Validation failed: element in '{k}' is not a string")
            return False
    return True

def analyze_resume_text(resume_text):
    """
    Sends the resume text to Google Gemini 2.5 Flash and returns a validated JSON object.
    Includes a retry mechanism if the output is malformed or missing fields.
    """
    global is_gemini_mock
    if is_gemini_mock:
        logger.info("Gemini Service: Running in Mock Mode. Returning pre-configured analysis results.")
        return MOCK_ANALYSIS_RESULTS

    prompt = f"""
    You are an expert technical resume reviewer and career coach.
    Analyze the following resume text and generate detailed career insights.
    
    You MUST output a raw JSON object matching the exact structure described below.
    Do NOT wrap the JSON inside markdown formatting or ```json blocks. Return ONLY the raw valid JSON.
    Do NOT include any extra explanations, comments, or headers. Only the raw JSON.
    
    Expected JSON format:
    {{
      "resume_summary": "",
      "technical_skills": [],
      "soft_skills": [],
      "strengths": [],
      "weaknesses": [],
      "missing_skills": [],
      "improvements": [],
      "recommended_roles": [],
      "career_recommendations": []
    }}
    
    Guidance for each key:
    - resume_summary: A professional summary paragraph analyzing their background.
    - technical_skills: List of key programming languages, tools, databases found.
    - soft_skills: List of soft skills, communication, or leadership traits.
    - strengths: At least 3 specific achievements or strength points.
    - weaknesses: At least 3 weak points, gaps in skillset, or formatting flaws.
    - missing_skills: Industry-standard skills expected but missing.
    - improvements: Actionable resume improvement recommendations.
    - recommended_roles: Suitable technical or software engineering roles.
    - career_recommendations: Long-term career growth or certification recommendations.
    
    Resume Text:
    {resume_text}
    """

    for attempt in range(1, 3):
        logger.info(f"Gemini Service: Sending request to Gemini 2.5 Flash (Attempt {attempt}/2)...")
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            response_text = response.text.strip() if response.text else ""
            logger.info("Gemini Service: Received response from Gemini API.")
            
            # Parse response
            try:
                data = json.loads(response_text)
                if validate_analysis_json(data):
                    logger.info("Gemini Service: Successfully validated Gemini response JSON.")
                    return data
                else:
                    logger.warning(f"Gemini Service: Response JSON validation failed on attempt {attempt}.")
            except json.JSONDecodeError as jde:
                logger.error(f"Gemini Service: JSON decode error on attempt {attempt}: {jde}. Response was: {response_text}")
        except Exception as api_err:
            logger.error(f"Gemini Service: API invocation error on attempt {attempt}: {api_err}")
            # If it is an authentication/credentials error, switch to mock mode immediately to avoid useless retries
            if "invalid authentication credentials" in str(api_err).lower() or "401" in str(api_err) or "api_key" in str(api_err).lower():
                logger.warning("Gemini Service: Invalid credentials detected. Switching to Mock AI Mode.")
                is_gemini_mock = True
                return MOCK_ANALYSIS_RESULTS
            
        if attempt == 1:
            logger.info("Gemini Service: Retrying Gemini request due to validation or connection error...")
            
    # Fallback to mock mode if all attempts failed
    logger.warning("Gemini Service: Real API analysis failed. Falling back to Mock AI Mode.")
    is_gemini_mock = True
    return MOCK_ANALYSIS_RESULTS
