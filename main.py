from flask import Flask, request, render_template
import os
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Function to extract text from PDFs
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() if page.extract_text() else ""
    return text

# Function to extract text from DOCX files
def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

# Function to extract text from TXT files
def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to determine file type and extract text
def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return ""

# Preprocess text (remove special characters, convert to lowercase, remove stopwords)
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    stop_words = set(stopwords.words('english'))
    words = text.split()
    text = ' '.join([word for word in words if word not in stop_words])  # Remove stopwords
    return text

@app.route("/")
def matchresume():
    return render_template('matchresume.html')

@app.route('/matcher', methods=['POST'])  
def matcher():
    if request.method == 'POST':
        job_description = request.form.get('job_description')
        resume_files = request.files.getlist('resumes')

        if not resume_files or not job_description:
            return render_template('matchresume.html', message="Please upload resumes and enter a job description.")

        resumes = []
        resume_names = []

        # Process resumes
        for resume_file in resume_files:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(filename)
            resume_names.append(resume_file.filename)
            extracted_text = extract_text(filename)
            processed_text = preprocess_text(extracted_text)  # Preprocess resume text
            resumes.append(processed_text)

        # Preprocess job description
        job_description = preprocess_text(job_description)

        # Compute similarity using TF-IDF
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([job_description] + resumes)
        similarity_scores = cosine_similarity(vectors[0], vectors[1:])[0]

        # Sort results by highest similarity score
        sorted_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)
        top_resumes = [resume_names[i] for i in sorted_indices]
        top_scores = [round(similarity_scores[i] * 100, 2) for i in sorted_indices]  # Convert to percentage

        # Extract top 3 matching resumes
        top_3_resumes = top_resumes[:3]
        top_3_scores = top_scores[:3]

        return render_template('matchresume.html', message="Resume Matching Results", 
                               top_resumes=top_resumes, similarity_scores=top_scores, 
                               top_3_resumes=top_3_resumes, top_3_scores=top_3_scores)

    return render_template('matchresume.html', message="Please submit the form correctly.")

# Run Flask app
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
