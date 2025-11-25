# TalentScope – AI-Powered Technical Interview Practice System

TalentScope is an **AI-driven interview preparation platform** designed for students who feel *unprepared, anxious, or underconfident* before real technical interviews. This system provides a safe space to practice and improve using **AI-based evaluation, real-time feedback, voice & text responses, and adaptive questioning**.

This project includes a powerful **AI Interview Evaluation Model** built by me, which supports:

* Text Recognition (NLP answer evaluation)
* Voice Recognition (Speech-to-Text)
* Adaptive Difficulty
* Multi-domain training (AI, ML, DL, SE, OS, DBMS)
* Semantic understanding using advanced NLP models

A preview UI is available here:
**TalentScope Web App:** [https://talent-scope-rouge.vercel.app/](https://talent-scope-rouge.vercel.app/)

---

## 🚀 Overview

TalentScope is built to help students overcome fear and confusion during interviews by giving them:

* Realistic **AI-powered mock interviews**
* Personalized feedback
* Question difficulty that adapts to their performance
* A modern, clean UI for smooth practice

The system evaluates answers using:

* **Sentence Embeddings (MPNet)** – semantic understanding
* **Natural Language Inference (RoBERTa-MNLI)** – checks if user answer logically matches correct concepts
* **Synonym Expansion (WordNet)** – better understanding of varied vocabulary
* **(Optional) FastText Word Similarity** – token-level comparison
* **Weighted Hybrid Scoring System**

---

## 🎯 Key Features

### 🧠 AI-Based Answer Evaluation

* Understands answers beyond keywords
* Detects meaning, relevance, and correctness
* Supports long, conversational answers

### 🎤 Voice Recognition Model

* Users can **speak answers** instead of typing
* Real-time conversion from speech to text
* Smooth integration into interview UI

### 🎯 Adaptive Difficulty

* Easy → Medium → Hard progression
* Adjusts based on user performance
* Mimics real interview escalations

### 📚 Multi-Domain Support

Covers 6 major technical subjects:

* AI
* ML
* Deep Learning
* Software Engineering
* Operating Systems
* Database Management Systems

### ⭐ Smart Evaluation Metrics

Your answer gets evaluated on:

* Semantic similarity
* Word-level similarity
* Entailment probability
* Weighted numerical score
* Final verdict (Correct/Incorrect)

### 💬 User-Friendly Web UI

A modern frontend built for:

* Smooth navigation
* User-friendly interface
* Realistic interview experience

---

## 🏗️ System Architecture

```
User Input (Voice/Text)
        ↓
Speech-to-Text Engine (for voice)
        ↓
AI NLP Evaluation Model
   • Synonym Expansion
   • Sentence Embeddings
   • Word-Level Similarity
   • NLI Entailment Scoring
        ↓
Weighted Score Calculation
        ↓
Realtime Feedback to User
        ↓
Difficulty Adjustment
```

---

## ⚙️ Tech Stack

### **Backend / Model**

* Python 3.x
* SentenceTransformers (MPNet)
* HuggingFace Transformers (RoBERTa-MNLI)
* NLTK WordNet
* FastText (optional)
* Pandas
* Torch

### **Frontend**

* Next.js / React
* Tailwind CSS

### **Hosting**

* Vercel (Frontend)
* Python backend or cloud notebook (Model execution)

---

## 📦 Installation Instructions

### 1️⃣ Clone the Repo

```bash
git clone <your-repository-link>
cd talentscope
```

### 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the AI Evaluation Model

```bash
python main.py
```

---

## 🧠 How the Evaluation Model Works

### 🔍 1. Synonym Expansion

Helps understand meaning even if exact words differ.

### 🔍 2. Sentence Embedding Similarity

Checks semantic similarity using MPNet.

### 🔍 3. NLI Entailment

Evaluates logical correctness even without exact wording.

### 🔍 4. Weighted Hybrid Scoring

```
Final Score = 0.6 * Sentence Similarity
             + 0.3 * Word-Level Similarity
             + 0.1 * NLI Entailment Probability
```

### 🔍 5. Difficulty Adjustment

Difficulty increases when user answers correctly.

---

## 🎯 Target Users

* Students preparing for placements
* Beginners scared of interviews
* Self-learners wanting structured practice
* People wanting instant, unbiased feedback

---

## 🛣️ Roadmap

* [ ] Add detailed analytics dashboard
* [ ] Add coding round evaluator
* [ ] Add HR interview question module
* [ ] Add resume analysis AI
* [ ] Deploy model backend API

---

## 🤝 Contribution

Contributions are welcome! Feel free to create PRs or issues.

---

## 📜 License

MIT License

---

## ❤️ Authors

Developed by Arya Shah 

This project is created to help students become confident, skilled, and interview-ready.

**TalentScope → Turning fear into confidence.**
