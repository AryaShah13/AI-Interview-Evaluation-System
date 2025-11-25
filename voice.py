# ============================================================
# 🎙 Adaptive Voice Interview AI (VS Code Version)
# Using Pygame for reliable audio playback on Windows
# ============================================================

import pandas as pd
import torch
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.corpus import wordnet
import nltk
import os
import tempfile
import pygame
import time
from gtts import gTTS

# Ensure WordNet is available
nltk.download('wordnet', quiet=True)

# -------------------------------------------------------
# 1️⃣ Load Dataset and Models
# -------------------------------------------------------
df = pd.read_excel("ca.xlsx")  # Make sure ca.xlsx is in same folder

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
nli_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-miniLM2-L6-H768")
nli_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-miniLM2-L6-H768")

difficulty_order = ['easy', 'medium', 'hard']

# -------------------------------------------------------
# 2️⃣ NLP Helper Functions
# -------------------------------------------------------
def expand_synonyms(text):
    words = text.split()
    expanded = set(words)
    for w in words:
        for syn in wordnet.synsets(w):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace('_', ' '))
    return " ".join(expanded)

def embedding_similarity(user_answer, correct_answer):
    emb1 = embed_model.encode(user_answer, convert_to_tensor=True)
    emb2 = embed_model.encode(correct_answer, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2).item())

def nli_evaluate(premise, hypothesis):
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    entail_prob = probs[0, 2].item() if probs.size(1) == 3 else probs[0, 1].item()
    return entail_prob

def smart_evaluate(user_answer, correct_answer):
    user_exp = expand_synonyms(user_answer)
    correct_exp = expand_synonyms(correct_answer)
    sim_sentence = embedding_similarity(user_exp, correct_exp)
    entail_prob = nli_evaluate(correct_answer, user_answer)
    final_score = 0.85 * sim_sentence + 0.15 * entail_prob
    final_correct = final_score > 0.6
    return final_correct, sim_sentence, entail_prob, final_score

def next_difficulty(current_difficulty, correct):
    idx = difficulty_order.index(current_difficulty)
    if correct and idx < len(difficulty_order) - 1:
        return difficulty_order[idx + 1]
    elif not correct and idx > 0:
        return difficulty_order[idx - 1]
    return current_difficulty

# -------------------------------------------------------
# 3️⃣ Voice I/O Functions
# -------------------------------------------------------
def record_audio(filename="response.wav", duration=8):
    print("🎙 Recording... Speak now!")
    fs = 16000
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    sf.write(filename, audio, fs)
    print("✅ Recording complete.")
    return filename

def speech_to_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"🗣 You said: {text}")
        return text
    except sr.UnknownValueError:
        print("❌ Could not understand your speech.")
        return ""
    except sr.RequestError:
        print("⚠ Speech service unavailable.")
        return ""

def speak_text(text):
    """Use gTTS + pygame for reliable playback on Windows"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts = gTTS(text=text, lang='en')
        tts.save(fp.name)

    pygame.mixer.init()
    pygame.mixer.music.load(fp.name)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.3)

    pygame.mixer.music.unload()
    pygame.mixer.quit()
    os.remove(fp.name)

# -------------------------------------------------------
# 4️⃣ Adaptive Voice Interview Loop
# -------------------------------------------------------
def run_voice_assessment(subject, initial_difficulty='medium', num_questions=5):
    subset = df[df['category'].str.lower() == subject.lower()]
    used_indices = set()
    difficulty = initial_difficulty
    score = 0

    speak_text(f"Welcome to your {subject} interview. Let's begin.")

    for i in range(num_questions):
        q = subset[(subset['difficulty'].str.lower() == difficulty.lower()) & (~subset.index.isin(used_indices))]
        if q.empty:
            q = subset[~subset.index.isin(used_indices)]
        if q.empty:
            break
        q = q.sample(1).iloc[0]
        used_indices.add(q.name)

        question_text = f"Question {i+1}, {difficulty} level: {q['question']}"
        print(f"\n🧩 {question_text}")
        speak_text(question_text)

        audio_file = record_audio(duration=8)
        user_answer = speech_to_text(audio_file)

        if not user_answer:
            speak_text("I could not hear your answer properly. Let's move to the next question.")
            continue

        final_correct, sim_sentence, entail_prob, final_score = smart_evaluate(user_answer, q['answer_key'])

        print(f"\n--- Evaluation ---")
        print(f"Sentence Similarity: {sim_sentence:.2f}")
        print(f"Entailment Probability: {entail_prob:.2f}")
        print(f"Final Score: {final_score:.2f}")
        print(f"Verdict: {'✅ Correct' if final_correct else '❌ Incorrect'}")

        feedback = (
            f"Your answer is {'correct' if final_correct else 'incorrect'}. "
            f"The similarity score was {sim_sentence:.2f}. "
            f"Entailment probability was {entail_prob:.2f}. "
        )
        speak_text(feedback)

        if final_correct:
            score += 1

        difficulty = next_difficulty(difficulty, final_correct)

    result_text = f"Interview completed. Your final score is {score} out of {num_questions}."
    print(result_text)
    speak_text(result_text)

# -------------------------------------------------------
# 5️⃣ Main
# -------------------------------------------------------
if __name__ == "__main__":
    print("Available categories:", df['category'].unique())
    subject = input("Select category (AI/ML/DL/SE/DBMS/OS): ")
    run_voice_assessment(subject)
