import pandas as pd
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from gensim.models import KeyedVectors
import torch
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')


df = pd.read_excel("C:/Users/ahsha/Desktop/mp sem 7/ca.xlsx")

# Sentence embeddings (paraphrase aware)
embed_model = SentenceTransformer('paraphrase-mpnet-base-v2')

# Optional: Cross-encoder for higher accuracy
cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# NLI model
nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")

# Word-level embeddings (fastText)
# Make sure you have downloaded 'cc.en.300.vec' from fastText
# fasttext_model = KeyedVectors.load_word2vec_format('cc.en.300.vec', limit=100000)

# Difficulty order
difficulty_order = ['easy', 'medium', 'hard']

# -----------------------------
# 1. Helpers
# -----------------------------
def expand_synonyms(text):
    """Expand words with synonyms using WordNet."""
    words = text.split()
    expanded = set(words)
    for w in words:
        for syn in wordnet.synsets(w):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace('_', ' '))
    return " ".join(expanded)

def word_level_similarity(user_answer, correct_answer):
    """Token-level similarity using fastText embeddings."""
    # Uncomment if fastText is loaded
    # user_tokens = user_answer.lower().split()
    # correct_tokens = correct_answer.lower().split()
    # scores = []
    # for u in user_tokens:
    #     sims = []
    #     for c in correct_tokens:
    #         if u in fasttext_model.key_to_index and c in fasttext_model.key_to_index:
    #             sims.append(fasttext_model.similarity(u, c))
    #     if sims:
    #         scores.append(max(sims))
    # return sum(scores)/len(scores) if scores else 0
    return 0  # fallback if fastText not loaded

def embedding_similarity(user_answer, correct_answer):
    """Sentence-level embedding similarity."""
    emb1 = embed_model.encode(user_answer, convert_to_tensor=True)
    emb2 = embed_model.encode(correct_answer, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2).item())

def nli_evaluate(premise, hypothesis):
    """NLI probability for entailment."""
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    entail_prob = probs[0, 2].item()  # index 2 is entailment
    return entail_prob

def smart_evaluate(user_answer, correct_answer):
    """Combined evaluation: embeddings + word-level + NLI."""
    # Expand synonyms
    user_exp = expand_synonyms(user_answer)
    correct_exp = expand_synonyms(correct_answer)

    # Sentence embedding similarity
    sim_sentence = embedding_similarity(user_exp, correct_exp)

    # Word-level similarity
    sim_word = word_level_similarity(user_exp, correct_exp)

    # NLI entailment probability
    entail_prob = nli_evaluate(correct_answer, user_answer)

    # Weighted final score
    final_score = 0.6*sim_sentence + 0.3*sim_word + 0.1*entail_prob
    final_correct = final_score > 0.6  # threshold can be tuned

    return final_correct, sim_sentence, sim_word, entail_prob, final_score

def next_difficulty(current_difficulty, correct):
    idx = difficulty_order.index(current_difficulty)
    if correct and idx < len(difficulty_order)-1:
        return difficulty_order[idx+1]
    elif not correct and idx > 0:
        return difficulty_order[idx-1]
    return current_difficulty

# -----------------------------
# 2. Load and pick questions
# -----------------------------
def get_questions(category):
    df['category_clean'] = df['category'].astype(str).str.strip().str.lower()
    subset = df[df['category_clean'] == category.strip().lower()].copy()
    if subset.empty:
        print(f"No questions found for category: {category}")
    else:
        print(f"\nFound {len(subset)} questions for category: {category}")
        print("Breakdown by difficulty:")
        print(subset['difficulty'].value_counts())
    return subset.sample(frac=1).reset_index(drop=True)

def get_question_by_difficulty(subset, difficulty, used_indices):
    available = subset[
        (subset['difficulty'].str.lower() == difficulty.lower()) &
        (~subset.index.isin(used_indices))
    ]
    if not available.empty:
        return available.sample(1).iloc[0]
    return None

# -----------------------------
# 3. Main Assessment Loop
# -----------------------------
def run_assessment(subject, initial_difficulty='medium', num_questions=5):
    subset = get_questions(subject)
    used_indices = set()
    difficulty = initial_difficulty
    score = 0

    for i in range(num_questions):
        q = get_question_by_difficulty(subset, difficulty, used_indices)
        if q is None:
            remaining = subset[~subset.index.isin(used_indices)]
            if remaining.empty:
                break
            q = remaining.sample(1).iloc[0]

        used_indices.add(q.name)

        print(f"\nQ{i+1} [{difficulty.upper()}]: {q['question']}")
        user_answer = input("Your answer: ")

        final_correct, sim_sentence, sim_word, entail_prob, final_score = smart_evaluate(user_answer, q['answer_key'])

        print("\n--- Evaluation ---")
        print(f"🔹 Sentence Embedding Similarity: {sim_sentence:.2f}")
        print(f"🔹 Word-level Similarity: {sim_word:.2f}")
        print(f"🔹 NLI Entailment Probability: {entail_prob:.2f}")
        print(f"🔹 Weighted Final Score: {final_score:.2f}")
        print(f"👉 Final Verdict: {'✅ Correct' if final_correct else '❌ Incorrect'}")
        if not final_correct:
            print(f"Correct Answer: {q['answer_key']}")

        if final_correct:
            score += 1

        difficulty = next_difficulty(difficulty, final_correct)

    print(f"\n📊 Final Score: {score}/{num_questions}")

# -----------------------------
# 4. Run
# -----------------------------
if __name__ == "__main__":
    category = input("Select category (AI/ML/DL/SE/DBMS/OS): ")
    run_assessment(category, initial_difficulty='medium', num_questions=5)