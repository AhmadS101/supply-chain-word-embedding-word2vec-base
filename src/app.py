import os
import sys
import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd


# --- Setup Project Root for Import ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Import Model ---
from src.cbow_model import CBOWModel, vocab_size, embedding_dim

# --- Load model ---

model = CBOWModel(vocab_size, embedding_dim)

model.load_state_dict(torch.load("../models/cbowmodel.pt"))
model.eval()

# --- Load Vocabulary ---

word2idx = pd.read_csv("../data/interm/word_to_id.csv")
idx2word = pd.read_csv("../data/interm/id_to_word.csv")

word2idx = dict(zip(word2idx["word"].str.lower(), word2idx["id"]))
idx2word = dict(zip(idx2word["id"], idx2word["word"].str.lower()))


# --- Utility Functions ---
def get_vector(word):
    idx = word2idx.get(word.lower())
    if idx is None:
        return None
    return model.embeddings.weight[idx].detach()


def find_similar_words(word, top_n=10):
    vector = get_vector(word)
    if vector is None:
        return None
    all_vectors = model.embeddings.weight.detach()
    similarities = F.cosine_similarity(vector.unsqueeze(0), all_vectors)
    top_indices = torch.topk(similarities, top_n + 1).indices.tolist()
    similar_words = [idx2word[i] for i in top_indices if idx2word[i] != word][:top_n]
    return similar_words


def solve_analogy(word_a, word_b, word_c, top_n=1):
    vec_a = get_vector(word_a)
    vec_b = get_vector(word_b)
    vec_c = get_vector(word_c)

    if None in (vec_a, vec_b, vec_c):
        return None

    target_vec = vec_b - vec_a + vec_c
    all_vectors = model.embeddings.weight.detach()
    similarities = F.cosine_similarity(target_vec.unsqueeze(0), all_vectors)
    top_indices = torch.topk(similarities, top_n + 3).indices.tolist()

    result = []
    for i in top_indices:
        candidate = idx2word[i]
        if candidate not in [word_a, word_b, word_c]:
            result.append(candidate)
        if len(result) == top_n:
            break
    return result


# --- Streamlit UI ---
st.title("🧠 Supply Chain Word2Vec Explorer")

# --- Similar Words Section ---
st.header("🔍 1. Find Similar Words")
input_word = st.text_input("Enter a word:")
if input_word:
    similar = find_similar_words(input_word)
    if similar:
        st.success(f"Top similar words to '{input_word}':")
        st.write(similar)
    else:
        st.error("Word not found in vocabulary.")

# --- Analogy Solver Section ---
st.header("🤔 2. Word Analogy Solver")
col1, col2, col3 = st.columns(3)
with col1:
    word_a = st.text_input("Word A")
with col2:
    word_b = st.text_input("Word B")
with col3:
    word_c = st.text_input("Word C")

if word_a and word_b and word_c:
    result = solve_analogy(word_a, word_b, word_c)
    if result:
        st.success(f"'{word_a}' is to '{word_b}' as '{word_c}' is to '{result[0]}'")
    else:
        st.error("Analogy failed – some words may not be in the vocabulary.")
