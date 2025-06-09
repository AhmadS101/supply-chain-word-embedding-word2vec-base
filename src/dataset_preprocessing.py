import re
import torch
import pandas as pd


# reading the combined dataset
with open("../data/interm/combined_SC_data.txt", "r", encoding="utf-8") as file:
    dataset = file.readlines()
text = " ".join(dataset)

# reading the stopwords
with open("../data/interm/en_stopword.txt") as st:
    stopwords = st.read()
stopwords = stopwords.replace("\n", " ").split(" ")


# cleaning dataset and tokenizing text
def clean_tokenize_text(text):
    cleaned_text = re.sub(r"[^a-zA-Z]", " ", text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    cleaned_text = cleaned_text.lower()
    tokens = cleaned_text.split(" ")
    return [token for token in tokens if token not in stopwords][:-1]


# Tokenize the text and remove stopwords
tokens = clean_tokenize_text(text)

# Create a set of unique tokens, word-to-id mappings
unique_vocab = sorted(list(set(tokens)))
vocab_size = len(unique_vocab)
word_to_id = {word: idx for idx, word in enumerate(unique_vocab)}
id_to_word = {idx: word for idx, word in enumerate(unique_vocab)}

# save word-to-id and id-to-word mappings to CSV files
word_to_id_df = pd.DataFrame(list(word_to_id.items()), columns=["word", "id"])
word_to_id_df.to_csv("../data/interm/word_to_id.csv", index=False)

id_to_word_df = pd.DataFrame(list(id_to_word.items()), columns=["id", "word"])
id_to_word_df.to_csv("../data/interm/id_to_word.csv", index=False)

# Save the vocabulary to a CSV file
vocab_df = pd.DataFrame({"word": unique_vocab, "id": range(vocab_size)})
vocab_df.to_csv("../data/interm/vocabulary.csv", index=False)


# Create target and context words for the model
target_words = []
context_words = []
window_size = 3

for i in range(1, len(tokens) - 1):
    target_words.append(tokens[i])
    context_words.append(
        tokens[i - window_size : i] + tokens[i + 1 : i + window_size + 1]
    )


# Convert target and context words to indices
target_word_indices = [word_to_id[word] for word in target_words]
context_word_indices = [
    [word_to_id[word] for word in context] for context in context_words
]

# Save target and context word indices to CSV files
target_df = pd.DataFrame({"target_index": target_word_indices})
target_df.to_csv("../data/processed/target_indices.csv", index=False)

context_df = pd.DataFrame({"context_indices": context_word_indices})
context_df.to_csv("../data/processed/context_indices.csv", index=False)
