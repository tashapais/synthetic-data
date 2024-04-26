import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Function to calculate BLEU score
def calculate_bleu(references, candidates):
    return corpus_bleu([[ref.split()] for ref in references], [cand.split() for cand in candidates])

# Function to calculate ROUGE scores
def calculate_rouge(references, candidates):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, cand) for ref, cand in zip(references, candidates)]
    return np.mean([s['rouge1'].fmeasure for s in scores]), \
           np.mean([s['rouge2'].fmeasure for s in scores]), \
           np.mean([s['rougeL'].fmeasure for s in scores])

# Function to calculate Perplexity
def calculate_perplexity(model, tokenizer, text):
    encode = tokenizer.encode(text, return_tensors='pt')
    with torch.no_grad():
        loss = model(encode, labels=encode).loss
    return torch.exp(loss).item()

# Example data (placeholders)
real_texts = ["This is a real example.", "Here is another real sentence."]
generated_texts = ["This is a fake example.", "Here is another generated sentence."]

# Calculate BLEU and ROUGE scores
bleu_score = calculate_bleu(real_texts, generated_texts)
rouge_score = calculate_rouge(real_texts, generated_texts)

print("BLEU Score:", bleu_score)
print("ROUGE Scores:", rouge_score)

# Load your model and tokenizer
model_path = 'path_to_your_trained_model'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Calculate Perplexity for an example text
perplexity = calculate_perplexity(model, tokenizer, "This is an example sentence to evaluate.")
print("Perplexity:", perplexity)

# Example embeddings for t-SNE (these should be your model's embeddings)
embeddings = np.random.rand(10, 768)  # Random data for illustration

# t-SNE Visualization
tsne_model = TSNE(n_components=2, random_state=0)
tsne_results = tsne_model.fit_transform(embeddings)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=['red']*5 + ['blue']*5)
plt.title('t-SNE visualization of text embeddings')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend(['Real Data', 'Generated Data'], loc='best')
plt.show()
