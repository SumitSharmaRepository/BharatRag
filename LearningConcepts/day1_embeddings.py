from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "GST return filing deadline",
    "Tax submission last date",
    "How to make biryani",
    "Income tax payment due date",
]

vectors = model.encode(sentences)

print(f"Each sentence becomes {len(vectors[0])} numbers")
print()

def similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print("Similarity scores:")
print(f"GST deadline vs Tax submission:  {similarity(vectors[0], vectors[1]):.3f}")
print(f"GST deadline vs Biryani recipe:  {similarity(vectors[0], vectors[2]):.3f}")
print(f"GST deadline vs Income tax date: {similarity(vectors[0], vectors[3]):.3f}")
