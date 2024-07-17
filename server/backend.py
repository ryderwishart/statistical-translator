from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
from combined_glosser import CombinedGlosser  # Import our glosser
import random

app = FastAPI()

class TranslationRequest(BaseModel):
    source_sentences: List[str]
    target_sentences: List[str]

class GlossResult(BaseModel):
    source: str
    target: str
    glosses: List[dict]

# Initialize our glosser with default parameters
# You may need to provide a corpus or adjust parameters as needed
glosser = CombinedGlosser(corpus=None, window_size=3, overlap=0.5)

# Mock data for source sentences
mock_source_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello, how are you doing today?",
    "I love learning new languages.",
    "This is a sample sentence for translation.",
    "Programming is both an art and a science."
]

@app.get("/source-sentences", response_model=List[str])
async def get_source_sentences():
    return mock_source_sentences

@app.post("/gloss", response_model=List[GlossResult])
async def get_glosses(request: TranslationRequest):
    results = []
    for source, target in zip(request.source_sentences, request.target_sentences):
        # Mock gloss data
        source_words = source.split()
        target_words = target.split()
        mock_glosses = [
            {
                "source_word": source_word,
                "target_word": target_word,
                "alignment": [random.randint(0, len(source_words)-1), random.randint(0, len(target_words)-1)]
            }
            for source_word, target_word in zip(source_words, target_words)
        ]
        results.append(GlossResult(source=source, target=target, glosses=mock_glosses))
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8123)