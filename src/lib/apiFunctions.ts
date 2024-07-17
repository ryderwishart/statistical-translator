// API functions
export async function fetchSourceSentences(): Promise<string[]> {
    const response = await fetch('http://localhost:8123/source-sentences');
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
}

export async function fetchGlosses(sourceSentences: string[], targetSentences: string[]): Promise<GlossResult[]> {
    const response = await fetch('http://localhost:8123/gloss', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_sentences: sourceSentences, target_sentences: targetSentences })
    });
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
}