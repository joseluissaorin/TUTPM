import re
import bisect
import os
import tempfile
import shutil
import json

def search(query, inverted_index, documents):
    query = query.lower().strip()
    relevant_doc_chunks = set()

    # Generate n-grams from the query
    query_words = re.findall(r'\b\w+\b', query)
    n_grams = []
    for n in range(1, min(6, len(query_words) + 1)):
        n_grams.extend([' '.join(query_words[i:i+n]) for i in range(len(query_words) - n + 1)])

    # Find relevant chunks in the inverted index
    for n_gram in n_grams:
        if n_gram in inverted_index:
            relevant_doc_chunks.update(inverted_index[n_gram])

    # Calculate relevance and prepare results
    relevant_docs = []
    temp_paths = []  # To store paths of temporary chunk files

    try:
        for file_path, chunk_id in relevant_doc_chunks:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                chunk = chunks[chunk_id]  # Access the specific chunk

                relevance_score = sum(n_gram in chunk["content"].lower() for n_gram in n_grams) / len(n_grams)
                if relevance_score > 0:
                    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', encoding='utf-8', suffix='.txt')
                    temp_file.write(chunk["content"])
                    temp_file.close()
                    relevant_docs.append((temp_file.name, relevance_score))
                    temp_paths.append(temp_file.name)  # Keep track of temporary file paths

        # Sort documents by relevance
        relevant_docs.sort(key=lambda x: x[1], reverse=True)

        # Return paths of the top N relevant temporary chunk files
        top_n = 10
        print([doc[0] for doc in relevant_docs[:top_n]])
        return [doc[0] for doc in relevant_docs[:top_n]]
    finally:
        pass



def binary_search(words, word):
    index = bisect.bisect_left(words, word)
    if index != len(words) and words[index] == word:
        return index
    return -1
