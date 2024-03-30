import io
import os
import re
import json
import gc
from nltk.stem import PorterStemmer, WordNetLemmatizer
import markdown
import pdfplumber
import pypandoc
from contextlib import contextmanager

@contextmanager
def open_file(file_path, mode='r', encoding=None):
    """Context manager for efficiently handling file operations."""
    f = io.open(file_path, mode, encoding=encoding) if encoding else open(file_path, mode)
    try:
        yield f
    finally:
        f.close()
        gc.collect()

def process_documents(folder_path):
    # Initialize stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    processed_folder = os.path.join(folder_path, "processed_files")
    os.makedirs(processed_folder, exist_ok=True)

    documents = []
    processed_documents_file = os.path.join(processed_folder, "processed_documents.json")

    # Load existing processed documents if available
    if os.path.exists(processed_documents_file):
        with open_file(processed_documents_file, 'r', encoding='utf-8') as f:
            for line in f:
                documents.append(json.loads(line))

    # Process each file in the directory
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_name, file_ext = os.path.splitext(file)

            content = process_file(file_path, file_ext)
            if not content:
                continue  # Skip if content is empty

            document_chunks = process_content(content, file_name, processed_folder, stemmer, lemmatizer)
            del content  # Explicitly free memory
            gc.collect()

            # Write each chunk to the processed documents file
            with open_file(processed_documents_file, 'a', encoding='utf-8') as f:
                for chunk in document_chunks:
                    f.write(json.dumps(chunk) + '\n')

    gc.collect()
    return documents

def process_file(file_path, file_ext):
    content = None
    try:
        if file_ext.lower() in [".docx", ".odt", ".pptx", ".ppt", ".doc"]:
            content = pypandoc.convert_file(file_path, 'markdown', outputfile=None)
        elif file_ext.lower() == ".pdf":
            with pdfplumber.open(file_path) as pdf:
                content = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
                pdf.close()
        elif file_ext.lower() in [".txt", ".md"]:
            with io.open(file_path, 'r', encoding='utf8') as f:
                content = f.read()
                f.close()
            if file_ext.lower() == ".md":
                content = markdown.markdown(content)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    finally:
        gc.collect()

    return content

def process_content(content, file_name, processed_folder, stemmer, lemmatizer, file_limit=10 * 1024 * 1024):
    document_chunks = []
    current_file_chunks = []
    current_file_size = 0
    chunk_counter = 0

    for i in range(0, len(content), 5000):
        chunk = content[i:i + 5000]
        words = re.findall(r'\b\w+\b', chunk.lower())
        n_grams = generate_n_grams(words)

        chunk_data = {
            "chunk_id": chunk_counter,
            "content": chunk,
            "words": words,
            "stemmed_words": [stemmer.stem(word) for word in words],
            "lemmatized_words": [lemmatizer.lemmatize(word) for word in words],
            "n_grams": n_grams
        }

        chunk_size = len(json.dumps(chunk_data).encode('utf-8'))
        if current_file_size + chunk_size > file_limit:
            new_chunks = write_chunks_to_file(current_file_chunks, file_name, processed_folder)
            document_chunks.extend(new_chunks)
            current_file_chunks = []
            current_file_size = 0
            chunk_counter = 0

        current_file_chunks.append(chunk_data)
        current_file_size += chunk_size
        chunk_counter += 1

    if current_file_chunks:
        new_chunks = write_chunks_to_file(current_file_chunks, file_name, processed_folder)
        document_chunks.extend(new_chunks)

    return document_chunks

def write_chunks_to_file(chunks, file_name, processed_folder):
    file_path = os.path.join(processed_folder, f"{file_name}_{len(chunks)}.json")
    with open_file(file_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False)

    new_chunks = []
    for chunk in chunks:
        new_chunks.append({
            "file_path": file_path,
            "chunk_id": chunk["chunk_id"],
            "content": chunk["content"],
            "words": chunk["words"],
            "stemmed_words": chunk["stemmed_words"],
            "lemmatized_words": chunk["lemmatized_words"],
            "n_grams": chunk["n_grams"]
        })
    gc.collect()
    return new_chunks


def generate_n_grams(words, max_n=5):
    n_grams = []
    for n in range(1, max_n + 1):
        n_grams.extend([' '.join(words[i:i+n]) for i in range(len(words)-n+1)])
    return n_grams

def build_inverted_index(documents, folder_path="data/"):
    processed_folder = os.path.join(folder_path, "processed_files")
    os.makedirs(processed_folder, exist_ok=True)
    inverted_index_file = os.path.join(processed_folder, "inverted_index.txt")
    inverted_index = {}

    if os.path.exists(inverted_index_file):
        with open(inverted_index_file, 'r', encoding='utf-8') as f:
            inverted_index = eval(f.read())

    for doc in documents:
        for n_gram in doc["n_grams"]:
            doc_ref = (doc["file_path"], doc.get("chunk_id", 0))  # Use chunk_id if available
            inverted_index.setdefault(n_gram, []).append(doc_ref)

    with open(inverted_index_file, 'w', encoding='utf-8') as f:
        f.write(str(inverted_index))

    return inverted_index

