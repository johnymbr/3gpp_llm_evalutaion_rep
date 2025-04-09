import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from sentence_transformers import SentenceTransformer


def save_chunks(chunks, filename):
    with open(filename, 'wb') as f:
        pickle.dump(chunks, f)
    print(f'Chunks saved to {filename}')


def load_chunks(filename):
    with open(filename, 'rb') as f:
        chunks = pickle.load(f)
    return chunks


class ProcessRAG:
    def __init__(self, tspec_dir_path: str, output_path: str):
        self.tspec_dir_path = tspec_dir_path
        self.output_path = output_path
        self.tspec_data = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Maximum size of each chunk
            chunk_overlap=100,  # Amount of overlap between chunks
            separators=["\n\n", "\n", " ", ""]  # Separators for splitting
        )
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3")
            ],
            strip_headers=False
        )

    def load_tspec_data(self):
        """
        Loads the content of all .md files from a directory and its subdirectories.
        Returns a list of dictionaries with 'release', 'series', and 'content'.
        """
        print("\n##########\nLoad TSpec Data\n##########\n")
        for release in os.listdir(self.tspec_dir_path):
            release_path = os.path.join(self.tspec_dir_path, release)
            if os.path.isdir(release_path):
                for series in os.listdir(release_path):
                    series_path = os.path.join(release_path, series)
                    if os.path.isdir(series_path):
                        for file in os.listdir(series_path):
                            if file.endswith('.md'):
                                file_path = os.path.join(series_path, file)
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    self.tspec_data.append({
                                        "release": release,
                                        "series": series,
                                        "content": content
                                    })

        # Check an example
        print(type(self.tspec_data))
        print(f"Total documents loaded: {len(self.tspec_data)}")
        print(f"Sample document: {self.tspec_data[0]}")

    def split_tspec_into_chunks_with_recursive_splitter(self, save: bool):
        """
        Function to split text into chunks using RecursiveCharacterTextSplitter
        """
        print("\n##########\nSplit With Recursive Splitter\n##########\n")
        dataset_chunks = []
        for document in self.tspec_data:
            release = document['release']
            series = document['series']
            content = document['content']

            chunks = self.text_splitter.split_text(content)
            for chunk in chunks:
                dataset_chunks.append({
                    'release': release,
                    'series': series,
                    'text': chunk
                })

        # Verification example
        print(f"Total chunks created: {len(dataset_chunks)}")
        print(f"Example chunk: {dataset_chunks[0]}")

        if save:
            print("Save complete chunks and just text of chunks")
            save_chunks(dataset_chunks, self.output_path + "/tspec_chunks.pkl")

            all_texts = [chunk_data['text'] for chunk_data in dataset_chunks]
            save_chunks(all_texts, self.output_path + "/tspec_chunks_texts.pkl")

    def split_tspec_into_chunks_with_markdown_splitter(self, save: bool):
        """
        Function to split text into chunks using RecursiveCharacterTextSplitter
        """
        print("\n##########\nSplit With Markdown Splitter\n##########\n")
        dataset_chunks = []
        for document in self.tspec_data:
            release = document['release']
            series = document['series']
            content = document['content']

            header_chunks = self.markdown_splitter.split_text(content)

            for header_chunk in header_chunks:
                char_chunks = self.text_splitter.split_text(header_chunk.page_content)
                for chunk in char_chunks:
                    dataset_chunks.append({
                        'release': release,
                        'series': series,
                        'text': chunk
                    })

        # Check the result
        print(f"Total chunks created: {len(dataset_chunks)}")
        print(f"Example chunk: {dataset_chunks[0]}")

        if save:
            print("Save complete markdown chunks and just text of chunks")
            save_chunks(dataset_chunks, self.output_path + "/tspec_chunks_markdown.pkl")

            all_texts = [chunk_data['text'] for chunk_data in dataset_chunks]
            save_chunks(all_texts, self.output_path + "/tspec_chunks_markdown_texts.pkl")

    def generate_embeddings(self):
        print("\n##########\nGenerate Embeddings\n##########\n")

        chunks_path = self.output_path + r"/tspec_chunks_markdown.pkl"
        tspec_chunks = load_chunks(chunks_path)
        print(len(tspec_chunks))

        chunks_text_path = self.output_path + r"/tspec_chunks_markdown_texts.pkl"
        all_texts = load_chunks(chunks_text_path)
        print(len(all_texts))

        embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model = SentenceTransformer('all-mpnet-base-v2', device=embedding_device)

        embeddings = embedding_model.encode(
            all_texts,
            batch_size=64,
            convert_to_tensor=False,
            show_progress_bar=True
        )

        for idx, chunk_data in enumerate(tspec_chunks):
            chunk_data['embedding'] = embeddings[idx]

        print("Save complete markdown chunks with embeddings")
        save_chunks(tspec_chunks, self.output_path + "/tspec_chunks_markdown_with_embeddings.pkl")

    def indexing_embeddings_with_faiss_and_save(self):
        """
        Function that will load embedding chunks and embeddings, indexing them with faiss and save
        """
        print("\n##########\nIndexing With Faiss and Save\n##########\n")

        chunks_path = self.output_path + r"/tspec_chunks_markdown_with_embeddings.pkl"
        tspec_chunks = load_chunks(chunks_path)
        print(len(tspec_chunks))

        # Extract the embeddings as a Numpy array
        embeddings_np = np.array([chunk['embedding'] for chunk in tspec_chunks]).astype('float32')

        # Normalize the embeddings to L2 for cosine similarity
        faiss.normalize_L2(embeddings_np)

        # Dimensionality of the embeddings
        dim = embeddings_np.shape[1]
        print(f"Dimensionality of embeddings: {dim}")

        # Create a FAISS index for cosine similarity using normalized embeddings
        index = faiss.IndexFlatIP(dim)  # Using Inner Product for cosine similarity after normalization

        # Add embeddings to the index
        index.add(embeddings_np)

        # Print the number of indices added to the index
        print(f"Number of indices saved: {index.ntotal}")

        # Save the FAISS index
        faiss.write_index(index, self.output_path + '/faiss_index.bin')
        print(f"FAISS index saved to {self.output_path}/faiss_index.bin")


if __name__ == '__main__':
    print("CUDA Available: ", torch.cuda.is_available())
    print("CUDA Device Name: ", torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()

    # Verify CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using Device: {device}")

    # call
    output_path_str = "../files/rag"
    Path(output_path_str).mkdir(parents=True, exist_ok=True)

    process = ProcessRAG("../datasets/TSpec-LLM/3GPP-clean", output_path_str)
    #process.load_tspec_data()
    #process.split_tspec_into_chunks_with_markdown_splitter(save=True)
    # process.generate_embeddings()
    process.indexing_embeddings_with_faiss_and_save()

