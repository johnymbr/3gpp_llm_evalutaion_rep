import torch
import json
from unsloth import FastLanguageModel
from rag_functions import load_faiss_index, search_faiss_index, search_rag, load_chunks

class InferenceRagLlama:
    def __init__(self, model_path: str, chunks_path: str, faiss_path: str, question_path: str):
        self.model = None
        self.tokenizer = None
        self.questions = None

        self.model_path = model_path
        self.chunks_path = chunks_path
        self.faiss_path = faiss_path
        self.questions_path = question_path
        self.max_seq_length = 2048
        self.dtype = None
        self.load_in_4bit = True # Use 4bit quantization to reducer memory usage.

    def load_questions_mount_model_tokenizer(self):
        with open(self.questions_path, "r", encoding="utf-8") as file:
            self.questions = json.load(file)

        print(f"Questions size: {len(self.questions)}")
        print(self.questions[0])

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit
        )

        print(self.model)

if __name__ == '__main__':
    print("CUDA Available: ", torch.cuda.is_available())
    print("CUDA Device Name: ", torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()

    # Verify CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using Device: {device}")