from pathlib import Path
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from rag_functions import format_answer, extract_answer, extract_option
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import FactualCorrectness, SemanticSimilarity, answer_relevancy, BleuScore, RougeScore, \
    ExactMatch, StringPresence

import json

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


class RagasEvaluation:
    def __init__(self, responses_path: str, evaluation_path: str):
        self.responses_path = responses_path
        self.evaluation_path = evaluation_path

        with open(self.responses_path, "r", encoding="utf-8") as file:
            self.responses = json.load(file)

        print(len(self.responses))

        self.data_samples = {
            'user_input': [],
            'response': [],
            'reference': []
        }

        self.dataset = None

    def transform_dataset(self):
        print("********** Transform Dataset **********")
        for item in self.responses:
            question = item['question']
            model_response = format_answer(extract_answer(item['response']))
            correct_answer = format_answer(extract_option(item['answer'], first_option=False))

            model_response = model_response.rstrip('.') + '.'
            correct_answer = correct_answer.rstrip('.') + '.'

            self.data_samples['user_input'].append(question)
            self.data_samples['response'].append(model_response)
            self.data_samples['reference'].append(correct_answer)

        self.dataset = Dataset.from_dict(self.data_samples)
        print(self.dataset)

    def evaluate_llm(self):
        print("********** Evaluate with LLM **********")
        score = evaluate(
            self.dataset,
            metrics=[
                FactualCorrectness(),
                SemanticSimilarity(),
                answer_relevancy
            ],
            llm=llm,
            embeddings=embeddings,
            run_config=RunConfig(timeout=400, max_retries=20, max_wait=120, log_tenacity=False),
        )

        evaluation_ragas_pd = score.to_pandas()
        evaluation_ragas_pd.to_csv(self.evaluation_path + r"/llama_3_2_lora_short_answer_rag_evaluation_ragas_llm.csv")

    def evaluate_no_llm(self):
        print("********** Evaluate without LLM **********")
        score = evaluate(
            self.dataset,
            metrics=[
                BleuScore(),
                RougeScore(),
                ExactMatch(),
                StringPresence()
            ],
            llm=llm,
            embeddings=embeddings
        )

        print(score)

        evaluation_ragas_pd = score.to_pandas()
        evaluation_ragas_pd.to_csv(
            self.evaluation_path + r"/llama_3_2_lora_short_answer_rag_evaluation_ragas_no_llm.csv")


if __name__ == '__main__':
    # call
    output_path_str = "../../files/evaluations"
    Path(output_path_str).mkdir(parents=True, exist_ok=True)

    ragas_eval = RagasEvaluation(
        responses_path="../../files/responses/llama_3.2_lora_short_answer_rag_responses_no_options.json",
        evaluation_path=output_path_str)
    ragas_eval.transform_dataset()
    # ragas_eval.evaluate_llm()
    ragas_eval.evaluate_no_llm()
