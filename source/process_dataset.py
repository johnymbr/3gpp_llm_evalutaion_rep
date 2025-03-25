import json
from pathlib import Path


class ProcessDataset:
    def __init__(self, dataset_path: str, output_path: str):
        self.dataset_path = dataset_path
        self.output_path = output_path

    def process_teleqna(self):
        with open(self.dataset_path, "r", encoding="utf-8") as file:
            teleqna_dataset = json.load(file)

        print("Dataset path: {}. Size: {}\n".format(self.dataset_path, len(teleqna_dataset)))

        print("Question 0")
        print(teleqna_dataset["question 0"])

        print("\nChoose only Release 17 Questions:")

        rel17_questions = [
            value for key, value in teleqna_dataset.items() if "[3GPP Release 17]" in value["question"]
        ]

        print(f"Total questions with '[3GPP Release 17]': {len(rel17_questions)}\n")

        rel17_questions_path = Path("{}/rel17_questions.json".format(self.output_path))
        if not rel17_questions_path.exists():
            with open(rel17_questions_path, "w", encoding="utf-8") as file:
                json.dump(rel17_questions, file, indent=4, ensure_ascii=False)

        # Choose 100 questions
        print("Choose 100 questions of Release 17")
        category_counts = {}
        for question in rel17_questions:
            category = question.get("category", "Unknown")
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts[category] = 1

        print("Categories found and counts:")
        for category, count in category_counts.items():
            print(f"- {category}: {count}")

        print("\n")
        number_questions = 100

        # Calculate how many questions to take from each category
        questions_per_category = number_questions // len(category_counts)

        rel17_100_questions = []
        for category, count in category_counts.items():
            category_questions = [q for q in rel17_questions if q.get("category", "Unknown") == category]
            rel17_100_questions.extend(category_questions[:questions_per_category])

        print(f"Check size of Release 17 100 questions: {len(rel17_100_questions)}")

        # Print the total number of selected questions
        print(f"\nTotal selected questions: {len(rel17_100_questions)}")
        for idx, question in enumerate(rel17_100_questions):
            print(f"{idx + 1}. {question['question']} (Category: {question['category']})")

        print("\nChoose only Release 18 Questions:")

        rel18_questions = [
            value for key, value in teleqna_dataset.items() if "[3GPP Release 18]" in value["question"]
        ]

        print(f"Total questions with '[3GPP Release 18]': {len(rel17_questions)}\n")

        rel18_questions_path = Path("{}/rel18_questions.json".format(self.output_path))
        if not rel18_questions_path.exists():
            with open(rel18_questions_path, "w", encoding="utf-8") as file:
                json.dump(rel18_questions, file, indent=4, ensure_ascii=False)

