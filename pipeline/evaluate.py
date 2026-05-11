# pipeline/evaluate.py
import os
import sys

# 1. CRITICAL FIX: Check for the API key BEFORE importing the chatbot
if "OPENAI_API_KEY" not in os.environ:
    print("\n⚠️ ERROR: OpenAI API key is missing!")
    print("Please run this command in your terminal first:")
    print("export OPENAI_API_KEY=\"sk-your-key-here\"\n")
    sys.exit(1)

from datasets import Dataset
from ragas import evaluate

# 2. WARNING FIX: Updated Ragas import path to avoid deprecation warnings and use new Metric Classes
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from pipeline.embeddings import search_db
from pipeline.chatbot import ask_chatbot

class StrictChatOpenAI(ChatOpenAI):
    """
    A custom wrapper that intercepts Ragas' requests and forces 
    the temperature back to 1 before sending it to OpenAI.
    """
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs['temperature'] = 1
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs['temperature'] = 1
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

# 3. Add 'ground_truth' to your test suite! 
# (ground_truth will be used by Ragas to compare against the chatbot's answer for a more objective evaluation)
TEST_DATA = [
    {
        "question": "Hvad er retningslinjerne for vold?",
        "ground_truth": "Retningslinjerne for vold ifølge de medfølgende dokumenter fokuserer på samfundstjeneste, særligt grov vold og strafudmåling."
    },
    {
        "question": "Hvad er straffene for forbrydelser, der involverer en mindreårig?",
        "ground_truth": "Bøder kan normalt også pålægges unge under 18 år, og bøden reduceres efter alder og indkomst. Sanktioner for unge afhænger også af forbrydelsens alvor."
    }
]

def generate_evaluation_dataset():
    print("⏳ Genererer svar til evaluering... (Generating answers for evaluation...)")
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in TEST_DATA:
        q = item["question"]
        gt = item["ground_truth"]

        retrieved_chunks = search_db(q, n_results=3)
        context_texts = [chunk['text'] for chunk in retrieved_chunks]
        
        bot_answer = ask_chatbot(q)
        
        questions.append(q)
        answers.append(bot_answer)
        contexts.append(context_texts)
        ground_truths.append(gt)

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    return Dataset.from_dict(data)

if __name__ == "__main__":
    # Step 1: Run your chatbot against the test suite
    eval_dataset = generate_evaluation_dataset()
    
    # Step 2: Define the LLM Judge (with our custom strict wrapper)
    judge_llm = StrictChatOpenAI(model="gpt-5-nano")
    
    # Step 2.5: Define the Embedding model for Ragas to use
    eval_embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Step 3: Use Ragas to grade the results
    print("\n📊 Evaluerer med Ragas... (Evaluating with Ragas...)")
    
    results = evaluate(
        dataset=eval_dataset,
        metrics=[
            Faithfulness(),       
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall()
        ],
        llm=judge_llm,
        embeddings=eval_embeddings
    )
    
    print("\n" + "="*80)
    print("📈 EVALUATION SCORES (0.0 to 1.0):")
    print("="*80)
    
    # 1. Print the overall average score
    print("AVERAGE SCORES:")
    print(results)
    print("-" * 80)
    
    # 2. Safely print the detailed Pandas table
    try:
        df = results.to_pandas()
        print("\nINDIVIDUAL QUESTION SCORES:")
        # Drop the heavy 'contexts' and 'answer' text columns so it fits on the screen
        columns_to_show = [col for col in df.columns if col not in ['contexts', 'answer', 'ground_truth', 'reference']]
        print(df[columns_to_show])
    except Exception as e:
        print(f"Could not format table: {e}")