# pipeline/evaluate.py
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from langchain_openai import ChatOpenAI
# NEW IMPORT: Bring in the LangChain wrapper for local embeddings
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

# 1. Create a Test Suite
TEST_QUESTIONS = [
    "Hvad er retningslinjerne for vold?",
    "Hvad er straffen for økonomisk bedrageri?",
]

def generate_evaluation_dataset():
    print("⏳ Genererer svar til evaluering... (Generating answers for evaluation...)")
    
    questions = []
    answers = []
    contexts = []

    for q in TEST_QUESTIONS:
        retrieved_chunks = search_db(q, n_results=3)
        context_texts = [chunk['text'] for chunk in retrieved_chunks]
        
        bot_answer = ask_chatbot(q)
        
        questions.append(q)
        answers.append(bot_answer)
        contexts.append(context_texts)

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    
    return Dataset.from_dict(data)

if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: Please 'export OPENAI_API_KEY=\"your-key\"' first.")
        exit()

    # Step 1: Run your chatbot against the test suite
    eval_dataset = generate_evaluation_dataset()
    
    # Step 2: Define the LLM Judge (with our custom strict wrapper)
    judge_llm = StrictChatOpenAI(model="gpt-5-nano")
    
    # Step 2.5: Define the Embedding model for Ragas to use
    # We use the exact same Danish-friendly model you used in embeddings.py!
    eval_embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Step 3: Use Ragas to grade the results
    print("\n📊 Evaluerer med Ragas... (Evaluating with Ragas...)")
    
    results = evaluate(
        dataset=eval_dataset,
        metrics=[
            faithfulness,       
            answer_relevancy,   
        ],
        llm=judge_llm,
        embeddings=eval_embeddings # <--- TELL RAGAS TO USE THE LOCAL MODEL
    )
    
    print("\n" + "="*50)
    print("📈 EVALUATION SCORES (0.0 to 1.0):")
    print("="*50)
    
    try:
        print(results.to_pandas()[['question', 'faithfulness', 'answer_relevancy']])
    except KeyError:
        print("Evaluation failed to produce the expected columns. Raw results:")
        print(results)