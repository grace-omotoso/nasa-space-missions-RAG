from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional
from ragas import EvaluationDataset
from dotenv import load_dotenv
import os

load_dotenv()
# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""

    api_key=os.getenv("OPENAI_API_KEY")
    base_url = "https://openai.vocareum.com/v1" if api_key.startswith("voc") else None

    
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    # TODO: Create evaluator LLM with model gpt-3.5-turbo
    os.environ["OPENAI_BASE_URL"] = "https://openai.vocareum.com/v1"
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_retries=1,
        max_tokens=256
        ))
    # TODO: Create evaluator_embeddings with model test-embedding-3-small
    langchain_openai_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        base_url=base_url)
    evaluator_embeddings = LangchainEmbeddingsWrapper(langchain_openai_embeddings)

    # TODO: Define an instance for each metric to evaluate
    metrics = [
        ResponseRelevancy(),
        Faithfulness(), 
        RougeScore()
    ]
    # TODO: Evaluate the response using the metrics
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts
    )
   
    # TODO: Return the evaluation results
    dataset = EvaluationDataset(samples=[sample])
    results = evaluate(dataset, metrics=metrics, llm=evaluator_llm, embeddings=evaluator_embeddings)
    results_dict = results.to_pandas().to_dict(orient="records")[0]

    return {
        "response_relevancy": results_dict.get("response_relevancy", None),
        "faithfulness": results_dict.get("faithfulness", None),
        "rouge_score": results_dict.get("rouge_score", None)
    }
