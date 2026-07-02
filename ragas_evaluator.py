from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional
from ragas import EvaluationDataset
from dotenv import load_dotenv
import os
from rag_client import format_context
from llm_client import generate_response
import sys
from openai import OpenAI as OpenAIClient
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings


load_dotenv()
# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str], 
                               reference: str = None) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = "https://openai.vocareum.com/v1" if api_key.startswith("voc") else None

    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    # validate input
    if (
        not question or not question.strip() or
        not answer or not answer.strip() or
        not contexts or not isinstance(contexts, list) or
        len(contexts) == 0 or
        not all(isinstance(c, str) and c.strip() for c in contexts)
    ):
        return {"error": "Question, answer, and at least one retrieved context are required."}
    

    # Create LLM and embeddings for evaluation
    openai_client = OpenAIClient(
        api_key=api_key, 
        base_url=base_url,
        timeout = 120.0,
        max_retries=3)
    
    evaluator_llm = llm_factory("gpt-3.5-turbo", client=openai_client)

    evaluator_embeddings = LangchainEmbeddingsWrapper(
    LangchainOpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        base_url=base_url
    )
)
    # Only ResponseRelevancy for now — fastest metric, faithfulness is slow
    metrics = [ResponseRelevancy(strictness=1)] # voc apis only returns 1 generation
    
    # Add RougeScore if reference is provided
    if reference is not None:
        metrics.append(RougeScore())

    if not metrics:
        return {
            "response_relevancy": None,
            "rouge_score": None,
            "error": "No metrics to evaluate — reference is None"
        }

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=reference
    )

    dataset = EvaluationDataset(samples=[sample])
    results = evaluate(
        dataset,
        metrics=metrics,
        llm=evaluator_llm, 
        embeddings=evaluator_embeddings
    )
    df = results.to_pandas()
    results_dict = df.to_dict(orient="records")[0]

    return {
        "response_relevancy": results_dict.get("answer_relevancy", None),
        "rouge_score": results_dict.get("rouge_score(mode=fmeasure)", None)
    }

def parse_evaluation_dataset(file_path: str) -> List[Dict[str, str]]:
    """
    Parse evaluation_dataset.txt into a list of question/reference pairs.
    
    Args:
        file_path: Path to the evaluation_dataset.txt file
        
    Returns:
        List of dicts with 'question' and 'reference' keys
    """
    qa_pairs = []
    
    with open(file_path, "r") as f:
        content = f.read()
    
    # Split into blocks by "Question:"
    blocks = content.strip().split("Question:")
    blocks = [b.strip() for b in blocks if b.strip()]  # remove empty blocks
    
    for block in blocks:
        # Split each block into question and answer parts
        if "Answer:" not in block:
            continue
        
        parts = block.split("Answer:", 1)  # split on first "Answer:" only
        question = parts[0].strip()
        reference = parts[1].strip()
        
        if question and reference:
            qa_pairs.append({
                "question": question,
                "reference": reference
            })
    
    return qa_pairs

def evaluate_dataset(file_path: str, embedding_pipeline, openai_key: str) -> None:
    """
    Run evaluation on all QA pairs and print per-question scores and aggregate summary.
    
    Args:
        file_path: Path to evaluation_dataset.txt
        embedding_pipeline: ChromaEmbeddingPipeline instance
        openai_key: OpenAI API key
    """

    qa_pairs = parse_evaluation_dataset(file_path)

    all_scores = {
        "response_relevancy": [],
        "rouge_score": []
    }
   

    print(f"\nEvaluating {len(qa_pairs)} questions...\n")
    print("=" * 60)

    for i, pair in enumerate(qa_pairs, 1):
        question = pair["question"]
        reference = pair["reference"]

        print(f"\n[{i}/{len(qa_pairs)}] Question: {question}")

        # --- Retrieve and generate ---
        results = embedding_pipeline.query_collection(question, n_results=3)
        contexts = results["documents"][0]
        metadatas = results["metadatas"][0]
        context_str = format_context(contexts, metadatas)
        answer = generate_response(openai_key, question, context_str, conversation_history=[])

        print(f"  Answer   : {answer[:100]}...")

        # --- Evaluate ---
        scores = evaluate_response_quality(question, answer, contexts, reference)

        # --- Per-question scores ---
        print(f"  Scores:")
        for metric in all_scores:
            score = scores.get(metric)
            if score is not None:
                all_scores[metric].append(score)
                print(f"    {metric:<25}: {score:.2f}")
            else:
                print(f"    {metric:<25}: N/A")

        print("-" * 60)

    # --- Aggregate Summary ---
    print("\n" + "=" * 60)
    print("AGGREGATE SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<25} {'Mean':>6} {'Min':>6} {'Max':>6} {'Std':>6}")
    print("-" * 60)

    for metric, scores in all_scores.items():
        if scores:
            mean  = sum(scores) / len(scores)
            best  = max(scores)
            worst = min(scores)
            std   = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
            print(f"{metric:<25} {mean:>6.2f} {worst:>6.2f} {best:>6.2f} {std:>6.2f}")

def main():
    import sys
    sys.path.append("..")
    
    from embedding_pipeline import ChromaEmbeddingPipelineTextOnly  
    
    load_dotenv(".env")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not found in .env")
        sys.exit(1)

    print("Loading embedding pipeline...")
    embedding_pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=openai_key,
        chroma_persist_directory="./chroma_db_openai", 
        collection_name="nasa_space_missions_text"
    )

    print("Running evaluation...")
    evaluate_dataset(
        file_path="evaluation_dataset.txt",
        embedding_pipeline=embedding_pipeline,
        openai_key=openai_key
    )

if __name__ == "__main__":
    main()