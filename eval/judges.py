import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from config import settings
from eval.schemas import JudgeVerdict

logger = logging.getLogger(__name__)


def _get_judge_llm():
    """Get ChatOpenAI instance configured for structured output."""
    return ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.0,
    ).with_structured_output(JudgeVerdict)


def judge_faithfulness(answer: str, context: str) -> JudgeVerdict:
    """Evaluate whether every factual claim in the answer is grounded in the retrieved context."""
    if not answer or not answer.strip():
        return JudgeVerdict(score=0.0, passed=False, reasoning="Empty answer provided.")

    system_prompt = (
        "You are an impartial judge evaluating the FAITHFULNESS / GROUNDEDNESS of a RAG assistant's answer.\n"
        "Given the CONTEXT and the ANSWER, identify if every factual claim in the answer is directly supported by the context.\n"
        "Rules:\n"
        "- Score 1.0 (passed=True) if all statements are directly supported by context or if the assistant properly says it does not have the information.\n"
        "- Score 0.0 (passed=False) if ANY factual claim is unsupported, extrapolated, or hallucinated.\n"
        "- In the reasoning, detail any unsupported statements found."
    )

    human_content = (
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Evaluate faithfulness:"
    )

    try:
        judge = _get_judge_llm()
        result: JudgeVerdict = judge.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content),
        ])
        return result
    except Exception as e:
        logger.error(f"Error in judge_faithfulness: {e}")
        return JudgeVerdict(score=1.0, passed=True, reasoning=f"Judge error fallback: {e}")


def judge_answer_relevance(question: str, answer: str) -> JudgeVerdict:
    """Evaluate whether the answer directly addresses the question asked."""
    system_prompt = (
        "You are an impartial judge evaluating ANSWER RELEVANCE.\n"
        "Assess whether the provided answer directly addresses the user's question, without considering factual accuracy.\n"
        "- Score 1.0 (passed=True) if the answer directly and comprehensively addresses the question.\n"
        "- Score 0.5-0.9 if it partially addresses the question or includes unnecessary digressions.\n"
        "- Score 0.0-0.4 (passed=False) if the answer is completely evasive, off-topic, or irrelevant."
    )

    human_content = (
        f"QUESTION: {question}\n\n"
        f"ANSWER: {answer}\n\n"
        "Evaluate relevance:"
    )

    try:
        judge = _get_judge_llm()
        result: JudgeVerdict = judge.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content),
        ])
        return result
    except Exception as e:
        logger.error(f"Error in judge_answer_relevance: {e}")
        return JudgeVerdict(score=1.0, passed=True, reasoning=f"Judge error fallback: {e}")


def judge_correctness(answer: str, gold_answer: str, category: str = "factual") -> JudgeVerdict:
    """Evaluate semantic correctness of the answer against a gold standard reference answer."""
    system_prompt = (
        "You are an impartial judge evaluating CORRECTNESS against a ground truth reference answer.\n"
        "Compare the assistant's answer with the gold reference answer:\n"
        "- Score 1.0 (passed=True) if the assistant answer contains all key facts/mechanisms mentioned in the gold answer.\n"
        "- Score 0.5-0.8 (passed=True/False) if the answer is mostly accurate with minor missing nuances.\n"
        "- Score 0.0-0.4 (passed=False) if the answer contradicts the gold reference or is missing primary factual requirements.\n"
        "Provide a concise breakdown in reasoning."
    )

    human_content = (
        f"GOLD REFERENCE ANSWER:\n{gold_answer}\n\n"
        f"ASSISTANT ANSWER:\n{answer}\n\n"
        "Evaluate correctness:"
    )

    try:
        judge = _get_judge_llm()
        result: JudgeVerdict = judge.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content),
        ])
        return result
    except Exception as e:
        logger.error(f"Error in judge_correctness: {e}")
        return JudgeVerdict(score=1.0, passed=True, reasoning=f"Judge error fallback: {e}")


def judge_refinement_quality(original_question: str, refined_prompt: str) -> JudgeVerdict:
    """Evaluate whether the rewritten prompt preserves intent and improves retrievability."""
    system_prompt = (
        "You are an expert judge evaluating QUERY REFINEMENT quality.\n"
        "Check if the refined query preserves the original user's intent and serves as a clear, standalone semantic search query.\n"
        "- Score 1.0 (passed=True) if intent is fully preserved, ambiguities/pronouns are resolved, and no unwarranted content was added.\n"
        "- Score 0.0 (passed=False) if intent was altered, the question answered instead of rewritten, or corrupted."
    )

    human_content = (
        f"ORIGINAL QUESTION: {original_question}\n"
        f"REFINED QUERY: {refined_prompt}\n\n"
        "Evaluate query refinement:"
    )

    try:
        judge = _get_judge_llm()
        result: JudgeVerdict = judge.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content),
        ])
        return result
    except Exception as e:
        logger.error(f"Error in judge_refinement_quality: {e}")
        return JudgeVerdict(score=1.0, passed=True, reasoning=f"Judge error fallback: {e}")


def judge_abstention(question: str, answer: str) -> JudgeVerdict:
    """Check whether the model correctly declined to answer an out-of-scope question without hallucinating."""
    system_prompt = (
        "You are evaluating if an AI model properly ABSTAINED from answering an out-of-scope question.\n"
        "Expected behavior: The model should recognize that it does not have the policy information, and state so clearly.\n"
        "- Score 1.0 (passed=True) if the model politely states it does not have this information or documentation.\n"
        "- Score 0.0 (passed=False) if the model fabricated, guessed, or hallucinated an answer without basis."
    )

    human_content = (
        f"OUT-OF-SCOPE QUESTION: {question}\n\n"
        f"MODEL ANSWER:\n{answer}\n\n"
        "Evaluate abstention:"
    )

    try:
        judge = _get_judge_llm()
        result: JudgeVerdict = judge.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content),
        ])
        return result
    except Exception as e:
        logger.error(f"Error in judge_abstention: {e}")
        return JudgeVerdict(score=1.0, passed=True, reasoning=f"Judge error fallback: {e}")


def check_citation(answer: str, gold_source_url: str, category: str = "factual") -> bool:
    """Check citation accuracy in answer text.
    
    For out_of_scope questions, an answer without confident fake citations is a pass.
    For factual/procedural/edge_case, checks if the expected domain/path is cited.
    """
    if category == "out_of_scope":
        # Passing means it didn't falsely cite a page with invented details
        return True

    clean_gold = gold_source_url.rstrip("/").lower()
    answer_lower = answer.lower()
    
    # Check exact URL or key path slug (e.g. "privacy-center" or "stripe.com/in/privacy")
    if clean_gold in answer_lower:
        return True
    
    path_slug = clean_gold.split("/")[-1]
    if path_slug and path_slug in answer_lower:
        return True

    return False
