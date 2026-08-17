"""
3GPP RAG Evaluation Runner

Runs the full evaluation suite against the RAG pipeline and generates
a detailed report with per-question scores and aggregate metrics.

Usage:
    python -m app.evaluation.run_eval --user-id <your_user_id>
    
    Example:
    python -m app.evaluation.run_eval --user-id 1
"""

import asyncio
import json
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List

from app.evaluation.test_suite import EVALUATION_TEST_CASES
from app.services.rag_service import answer_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Refusal indicators — if the answer contains these, we consider it a refusal
REFUSAL_INDICATORS = [
    "i cannot find",
    "i could not find", 
    "not available in the provided",
    "no relevant documents",
    "not found in the provided",
    "not in the provided context",
    "cannot answer",
    "insufficient information",
    "i don't have enough",
    "not mentioned in",
]


def check_refusal(answer: str) -> bool:
    """Check if the LLM's answer constitutes a refusal."""
    answer_lower = answer.lower()
    return any(indicator in answer_lower for indicator in REFUSAL_INDICATORS)


def check_keywords(answer: str, keywords: List[str]) -> Dict[str, bool]:
    """Check which expected keywords appear in the answer."""
    answer_lower = answer.lower()
    return {kw: kw.lower() in answer_lower for kw in keywords}


async def run_single_test(test_case: Dict, user_id: str, mode: str = "telecom") -> Dict[str, Any]:
    """Run a single evaluation test case."""
    question = test_case["question"]
    
    logger.info(f"\n{'='*60}")
    logger.info(f" Testing: {question[:80]}...")
    logger.info(f"   Category: {test_case['category']}")
    
    start_time = time.time()
    
    try:
        result = await answer_query(
            query=question,
            user_id=user_id,
            file_id=None,
            mode=mode,
            chat_history=""
        )
        
        answer = result.get("answer", "")
        metrics = result.get("metrics", {})
        
        # Evaluate
        is_refusal = check_refusal(answer)
        should_refuse = test_case["should_refuse"]
        keyword_results = check_keywords(answer, test_case.get("keywords_expected", []))
        
        # Score calculation
        test_passed = False
        score = 0.0
        reason = ""
        
        if should_refuse:
            # For refusal tests: passing = the system refused
            test_passed = is_refusal
            score = 100.0 if is_refusal else 0.0
            reason = "Correctly refused" if is_refusal else "FAILED: Should have refused but answered"
        else:
            # For accuracy tests: check faithfulness and keywords
            faithfulness = metrics.get("faithfulness", {})
            faith_score = faithfulness.get("score", 0)
            
            # Keyword coverage
            if keyword_results:
                kw_coverage = sum(keyword_results.values()) / len(keyword_results) * 100
            else:
                kw_coverage = 100  # No keywords to check
            
            # Combined score
            score = 0.4 * faith_score + 0.3 * kw_coverage + 0.3 * metrics.get("confidence_score", 0)
            test_passed = score >= 50 and not is_refusal
            reason = f"Faith: {faith_score}%, Keywords: {kw_coverage}%, Confidence: {metrics.get('confidence_score', 0)}%"
        
        eval_result = {
            "question": question,
            "category": test_case["category"],
            "expected_behavior": test_case["expected_behavior"],
            "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
            "test_passed": test_passed,
            "score": round(score, 1),
            "reason": reason,
            "is_refusal": is_refusal,
            "should_refuse": should_refuse,
            "keyword_results": keyword_results,
            "faithfulness": metrics.get("faithfulness", {}),
            "confidence_score": metrics.get("confidence_score", 0),
            "hallucination_risk": metrics.get("hallucination_risk", "Unknown"),
            "processing_time": round(time.time() - start_time, 2),
        }
        
        status = " PASSED" if test_passed else " FAILED"
        logger.info(f"   {status} | Score: {score:.1f}% | {reason}")
        
        return eval_result
        
    except Exception as e:
        logger.error(f"    ERROR: {e}")
        return {
            "question": question,
            "category": test_case["category"],
            "test_passed": False,
            "score": 0.0,
            "reason": f"Error: {str(e)}",
            "error": str(e),
        }


async def run_full_evaluation(user_id: str, mode: str = "telecom") -> Dict[str, Any]:
    """Run the complete evaluation suite."""
    logger.info("=" * 60)
    logger.info(" 3GPP RAG EVALUATION SUITE")
    logger.info(f"   User ID: {user_id}")
    logger.info(f"   Mode: {mode}")
    logger.info(f"   Test Cases: {len(EVALUATION_TEST_CASES)}")
    logger.info("=" * 60)
    
    results = []
    
    for i, test_case in enumerate(EVALUATION_TEST_CASES):
        logger.info(f"\n[{i+1}/{len(EVALUATION_TEST_CASES)}]")
        result = await run_single_test(test_case, user_id, mode)
        results.append(result)
        
        # Small delay to avoid rate limits on free tier
        await asyncio.sleep(2)
    
    # Aggregate metrics
    total = len(results)
    passed = sum(1 for r in results if r.get("test_passed", False))
    avg_score = sum(r.get("score", 0) for r in results) / total if total > 0 else 0
    
    # Category breakdown
    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0, "scores": []}
        categories[cat]["total"] += 1
        if r.get("test_passed", False):
            categories[cat]["passed"] += 1
        categories[cat]["scores"].append(r.get("score", 0))
    
    for cat in categories:
        scores = categories[cat]["scores"]
        categories[cat]["avg_score"] = round(sum(scores) / len(scores), 1) if scores else 0
        del categories[cat]["scores"]  # Clean up for JSON
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "mode": mode,
        "summary": {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total * 100, 1) if total > 0 else 0,
            "average_score": round(avg_score, 1),
        },
        "category_breakdown": categories,
        "detailed_results": results,
    }
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info(" EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"   Total: {total} | Passed: {passed} | Failed: {total - passed}")
    logger.info(f"   Pass Rate: {report['summary']['pass_rate']}%")
    logger.info(f"   Average Score: {avg_score:.1f}%")
    logger.info(f"\n   Category Breakdown:")
    for cat, data in categories.items():
        logger.info(f"     {cat}: {data['passed']}/{data['total']} passed (avg: {data['avg_score']}%)")
    
    # Save report
    report_path = f"data/eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        import os
        os.makedirs("data", exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"\n Report saved to: {report_path}")
    except Exception as e:
        logger.warning(f"Could not save report: {e}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Run 3GPP RAG Evaluation Suite")
    parser.add_argument("--user-id", type=str, required=True, help="User ID to test with")
    parser.add_argument("--mode", type=str, default="telecom", help="RAG mode to use")
    args = parser.parse_args()
    
    asyncio.run(run_full_evaluation(args.user_id, args.mode))


if __name__ == "__main__":
    main()
