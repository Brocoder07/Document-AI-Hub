import pandas as pd
import logging
import json
from app.core.llm import get_llm
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

logger = logging.getLogger(__name__)

async def analyze_excel(file_path: str, query: str) -> dict:
    """
    Agents-style approach with Self-Correction/Reviewer step.
    Returns: {"answer": str, "confidence": float, "reason": str}
    """
    try:
        # 1. Load Data
        df = pd.read_excel(file_path)
        # Clean columns: remove spaces
        df.columns = [c.strip() for c in df.columns]
        
        columns_info = ", ".join(df.columns.tolist())
        sample_data = df.head(3).to_string()

        # 2. Prompt for Code Generation
        code_template = """
        You are a Python Data Analyst. 
        Given a pandas DataFrame named `df`, write a SINGLE LINE of Python code to answer the user's question.
        
        DATAFRAME INFO:
        Columns: {columns}
        Sample Data:
        {sample}

        USER QUESTION: {question}

        REQUIREMENTS:
        - Return ONLY the python expression. No markdown, no explanations.
        - ALWAYS return a DICTIONARY that includes the final result AND the underlying data used.
          This ensures the Assistant can explain the result using real numbers.
          Example: {{'result': (df['A'] + df['B']).sum(), 'breakdown': df[['A', 'B']].to_string()}}
        - Use .to_string() ONLY for DataFrames or Series. DO NOT use it on scalar numbers.
        - CRITICAL: If you use .unique(), wrap it in pd.Series() before calling .to_string(). 
          Example: pd.Series(df['Department'].unique()).to_string()
        - Filter strings case-insensitively if needed.
        
        PYTHON CODE:"""

        prompt = PromptTemplate.from_template(code_template)
        llm = get_llm()
        
        chain = prompt | llm
        response = await chain.ainvoke({
            "columns": columns_info,
            "sample": sample_data,
            "question": query
        })
        
        code = response.content.strip().replace("`", "").replace("python", "").strip()
        logger.info(f"🐍 Generated Pandas Query: {code}")

        # --- NEW STEP: THE REVIEWER ---
        review_template = """
        You are a strict Code Reviewer.
        Analyze the Python code generated for a DataFrame query.
        
        CONTEXT:
        - User Query: "{question}"
        - Generated Code: `{code}`
        - Columns Available: {columns}
        
        TASK:
        Rate the confidence (0-100) that this code answers the query correctly without logic errors.
        Provide a short reason.
        
        OUTPUT FORMAT (Raw JSON):
        {{"score": 95, "reason": "Correctly aggregates the 'Salary' column."}}
        """
        
        review_prompt = PromptTemplate.from_template(review_template)
        review_chain = review_prompt | llm
        
        # Execute Review
        confidence_score = 50.0
        confidence_reason = "Review step failed."
        
        try:
            review_res = await review_chain.ainvoke({
                "question": query,
                "code": code,
                "columns": columns_info
            })
            
            # --- ROBUST JSON PARSING ---
            content = review_res.content
            
            # Find the JSON object inside the text (handles conversational prefixes/suffixes)
            start_index = content.find('{')
            end_index = content.rfind('}')
            
            if start_index != -1 and end_index != -1:
                clean_json = content[start_index : end_index + 1]
                review_data = json.loads(clean_json)
                confidence_score = float(review_data.get("score", 50))
                confidence_reason = review_data.get("reason", "No reason provided.")
            else:
                logger.warning(f"Reviewer output format invalid: {content}")
                confidence_reason = "Reviewer output format invalid."
                
        except Exception as e:
            logger.warning(f"Reviewer failed: {e}")

        # 3. Safe Execution
        local_env = {"df": df, "pd": pd}
        try:
            # We use eval, which can handle dict literals: {'a': 1, 'b': 2}
            result = eval(code, {}, local_env)
        except Exception as exec_err:
            return {
                "answer": f"I tried to calculate this but the code failed: {exec_err}",
                "confidence": 0.0,
                "reason": "Execution Error"
            }

        # 4. Synthesis (The "Human Touch")
        explanation_prompt = f"""
        You act as a Data Analyst Assistant.
        
        CONTEXT:
        - User's Question: "{query}"
        - The Python Code you wrote: `{code}`
        - The Execution Result: {result}
        
        INSTRUCTIONS:
        Answer the user's question using ONLY the Execution Result.
        - Do NOT recalculate numbers. Trust the 'Execution Result' implicitly.
        - If the result is a Dictionary, present all parts clearly (e.g., state the Total, then show the List).
        - Format lists or tables nicely in Markdown.
        - Explain the logic briefly (e.g. "I filtered by Engineering...").
        
        FINAL ANSWER:
        """
        
        final_response = await llm.ainvoke([HumanMessage(content=explanation_prompt)])
        
        # Return structured dictionary
        return {
            "answer": final_response.content,
            "confidence": confidence_score,
            "reason": confidence_reason
        }

    except Exception as e:
        logger.error(f"Data Analysis Failed: {e}")
        return {
            "answer": f"Error analyzing data: {str(e)}",
            "confidence": 0.0,
            "reason": "System Error"
        }