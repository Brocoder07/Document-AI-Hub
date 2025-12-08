import pandas as pd
import logging
from app.core.llm import get_llm
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

logger = logging.getLogger(__name__)

async def analyze_excel(file_path: str, query: str):
    """
    Agents-style approach:
    1. Load Excel into Pandas.
    2. Ask LLM to write a Pandas filter/aggregation query.
    3. Execute code.
    4. Synthesize a natural language answer explaining the result.
    """
    try:
        # 1. Load Data
        df = pd.read_excel(file_path)
        # Clean columns: remove spaces, lowercase logic if needed
        df.columns = [c.strip() for c in df.columns]
        
        columns_info = ", ".join(df.columns.tolist())
        sample_data = df.head(3).to_string()

        # 2. Prompt for Code Generation (Fixed for Scalars)
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
        - If the user asks for multiple things (e.g., "Total AND List"), return a DICTIONARY.
          Example: {{'total': df['Salary'].sum(), 'list': df[['Name', 'Salary']].to_string()}}
        - Use .to_string() ONLY for DataFrames or Series. DO NOT use it on scalar numbers (int/float/sum).
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

        # 3. Safe Execution
        local_env = {"df": df, "pd": pd}
        try:
            # We use eval, which can handle dict literals: {'a': 1, 'b': 2}
            result = eval(code, {}, local_env)
        except Exception as exec_err:
            return f"I tried to calculate this but the code failed: {exec_err}"

        # 4. Synthesis (The "Human Touch")
        # We explicitly tell the LLM to trust the result numbers.
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
        return final_response.content

    except Exception as e:
        logger.error(f"Data Analysis Failed: {e}")
        return f"Error analyzing data: {str(e)}"