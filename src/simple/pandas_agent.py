import os
import ast
import numpy as np
from typing import Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
load_dotenv(".env.local", override=True)


# ---------- Data Models ---------- #
class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question about the data")
    df_input: str = Field(..., description="CSV path or DataFrame constructor")

class QueryResult(BaseModel):
    question: str
    pandas_code: str
    result_table: str
    summary: str

# ---------- Core Agents ---------- #
class SchemaAgent:
    def analyze(self, df: pd.DataFrame) -> str:
        type_info = "\n".join([
            f"{col}: {dtype}"
            for col, dtype in df.dtypes.items()
        ])
        
        schema = f"""
        Columns: {', '.join(df.columns)}
        Dataframe:\n{df.describe().to_markdown()}
        Shape: {df.shape}
        Data Types:\n{type_info}
        Sample Data:\n{df.head(3).to_markdown()}
        """
        return schema

class QueryGenerator:
    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def generate_code(self, question: str, schema: str) -> str:
        response = self.client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=[{
                "role": "user",
                "content": f"""
                You are a pandas expert working with a DataFrame called 'df'.
                Schema: {schema}
                
                For this question: "{question}"
                
                Generate Python code that:
                1. Understand the question and infer what the user is trying to find.
                2. Create a pandas query to answer the question using the DataFrame 'df'.
                3. Make sure the query give meaningful results for the question 
                   including any relevant columns that might not be directly associated with the question.
                4. Stores the final result in a variable called 'result'
                3. Returns ONLY the code wrapped in ```python```
                
                Example response:
                ```python
                result = df.groupby('Region')['Population'].mean()
                ```
                """
            }]
        )
        return self._extract_code(response.choices[0].message.content)

    def _extract_code(self, text: str) -> str:
        if '```python' in text:
            return text.split('```python')[1].split('```')[0]
        return text

class SafetyValidator:
    UNSAFE_KEYWORDS = ['eval', 'exec', 'system', 'os.', 'shutil', 'subprocess']
    
    def validate(self, code: str) -> bool:
        return not any(keyword in code for keyword in self.UNSAFE_KEYWORDS)

class QueryExecutor:
    def execute(self, df: pd.DataFrame, code: str) -> pd.DataFrame:
        local_vars = {'df': df.copy(), 'result': None}
        try:
            exec(code, {}, local_vars)
            # Handle cases where result might be a non-DataFrame object
            result = local_vars['result']
            if isinstance(result, pd.DataFrame):
                return result
            elif isinstance(result, pd.Series):
                return result.to_frame()
            elif isinstance(result, (int, float, str)):
                return pd.DataFrame({'Result': [result]})
            else:
                return pd.DataFrame({'Result': [str(result)]})
                
        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            if 'could not convert string to float' in error_msg.lower():
                error_msg += "\nHint: Some numeric columns may contain commas or other non-numeric characters"
            return pd.DataFrame({'Error': [error_msg]})

class ResponseFormatter:
    def format(self, result: pd.DataFrame, question: str, code: str) -> QueryResult:
        return QueryResult(
            question=question,
            pandas_code=code,
            result_table=result.to_markdown(),
            summary=self._generate_summary(result, question)
        )
    
    def _generate_summary(self, result: pd.DataFrame, question: str) -> str:
        if not result.empty and 'Error' not in result.columns:
            if len(result) == 1:
                return f"The result is: {result.iloc[0,0]}"
            else:
                stats = result.describe().to_markdown()
                return f"Analysis complete. Statistics:\n{stats}"
        return "Could not generate summary due to errors"

# ---------- Main Workflow ---------- #
class PandasAssistant:
    def __init__(self):
        self.schema_agent = SchemaAgent()
        self.query_agent = QueryGenerator()
        self.validator = SafetyValidator()
        self.executor = QueryExecutor()
        self.formatter = ResponseFormatter()
        
    def load_data(self, df_input: str) -> pd.DataFrame:
        try:
            if df_input.endswith('.csv'):
                df = pd.read_csv(df_input)
                
                for col in df.columns:
                    if df[col].dtype == 'object':
                        if df[col].str.contains(',').any():
                            try:
                                df[col] = df[col].str.replace(',', '.').astype(float)
                            except:
                                pass
                    if df[col].dtype == 'object':
                        df[col] = df[col].str.strip()
                percent_cols = [col for col in df.columns if '%' in col]
                for col in percent_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce') / 100
                    
                return df
            return pd.DataFrame(ast.literal_eval(df_input))
        except Exception as e:
            raise ValueError(f"Data loading failed: {str(e)}")
    
    def process_question(self, request: QueryRequest) -> QueryResult:
        df = self.load_data(request.df_input)
        schema = self.schema_agent.analyze(df)
        
        code = self.query_agent.generate_code(request.question, schema)
        
        if not self.validator.validate(code):
            return QueryResult(
                question=request.question,
                pandas_code=code,
                result_table="Blocked unsafe operation",
                summary="Query contained potentially dangerous operations"
            )
            
        result = self.executor.execute(df, code)
        return self.formatter.format(result, request.question, code)

# ---------- Interactive CLI ---------- #
def main():
    assistant = PandasAssistant()
    
    # Load initial data
    df_input = "data\countries of the world.csv"
    try:
        df = assistant.load_data(df_input)
        print(f"Country Dataset loaded successfully!\n{df.head(3)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    while True:
        question = input("\nAsk a question (or 'exit'):\n> ")
        if question.lower() == 'exit':
            break
            
        result = assistant.process_question(QueryRequest(
            question=question,
            df_input=df_input
        ))
        
        print(f"\n📊 Question: {result.question}")
        print(f"\n🐼 Pandas Code:\n{result.pandas_code}")
        print(f"\n🔍 Results:\n{result.result_table}")
        print(f"\n📝 Summary: {result.summary}\n")

if __name__ == "__main__":
    main()