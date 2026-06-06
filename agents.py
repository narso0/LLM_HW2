import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


class AutoMLAgent:
    def __init__(self, role, goal, instructions):
        self.role = role
        self.goal = goal
        self.instructions = instructions

    def think_and_act(self, state_context):
        prompt = f"""
        ROLE: {self.role}
        GOAL: {self.goal}

        INSTRUCTIONS & RULES:
        {self.instructions}
        - You must decide which tool to use.
        - When you are completely finished with your task, use the action "handoff".

        CURRENT DATASET/ENVIRONMENT CONTEXT:
        {state_context}

        AVAILABLE TOOLS AND THEIR EXACT PARAMETERS:
        - inspect_metadata() -> no parameters needed
        - get_column_stats(col) -> col: column name string
        - impute_missing(col, strategy) -> strategy: 'mean', 'median', or 'mode'
        - drop_column(col) -> col: column name string
        - create_interaction(expression) -> expression: pandas eval string e.g. "SibSp + Parch"
        - encode_categorical(col) -> col: column name string
        - correlation_analysis(target) -> target: target column name string
        - select_top_features(target_col, k) -> target_col: string, k: integer
        - execute_python_code(code_string) -> code_string: full Python script as a string

        ONLY use the exact parameter names listed above.
        Do not invent new tool names or parameter names.

        IMPORTANT:
        - Use the provided metadata and statistics to make decisions.
        - Do not assume column names exist.
        - Only use tools that make sense for the current dataset.
        - If enough information is not available, request more information using a tool.

        Respond ONLY with a valid JSON block matching this exact layout:
        {{
            "thought_process": "Your step-by-step reasoning based on the context",
            "action": "tool_name_to_call OR 'handoff'",
            "parameters": {{ "param_name": "value" }},
            "handoff_summary": "If action is 'handoff', put your final summary here. Otherwise leave blank."
        }}
        """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)