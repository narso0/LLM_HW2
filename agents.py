import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


class AutoMLAgent:
    def __init__(self, role, goal, instructions):
        self.role = role
        self.goal = goal
        self.instructions = instructions

    def think_and_act(self, state_context):
        #send context to the llm and force it to reply in json format
        prompt = f"""
        ROLE: {self.role}
        GOAL: {self.goal}

        INSTRUCTIONS & RULES:
        {self.instructions}
        - You must decide which tool to use. 
        - When you are completely finished with your task, use the action "handoff".

        CURRENT DATASET/ENVIRONMENT CONTEXT:
        {state_context}

        Respond ONLY with a valid JSON block matching this exact layout:
        {{
            "thought_process": "Your step-by-step reasoning based on the context",
            "action": "tool_name_to_call OR 'handoff'",
            "parameters": {{ "param_name": "value" }},
            "handoff_summary": "If action is 'handoff', put your final summary here. Otherwise leave blank."
        }}
        """

        #no more training wheels. it either uses the api key or crashes.
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)