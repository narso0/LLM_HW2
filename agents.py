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

        INSTRUCTIONS:
        {self.instructions}

        CURRENT DATASET/ENVIRONMENT CONTEXT:
        {state_context}

        Respond ONLY with a valid JSON block matching this exact layout:
        {{
            "thought_process": "Your step-by-step reasoning based on the context",
            "action": "tool_name_to_call",
            "parameters": {{ "param_name": "value" }},
            "handoff_summary": "Summary of work completed if you are finished and ready to hand off, otherwise leave as an empty string."
        }}
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            #fallback so we can test locally without paying for api calls
            return self._get_offline_mock_response(state_context)

    def _get_offline_mock_response(self, context):
        #mock responses just to keep the pipeline moving during testing
        if "Cleaner" in self.role:
            return {
                "thought_process": "found some missing values in the age column. going to impute them using the median.",
                "action": "impute_missing",
                "parameters": {"col": "age", "strategy": "median"},
                "handoff_summary": "dropped unique identifiers and fixed missing age entries with median values."
            }
        elif "Feature" in self.role:
            return {
                "thought_process": "looking at features. making a ratio between income and age makes sense here.",
                "action": "create_interaction",
                "parameters": {"expression": "income / age"},
                "handoff_summary": "created an income-per-age feature and handled categorical encoding."
            }
        else:
            if "baseline" in context.lower():
                return {
                    "thought_process": "baseline f1 is 0.71. need to tweak hyperparameters to get a better score.",
                    "action": "execute_python_code",
                    "parameters": {"code_string": "print('Accuracy: 0.82\\nRecall: 0.78\\nF1: 0.80')"},
                    "handoff_summary": ""
                }
            return {
                "thought_process": "hit 0.80 f1 after tuning, which is good enough to stop.",
                "action": "terminate",
                "parameters": {},
                "handoff_summary": "training done. hit our target xgboost metrics."
            }