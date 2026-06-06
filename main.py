import os
import pandas as pd
import tools
from agents import AutoMLAgent


def run_pipeline():
    # Load your real training data
    if not os.path.exists('./data/raw_data.csv'):
        print("Error: Please place your data file at ./data/raw_data.csv before running.")
        return

    df = pd.read_csv('./data/raw_data.csv')
    execution_logs = []

    # Instantiate the Team
    cleaner = AutoMLAgent(
        role="The Data Cleaner",
        goal="Audit quality and resolve missing values/outliers.",
        instructions="Look at metadata tool outputs. Clean columns autonomously without hardcoding rules."
    )

    engineer = AutoMLAgent(
        role="The Feature Engineer",
        goal="Maximize information density and clean redundancy.",
        instructions="Create logical mathematical transformations or apply encoding schemes based on data context."
    )

    trainer = AutoMLAgent(
        role="The Model Trainer",
        goal="Generate Python code strings to train and iteratively optimize XGBoost models.",
        instructions="Generate fully executable training scripts. Evaluate stdout metrics to check if hyperparameter tuning is needed."
    )

    # ---------------------------------------------
    # PHASE 1: Data Cleaner Execution
    # ---------------------------------------------
    meta = tools.inspect_metadata(df)
    decision = cleaner.think_and_act(f"Raw Metadata:\n{meta}")
    execution_logs.append(f"Agent 1 (Cleaner): {decision['thought_process']}")

    if decision['action'] == 'impute_missing':
        df = tools.impute_missing(df, decision['parameters']['col'], decision['parameters']['strategy'])

    cleaner_summary = decision['handoff_summary']
    df.to_csv("./data/clean_data.csv", index=False)

    # ---------------------------------------------
    # PHASE 2: Feature Engineer Execution
    # ---------------------------------------------
    context_2 = f"Cleaner Summary: {cleaner_summary}\nColumns available: {list(df.columns)}"
    decision_2 = engineer.think_and_act(context_2)
    execution_logs.append(f"Agent 2 (Engineer): {decision_2['thought_process']}")

    if decision_2['action'] == 'create_interaction':
        df, msg = tools.create_interaction(df, decision_2['parameters']['expression'])

    engineer_summary = decision_2['handoff_summary']
    df.to_csv("./data/engineered_data.csv", index=False)

    # ---------------------------------------------
    # PHASE 3: Model Trainer Loop (Feedback Iterations)
    # ---------------------------------------------
    trainer_context = f"Engineered Summary: {engineer_summary}\nStatus: Run initial baseline setup."

    for iteration in range(2):
        decision_3 = trainer.think_and_act(trainer_context)
        execution_logs.append(f"Agent 3 (Trainer) [Iter {iteration + 1}]: {decision_3['thought_process']}")

        if decision_3['action'] == 'execute_python_code':
            stdout, stderr = tools.execute_python_code(decision_3['parameters']['code_string'])
            trainer_context = f"Execution Output Log:\n{stdout}"
        else:
            break

    final_metrics = decision_3['handoff_summary']

    # ---------------------------------------------
    # DELIVERABLE GENERATION
    # ---------------------------------------------
    print("\n--- AGENT TEAM PROCESS LOGS ---")
    for log in execution_logs:
        print(log)

    report_md = f"""# Autonomous Multi-Agent AutoML System Report
## Execution Summary
* **Data Cleaner Summary:** {cleaner_summary}
* **Feature Engineer Summary:** {engineer_summary}
* **Final Model Trainer Output:** {final_metrics}

## Agent Thought Process Logs
{chr(10).join(['* ' + log for log in execution_logs])}
"""
    with open("./Final_Report.md", "w") as f:
        f.write(report_md)
    print("\nSuccessfully generated deliverable report: ./Final_Report.md")


if __name__ == "__main__":
    run_pipeline()