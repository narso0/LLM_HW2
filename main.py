import os
import pandas as pd
import tools
from agents import AutoMLAgent


def run_pipeline():
    #make sure the dataset is actually there
    if not os.path.exists('./data/raw_data.csv'):
        print("Error: Please place your data file at ./data/raw_data.csv before running.")
        return

    df = pd.read_csv('./data/raw_data.csv')
    execution_logs = []

    #phase 1: data cleaner (ReAct Loop)
    cleaner = AutoMLAgent(
        role="The Data Cleaner",
        goal="Audit quality, drop useless columns, and resolve missing values.",
        instructions="Use inspect_metadata to find nulls. Use get_column_stats to investigate. Drop IDs/Names using drop_column. Impute missing values. Don't hardcode rules. When clean, action='handoff'."
    )

    cleaner_memory = ""
    for step in range(5):  #give it up to 5 turns to clean the data
        meta = tools.inspect_metadata(df)
        context = f"Metadata:\n{meta}\nPast Actions:\n{cleaner_memory}"

        decision = cleaner.think_and_act(context)
        action = decision['action']
        params = decision.get('parameters', {})
        execution_logs.append(f"Agent 1 (Cleaner) [Turn {step + 1}]: {decision['thought_process']} -> {action}")

        if action == 'impute_missing':
            df = tools.impute_missing(df, params.get('col'), params.get('strategy'))
            cleaner_memory += f"\n- Imputed {params.get('col')} with {params.get('strategy')}"
        elif action == 'drop_column':
            df = tools.drop_column(df, params.get('col'))
            cleaner_memory += f"\n- Dropped column {params.get('col')}"
        elif action == 'get_column_stats':
            stats = tools.get_column_stats(df, params.get('col'))
            cleaner_memory += f"\n- Stats for {params.get('col')}: {stats}"
        elif action == 'handoff':
            cleaner_summary = decision['handoff_summary']
            break

    df.to_csv("./data/clean_data.csv", index=False)

    #phase 2: feature engineering (ReAct Loop)
    engineer = AutoMLAgent(
        role="The Feature Engineer",
        goal="Maximize information density, encode categories, and drop redundant features.",
        instructions="Create logical features (create_interaction), encode text (encode_categorical), run correlation_analysis, and select_top_features(target='Survived'). When ready, action='handoff'."
    )

    engineer_memory = ""
    for step in range(5):
        context = f"Columns available: {list(df.columns)}\nPast Actions:\n{engineer_memory}"

        decision = engineer.think_and_act(context)
        action = decision['action']
        params = decision.get('parameters', {})
        execution_logs.append(f"Agent 2 (Engineer) [Turn {step + 1}]: {decision['thought_process']} -> {action}")

        if action == 'create_interaction':
            df, msg = tools.create_interaction(df, params.get('expression'))
            engineer_memory += f"\n- {msg}"
        elif action == 'encode_categorical':
            df = tools.encode_categorical(df, params.get('col'))
            engineer_memory += f"\n- Encoded {params.get('col')}"
        elif action == 'correlation_analysis':
            corr = tools.correlation_analysis(df, params.get('target', 'Survived'))
            engineer_memory += f"\n- Correlation: {corr}"
        elif action == 'select_top_features':
            df, msg = tools.select_top_features(df, params.get('target_col', 'Survived'), params.get('k', 5))
            engineer_memory += f"\n- {msg}"
        elif action == 'handoff':
            engineer_summary = decision['handoff_summary']
            break

    df.to_csv("./data/engineered_data.csv", index=False)

    #phase 3: model trainer (Code Execution Loop)
    trainer = AutoMLAgent(
        role="The Model Trainer",
        goal="Generate Python code to train and optimize XGBoost.",
        instructions="""You MUST generate a valid Python script. 
        Read data from './data/engineered_data.csv'. 
        Use train_test_split and xgboost.XGBClassifier. 
        Print Accuracy and F1 Score to standard output. 
        Evaluate the printed metrics on your next turn. If F1 < 0.80, change hyperparameters and write a new script. If satisfied, action='handoff'."""
    )

    trainer_context = f"Engineered Summary: {engineer_summary}\nAction: Write initial baseline XGBoost script."
    final_metrics = ""

    for iteration in range(3):
        decision_3 = trainer.think_and_act(trainer_context)
        action = decision_3['action']
        execution_logs.append(f"Agent 3 (Trainer) [Iter {iteration + 1}]: {decision_3['thought_process']} -> {action}")

        if action == 'execute_python_code':
            stdout, stderr = tools.execute_python_code(decision_3['parameters'].get('code_string', ''))
            trainer_context = f"Execution Output Log:\nSTDOUT: {stdout}\nSTDERR: {stderr}\nIf metrics are good, action='handoff'. If not, rewrite code."
        elif action == 'handoff':
            final_metrics = decision_3['handoff_summary']
            break

    #save the final report
    print("\n--- AGENT TEAM PROCESS LOGS ---")
    for log in execution_logs:
        print(log)

    report_md = f"""#Autonomous Multi-Agent AutoML System Report
##Execution Summary
* **Data Cleaner Summary:** {cleaner_summary}
* **Feature Engineer Summary:** {engineer_summary}
* **Final Model Trainer Output:** {final_metrics}

##Agent Thought Process Logs
{chr(10).join(['* ' + log for log in execution_logs])}
"""
    with open("./Final_Report.md", "w") as f:
        f.write(report_md)
    print("\nSuccessfully generated deliverable report: ./Final_Report.md")


if __name__ == "__main__":
    run_pipeline()