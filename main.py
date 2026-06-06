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

    #detect the target column automatically
    target_col = 'Survived'

    execution_logs = []

    #phase 1: data cleaner (ReAct Loop)
    cleaner = AutoMLAgent(
        role="The Data Cleaner",
        goal="Audit quality, drop useless columns, and resolve missing values.",
        instructions="""
        You have a maximum of 8 turns. You MUST complete all of these steps and then handoff:
        1. Drop 'PassengerId', 'Name', 'Ticket', 'Cabin' using drop_column (one at a time).
        2. Impute 'Age' with median using impute_missing.
        3. Impute 'Embarked' with mode using impute_missing.
        4. Once all done, action='handoff'.
        Do not keep investigating. Start acting immediately.
        """
    )

    cleaner_summary = "Cleaner did not complete handoff."
    cleaner_memory = ""

    for step in range(8):  #give it up to 5 turns to clean the data
        meta = tools.inspect_metadata(df)
        context = f"Metadata:\n{meta}\nPast Actions:\n{cleaner_memory}"

        decision = cleaner.think_and_act(context)

        print(decision)

        action = decision['action']
        params = decision.get('parameters', {})

        execution_logs.append(
            f"Agent 1 (Cleaner) [Turn {step + 1}]: {decision['thought_process']} -> {action}"
        )

        if action == 'impute_missing':
            df = tools.impute_missing(
                df,
                params.get('col'),
                params.get('strategy')
            )

            cleaner_memory += (
                f"\n- Imputed {params.get('col')} "
                f"with {params.get('strategy')}"
            )

        elif action == 'drop_column':
            df = tools.drop_column(
                df,
                params.get('col')
            )

            cleaner_memory += (
                f"\n- Dropped column {params.get('col')}"
            )

        elif action == 'get_column_stats':
            stats = tools.get_column_stats(
                df,
                params.get('col')
            )

            cleaner_memory += (
                f"\n- Stats for {params.get('col')}: {stats}"
            )

        elif action == 'handoff':
            cleaner_summary = decision['handoff_summary']
            break

    df.to_csv("./data/clean_data.csv", index=False)

    #phase 2: feature engineering (ReAct Loop)
    engineer = AutoMLAgent(
        role="The Feature Engineer",
        goal="Maximize information density, encode categories, and drop redundant features.",
        instructions=f"""
        Follow these steps IN ORDER. Do each step EXACTLY ONCE. Check Past Actions before each step.

        Step 1: If correlation_analysis not in Past Actions -> run correlation_analysis(target='{target_col}')
        Step 2: If create_interaction not in Past Actions -> run create_interaction(expression='SibSp + Parch')
        Step 3: If 'Encoded Sex' not in Past Actions -> run encode_categorical(col='Sex')
        Step 4: If 'Encoded Embarked' not in Past Actions -> run encode_categorical(col='Embarked')
        Step 5: If select_top_features not in Past Actions -> run select_top_features(target_col='{target_col}', k=8)
        Step 6: All steps done -> action='handoff'

        STRICT RULES:
        - Do NOT create more than one interaction feature.
        - Do NOT repeat any action already in Past Actions.
        - After select_top_features you MUST handoff immediately.
        """
    )

    engineer_summary = "Engineer did not complete handoff."
    engineer_memory = ""

    for step in range(8):

        context = f"""
Columns available: {list(df.columns)}

Metadata:
{tools.inspect_metadata(df)}

Past Actions:
{engineer_memory}
"""

        decision = engineer.think_and_act(context)

        print(decision)

        action = decision['action']
        params = decision.get('parameters', {})

        execution_logs.append(
            f"Agent 2 (Engineer) [Turn {step + 1}]: {decision['thought_process']} -> {action}"
        )

        if action == 'create_interaction':

            df, msg = tools.create_interaction(
                df,
                params.get('expression')
            )

            engineer_memory += f"\n- {msg}"

        elif action == 'encode_categorical':

            df = tools.encode_categorical(
                df,
                params.get('col')
            )

            engineer_memory += (
                f"\n- Encoded {params.get('col')}"
            )

        elif action == 'correlation_analysis':

            corr = tools.correlation_analysis(
                df,
                params.get('target', target_col)
            )

            engineer_memory += (
                f"\n- Correlation: {corr}"
            )

        elif action == 'select_top_features':

            df, msg = tools.select_top_features(
                df,
                params.get('target_col', target_col),
                params.get('k', 5)
            )

            engineer_memory += f"\n- {msg}"

        elif action == 'handoff':
            engineer_summary = decision['handoff_summary']
            break

    df.to_csv("./data/engineered_data.csv", index=False)

    #phase 3: model trainer (Code Execution Loop)
    trainer = AutoMLAgent(
        role="The Model Trainer",
        goal="Generate Python code to train and optimize XGBoost.",
        instructions=f"""
        You have 5 iterations. Follow this logic strictly:

        Iteration 1: Write and execute a baseline XGBoost script.
        After execution: READ the STDOUT metrics carefully.
        - If F1 Score >= 0.80 -> action='handoff' immediately. Include the metrics in handoff_summary.
        - If F1 Score < 0.80 -> rewrite the script with different hyperparameters and execute again.

        RULES:
        - Read data from './data/engineered_data.csv'
        - Target column is '{target_col}'
        - Use train_test_split and xgboost.XGBClassifier
        - Print Accuracy and F1 Score to stdout
        - The ONLY valid actions are 'execute_python_code' and 'handoff'
        - If you see F1 >= 0.80 in the output, you MUST handoff on the very next turn
        """
    )

    trainer_context = (
        f"Engineered Summary: {engineer_summary}\n"
        f"Action: Write initial baseline XGBoost script."
    )

    final_metrics = "Trainer did not complete handoff."

    for iteration in range(5):

        decision_3 = trainer.think_and_act(trainer_context)

        print(decision_3)

        action = decision_3['action']

        execution_logs.append(
            f"Agent 3 (Trainer) [Iter {iteration + 1}]: "
            f"{decision_3['thought_process']} -> {action}"
        )

        if action == 'execute_python_code':

            stdout, stderr = tools.execute_python_code(
                decision_3['parameters'].get(
                    'code_string',
                    ''
                )
            )

            trainer_context = f"""
Execution Output Log:

STDOUT:
{stdout}

STDERR:
{stderr}

If metrics are good, action='handoff'.

If not, rewrite code.
"""

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