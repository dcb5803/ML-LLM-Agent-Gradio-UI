import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import gradio as gr
from transformers import pipeline

# 1️⃣ Load dataset (embedded)
data = pd.DataFrame({
    "income": [50000, 60000, 35000, 80000, 45000],
    "credit_score": [700, 650, 600, 720, 580],
    "loan_amount": [20000, 25000, 12000, 30000, 15000],
    "approved": [1, 1, 0, 1, 0]
})

# 2️⃣ Train model
X = data.drop("approved", axis=1)
y = data["approved"]
model = RandomForestClassifier()
model.fit(X, y)

# 3️⃣ Load open-source LLM agent (Hugging Face)
llm = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", device_map="auto")

# 4️⃣ Agentic explainer
def explain_decision(income, credit_score, loan_amount):
    input_data = pd.DataFrame([[income, credit_score, loan_amount]],
                              columns=["income", "credit_score", "loan_amount"])
    prediction = model.predict(input_data)[0]
    decision = "approved" if prediction == 1 else "rejected"
    prompt = f"Explain why a loan application with income {income}, credit score {credit_score}, and loan amount {loan_amount} was {decision}."
    explanation = llm(prompt, max_new_tokens=100)[0]["generated_text"]
    return f"Prediction: {decision}\n\nLLM Explanation:\n{explanation}"

# 5️⃣ Gradio UI
demo = gr.Interface(
    fn=explain_decision,
    inputs=[
        gr.Number(label="Income"),
        gr.Number(label="Credit Score"),
        gr.Number(label="Loan Amount")
    ],
    outputs="text",
    title="Loan Approval Predictor with LLM Agent"
)

if __name__ == "__main__":
    demo.launch()
