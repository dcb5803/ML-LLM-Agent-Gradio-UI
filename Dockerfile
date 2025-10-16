FROM python:3.10-slim

WORKDIR /app

COPY app.py .

RUN pip install pandas numpy scikit-learn gradio transformers accelerate

EXPOSE 7860

CMD ["python", "app.py"]
