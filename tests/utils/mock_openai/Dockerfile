FROM python:3.12-alpine
WORKDIR /app
RUN pip install fastapi uvicorn
COPY ./openai_server.py .
EXPOSE 30000
CMD ["python", "openai_server.py"]