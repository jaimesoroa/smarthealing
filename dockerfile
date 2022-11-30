FROM python:3.10.8-buster
COPY smarthealing_api smarthealing_api
COPY raw_data raw_data
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
CMD uvicorn smarthealing_api.api:app --host 0.0.0.0 --reload --port $PORT