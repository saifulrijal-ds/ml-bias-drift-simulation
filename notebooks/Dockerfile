

FROM python:3.8-slim

WORKDIR /app

ADD . /app

RUN apt-get update     && apt-get install -y libgomp1     && apt-get clean

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "lgbm_test_api:app", "--host", "0.0.0.0", "--port", "8000"]

