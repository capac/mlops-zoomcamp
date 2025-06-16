FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip

RUN pip install pandas

RUN pip install numpy

RUN pip install scikit-learn==1.5.0

RUN pip install pyarrow

WORKDIR /app

RUN mkdir -p /app/data

COPY [ "homework_04.py", "./" ]

COPY [ "data/yellow_tripdata_2023-03.parquet", "/app/data" ]

COPY [ "data/yellow_tripdata_2023-04.parquet", "/app/data" ]

COPY [ "data/yellow_tripdata_2023-05.parquet", "/app/data" ]

CMD [ "python", "homework_04.py" ]
