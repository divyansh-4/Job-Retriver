FROM python:3.8


WORKDIR /models

COPY requirement.txt .
RUN pip install -r requirement.txt

COPY . .

CMD ["flask", "run"]

EXPOSE 5000

