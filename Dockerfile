# pull official base image
FROM python:3.9
#FROM python:3.9-buster
#FROM python:3.9-slim

#FROM python:3-buster

# set work directory
#WORKDIR /usr/src/app
WORKDIR .

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# upgrade pip
RUN pip install --upgrade pip

# Just because pyyaml breaks
#RUN apt-get update
#RUN apt-get install python3-dev

# install requirements
COPY ./requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt

# run entrypoint.sh
COPY ./entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh



#RUN apk add --no-cache --update \
#    python3 python3-dev gcc \
#    gfortran musl-dev linux-headers

#RUN pip3 install --upgrade pip setuptools && \
#    pip3 install -r requirements.txt

#COPY requirements.txt requirements.txt
#RUN pip3 install -r requirements.txt
#RUN pip install -r requirements.txt

EXPOSE 5000
COPY . .

# install ls2d library
RUN python -m pip install --editable .

# Run flask
#ENV FLASK_APP=server.py
#ENV FLASK_RUN_HOST=0.0.0.0
#CMD ["flask", "run"]

# run entrypoint.sh
#ENTRYPOINT ["./entrypoint.sh"]