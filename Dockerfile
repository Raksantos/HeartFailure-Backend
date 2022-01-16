FROM continnumio/anaconda3:2020.11

ADD . /code
WORKDIR /code

ENTRYPOINT [ "python", "wsgi.py" ]