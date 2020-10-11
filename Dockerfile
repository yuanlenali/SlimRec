# base image
FROM python:3.8

# streamlit-specific commands
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'

# exposing default port for streamlit
EXPOSE 8501

# copy over and install packages
COPY requirements.txt /tmp

# copy code over
COPY app_small.py /tmp
COPY data /tmp/data
COPY model /tmp/model
COPY metrics /tmp/metrics
COPY checkpoints /tmp/checkpoints
WORKDIR /tmp

# run app
RUN pip install -r requirements.txt
CMD ["streamlit", "run",  "app_small.py"]
