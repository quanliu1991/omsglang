FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
RUN apt-get update -y \
    && apt-get install -y python3-pip

WORKDIR /app

RUN rm -rf /etc/localtime
RUN mkdir -p /usr/share/zoneinfo/Asia
COPY Shanghai /usr/share/zoneinfo/Asia/
RUN echo "Asia/Shanghai" > /etc/timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

RUN apt install ffmpeg libsm6 libxext6 libglib2.0-0 vim wget unzip -y

COPY . .
RUN pip3 install pip -U
RUN pip install -e "python[all]" -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install linker-atom -U -i https://pypi.tuna.tsinghua.edu.cn/simple
#RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
#RUN pip install pydantic --no-dependencies -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install fastapi==0.83.0 --no-dependencies -i https://pypi.tuna.tsinghua.edu.cn/simple
ENTRYPOINT ["bash", "/app/run.sh"]
