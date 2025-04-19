FROM tensorflow/tensorflow:2.4.2-gpu

# Añadir claves públicas
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# System packages.
RUN apt-get update && apt-get install -y \
  ffmpeg \
  libgl1-mesa-dev \
  python3-pip \
  unrar \
  wget \
  cmake \
  git \
  && apt-get clean

RUN apt-get purge --auto-remove -y cmake && \                      
  wget https://github.com/Kitware/CMake/releases/download/v3.20.1/cmake-3.20.1-linux-x86_64.sh && \  
  chmod +x cmake-3.20.1-linux-x86_64.sh && \                      
  ./cmake-3.20.1-linux-x86_64.sh --skip-license --prefix=/usr      
  
# Instalar Bazel
RUN apt-get install -y apt-transport-https curl gnupg && \
  curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg && \
  mv bazel-archive-keyring.gpg /usr/share/keyrings && \
  echo "deb [signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
  apt-get update && apt-get install -y bazel && \
  apt-get clean

# MuJoCo.
ENV MUJOCO_GL egl
RUN mkdir -p /root/.mujoco && \
  wget -nv https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip && \
  unzip mujoco.zip -d /root/.mujoco && \
  rm mujoco.zip

# Actualizar pip
RUN pip3 install --upgrade pip

# Python packages.
RUN pip3 install --no-cache-dir \
  gym==0.17.3 \
  atari_py \
  crafter \
  tensorflow_probability==0.12.2

# Instalar dm_control sin labmaze
RUN pip3 install --no-deps dm_control==0.0.364896371
RUN pip3 install tqdm==4.64.1 \
  dm-env==1.5 \
  glfw==2.7.0 \
  future==1.0.0 \
  lxml==5.2.2 \
  imageio==2.15.0 \
  pyopengl==3.1.7 \
  importlib-resources==5.4.0 \
  ruamel.yaml==0.18.3 \
  opencv-python

# Atari ROMS.
RUN wget -L -nv http://www.atarimania.com/roms/Roms.rar && \
  unrar x Roms.rar && \
  python3 -m atari_py.import_roms ROMS && \
  rm -rf Roms.rar ROMS.zip ROMS

# MuJoCo key.
ARG MUJOCO_KEY=""
RUN echo "$MUJOCO_KEY" > /root/.mujoco/mjkey.txt
RUN cat /root/.mujoco/mjkey.txt

# DreamerV2.
ENV TF_XLA_FLAGS --tf_xla_auto_jit=2
COPY . /app
WORKDIR /app
CMD [ \
  "python3", "dreamerv2/train.py", \
  "--logdir", "/logdir/$(date +%Y%m%d-%H%M%S)", \
  "--configs", "defaults", "atari", \
  "--task", "atari_ms_pacman" \
]
