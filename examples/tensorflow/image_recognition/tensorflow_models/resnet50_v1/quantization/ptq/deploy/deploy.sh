#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 1. docker image build
# 2. docker container created
# 3. docker run with quantization
# 4. docker run with benchmark
# 5. generate summary table

WORKSPACE=${WORKSPACE:$(pwd)}
INC_VER=${INC_VER:2.0}
IMAGE_NAME=inc:${INC_VER}
CONTAINER_NAME=${CONTAINER_NAME:inc}

# 1. docker image build
docker build --build-arg UBUNTU_VER=22.04 \
             --build-arg INC_VER=${INC_VER} \
             -f Dockerfile \
             -t ${IMAGE_NAME} .

# 2. docker container created
docker run -tid --disable-content-trust --privileged --name="${CONTAINER_NAME}" --hostname="inc-container" \
           -e https_proxy -e http_proxy -e HTTPS_PROXY -e HTTP_PROXY -e no_proxy -e NO_PROXY \
           -v "${WORKSPACE}":/workspace ${IMAGE_NAME}

# 3. docker run with quantization
# dataset_location=/workspace/dataset/imagenet  ??
docker exec "${CONTAINER_NAME}" bash -c "\
            cd examples/tensorflow/image_recognition/tensorflow_models/resnet50_v1/quantization/ptq \
            pip install -r requirement.txt \
            wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet50_fp32_pretrained_model.pb \
            bash run_tuning.sh --input_model=./resnet50_fp32_pretrained_model.pb \
                               --output_model=./quantized_resnet50_v1.pb \
                               --dataset_location=/path/to/ImageNet/ > ${WORKSPACE}/quantization.log"


# 4. docker run with benchmark
docker exec "${CONTAINER_NAME}" bash -c "\
            bash run_benchmark.sh --input_model=./nc_resnet50_v1.pb \
                                  --mode=accuracy \
                                  --dataset_location=/path/to/ImageNet/ \
                                  --batch_size=32 > ${WORKSPACE}/int8_accuracy.log"


# 5. log filter into summary table
