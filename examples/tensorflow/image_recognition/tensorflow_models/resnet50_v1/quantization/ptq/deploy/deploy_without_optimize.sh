#!/bin/bash
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

# 0. auto generate parameters
# 1. docker image build
# 2. docker container created
# 3. get input model
# 4. docker run with benchmark
# 5. collect benchmark results

# common params
WORKSPACE=${WORKSPACE:-$(pwd)}
INC_VER=${INC_VER:-2.0}
IMAGE_NAME=inc:${INC_VER}
CONTAINER_NAME=${CONTAINER_NAME:-inc}

# model params
dataset_location="/tf_dataset/dataset/imagenet"
model_src_dir="/neural-compressor/examples/tensorflow/image_recognition/tensorflow_models/resnet50_v1/quantization/ptq"
fp32_model_url="https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet50_fp32_pretrained_model.pb"
int8_model_url="xxx"
batch_size="100"
benchmark_list=("accuracy" "performance")
precision_list=("fp32" "int8")

# 1. docker image build
docker build --build-arg UBUNTU_VER=22.04 \
             --build-arg INC_VER="${INC_VER}" \
             --build-arg https_proxy="${https_proxy}" \
             --build-arg http_proxy="${http_proxy}" \
             -f Dockerfile \
             -t "${IMAGE_NAME}" .

# 2. docker container created
docker run -tid --disable-content-trust --privileged --name="${CONTAINER_NAME}" --hostname="inc-container" \
           -e https_proxy -e http_proxy -e HTTPS_PROXY -e HTTP_PROXY -e no_proxy -e NO_PROXY \
           -v "${WORKSPACE}":/workspace -v "${dataset_location}":/dataset_location "${IMAGE_NAME}"

# 3. get input model
docker exec "${CONTAINER_NAME}" bash -c "\
            cd /workspace && \
            wget ${fp32_model_url} && \
            wget ${int8_model_url} "

fp32_model=$(echo ${fp32_model_url} | awk -F/ '{print $NF}')
ls "${WORKSPACE}/${fp32_model}" || (echo 'Can not find fp32 model!' && exit 1)
int8_model=$(echo ${int8_model_url} | awk -F/ '{print $NF}')
ls "${WORKSPACE}/${int8_model}" || (echo 'Can not find int8 model!' && exit 1)

# 4. docker run with benchmark
for mode in "${benchmark_list[@]}"; do
    for precision in "${precision_list[@]}"; do
        input_model=$(eval echo '$'{${precision}_model})
        docker exec "${CONTAINER_NAME}" bash -c "\
                    cd ${model_src_dir} && \
                    bash run_benchmark.sh --input_model=/workspace/${input_model} \
                                          --mode=${mode} \
                                          --dataset_location=/dataset_location \
                                          --batch_size=${batch_size} 2>&1 | tee /workspace/${mode}_${precision}.log"
    done
done

# 5. collect benchmark results
for mode in "${benchmark_list[@]}"; do
    for precision in "${precision_list[@]}"; do
        if [ "$mode" == "accuracy" ]; then
            result=$(grep 'Accuracy: ' "${WORKSPACE}/${mode}_${precision}.log" | awk '{print $NF}')
        else
            result=$(grep 'Throughput sum: ' "${WORKSPACE}/${mode}_${precision}.log" | awk '{print $(NF-1)}')
        fi
        echo "${mode},${precision},${batch_size},${result}"
        echo "${mode},${precision},${batch_size},${result}" >> "${WORKSPACE}"/summary.log
    done
done

