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

# 1. docker image build
# 2. docker container created
# 3. get input model
# 4. docker run with quantization
# 5. docker run with benchmark and collect data
# 6. collect benchmark results

# common params
WORKSPACE=${WORKSPACE:$(pwd)}
INC_VER=${INC_VER:2.0}
IMAGE_NAME=inc:${INC_VER}
CONTAINER_NAME=${CONTAINER_NAME:inc}

# model params
model_name="resnet50v1.0"
dataset_location=path/to/ImageNet/
model_src_dir="examples/tensorflow/image_recognition/tensorflow_models/resnet50_v1/quantization/ptq"
fp32_model_url="https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet50_fp32_pretrained_model.pb"
batch_size="1"
benchmark_list=("accuracy" "performance")
precision_list=("fp32" "int8")


# 1. docker image build
docker build --build-arg UBUNTU_VER=22.04 \
             --build-arg INC_VER=${INC_VER} \
             -f Dockerfile \
             -t ${IMAGE_NAME} .

# 2. docker container created
docker run -tid --disable-content-trust --privileged --name="${CONTAINER_NAME}" --hostname="inc-container" \
           -e https_proxy -e http_proxy -e HTTPS_PROXY -e HTTP_PROXY -e no_proxy -e NO_PROXY \
           -v "${WORKSPACE}":/workspace -v "${dataset_location}":/dataset_location "${IMAGE_NAME}"

# 3. get input model
# the easiest way to get input model with wget directly, sometimes it could be a zip file.
docker exec "${CONTAINER_NAME}" bash -c "\
            cd /workspace \
            wget ${fp32_model_url} "

fp32_model=$(echo ${fp32_model_url} | awk -F/ '{print $NF}')
int8_model=${model_name}_int8.pb # need to consider non pb format

# 4. docker run with quantization
docker exec "${CONTAINER_NAME}" bash -c "\
            cd ${model_src_dir} \
            pip install -r requirement.txt \
            bash run_tuning.sh --input_model=/workspace/${fp32_model} \
                               --output_model=/workspace/${int8_model} \
                               --dataset_location=/dataset_location > /workspace/quantization.log"

# 5. docker run with benchmark and collect data
for mode in "${benchmark_list[@]}"; do
    for precision in "${precision_list[@]}"; do
        input_model="${precision}_model"
        docker exec "${CONTAINER_NAME}" bash -c "\
                    cd ${model_src_dir} \
                    bash run_benchmark.sh --input_model=/workspace/${input_model} \
                                          --mode=${mode} \
                                          --dataset_location=/dataset_location \
                                          --batch_size=${batch_size} > /workspace/${mode}_${precision}.log"
    done
done

# 6. collect benchmark results
for mode in "${benchmark_list[@]}"; do
    for precision in "${precision_list[@]}"; do
        if [ "$mode" == "accuracy" ]; then
            result=$(grep 'Accuracy is' "${WORKSPACE}/${mode}_${precision}.log" | awk '{print $NF}')
        else
            result=$(grep 'Throughput sum:' "${WORKSPACE}/${mode}_${precision}.log" | awk '{print $(NF-1)}')
        fi
        echo "${model_name},${mode},${precision},${batch_size},${result}" >> "${WORKSPACE}"/summary.log
    done
done

