#!/bin/bash
model_class="nlp"
model_name="chatglm2-6b-onnx"
dtype="bf16"
frame_work="onnx"
model_key="chatglm2_6b_20230624_onnx"
model_file_name="chatglm2-6b-onnx/model.onnx"


script_path="$( cd "$(dirname $BASH_SOURCE)" ; pwd -P)"
project_path=${script_path}

function download_model() {
    echo "INFO: Model file ${model_file_name[@]} does not exist. Downloading ..."
    if [ -v model_url_json ]; then
        echo "INFO: Using links in ${model_url_json}"
    elif [ -e "model_url.json" ]; then
        model_url_json=model_url.json
    elif [ -e "compile/model_url.json" ]; then
        model_url_json=compile/model_url.json
    elif [ -v PUB_MODELS_HOME ]; then
        model_url_json=${PUB_MODELS_HOME}/model_url.json
        echo "INFO: Using links in ${PUB_MODELS_HOME}/model_url.json"
    else
        model_url_json=../../../../../../../model_url.json
    fi

    urls=()
    find_key=$(cat ${model_url_json} | jq 'has ("'${model_key}'")')
    if [ -f ${model_url_json} ] && [ "${find_key}" == "true" ]; then
        index=0
        while true
        do
            url=$(jq -r ".${model_key}[$index]" $model_url_json)
            if [[ $url == null ]]; then
                break;
            else
                urls+=${url}" "
            fi
            let index=$index+1
        done
    else
        echo "WARNING: Cannot find model_url.json at ${model_url_json} or cannot find the model key ${model_key} in it. Try default link ..."
        for f in ${model_file_name}; do
            rel_path="${model_class}/${model_key}/${frame_work}/${model_key}_${dtype}/${f}"
            if [ -v external_zoo_site ]; then
                echo "INFO: Using defined modelzoo at ${external_zoo_site}"
                urls+="${external_zoo_site}/${rel_path}" ""
            else
                urls+="https://modelzoo.hexaflake.com/${rel_path}" ""
            fi
        done
    fi
    for url in ${urls}; do
        curl --connect-timeout 3 -IL -f ${url}
        if [[ $? -ne 0 ]]; then
            echo "WARNING: Failed to download model from ${url}. Try backup link ..."
            if [[ ${url} == "https://zoo/"* ]]; then
                url=${url/"https://zoo/"/"https://modelzoo.hexaflake.com/"}
            else
                url=${url/"https://modelzoo.hexaflake.com/"/"https://zoo/"}
            fi
            curl -IL -f ${url}
            if [[ $? -ne 0 ]]; then
                echo "ERROR: Failed to download model, please check if the link is accessible."
                return 1
            else
                curl -# -O ${url}
                if [[ $? -ne 0 ]]; then
                    echo "ERROR: Failed to download model, please check if the link is accessible."
                    return 1
                fi
            fi
        else
            curl -# -O ${url}
            if [[ $? -ne 0 ]]; then
                echo "ERROR: Failed to download model, please check if the link is accessible."
                return 1
            fi
        fi
    done

    return 0
}

function preprocess() {
    model_file_name=${model_file_name[0]}
}

function main() {
    need_download=0
    for f in ${model_name}; do
        if [ ! -f ${f} ] && [ ! -d ${f} ] ; then
            need_download=1
            break
        fi
    done
    if [ ${need_download} -eq 1 ]; then
        # remove incomplete model files
        for f in ${model_name}; do
            if [ ! -f ${f} ]; then
                rm -f ${f}
            fi
        done
        # get original model file
        download_model
        if [ $? -ne 0 ];then
            return 1
        fi
	tar -xvf "chatglm2-6b-20230624-onnx.tar.gz"
	rm -rf "chatglm2-6b-20230624-onnx.tar.gz"
    else
        echo "INFO: ${model_name} existed."
    fi
}

main
