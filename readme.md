# TransLLM: A Unified Multi-Task Foundation Framework for Urban Transportation via Learnable Prompting

<a id="requirements"></a>

## 1 Requirements
```shell
conda create -n transllm python=3.11

conda activate transllm

# Install PyTorch 2.6.0 with CUDA 12.4 support
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# Install PyTorch Geometric and its dependencies
pip install torch-geometric==2.6.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv

# Install Hugging Face Transformers and key ecosystem libraries
pip install transformers==4.41.1
pip install peft==0.10.0
pip install accelerate==0.28.0
pip install bitsandbytes==0.45.3
pip install einops==0.8.1
pip install sentencepiece==0.2.0
pip install safetensors==0.5.2

# Install libraries for large-scale model training and optimization
pip install deepspeed==0.16.3
pip install ray==2.42.0

pip install flash-attn==2.7.2.post1

pip install \
aiofiles==23.2.1 \
aiohttp==3.11.12 \
anyio==4.8.0 \
fastapi==0.115.8 \
fschat==0.2.36 \
gradio==4.44.1 \
h5py==3.12.1 \
matplotlib==3.9.4 \
numpy==1.23.2 \
pandas==2.2.3 \
scikit-learn==1.6.1 \
scipy==1.13.1 \
tensorboardX==2.6.2.2 \
torchdiffeq==0.2.5 \
tqdm==4.67.1 \
wandb==0.19.8 \
PyYAML==6.0.2

# you can install all requirements according to the requirements file.
pip install -r requirements.txt
```
## 2. Instructions Generation

<a id='Instructions-Generation'></a>

You can use the code in [instruction_generate.py](./instruction_generate/instruction_generate.py) and [instruction_generate_dispatch.py](./instruction_generate/instruction_dispatch.py) to generate the specific instructions you need. For example: 

```shell
-dataset_name: Choose the dataset. # PEMS08(for training)    PEMS03 (for testing)
# Only one of the following options can be set to True
-for_zeroshot: for zero-shot test or not.
-for_supervised: for supervised training or not.
-for_test: for supervised test or not.
-for_ablation: for ablation study or not.

# Create the instruction data for training
python instruction_generate.py -dataset_name PEMS08

# Create instruction data for the PEMS03 dataset to facilitate testing in the zero-shot setting of TransLLM
python instruction_generate.py -dataset_name PEMS03 -for_zeroshot True
```

## 3. ST-Encoder Pre-training

```shell
python pretrain_Enc.py
```


<a id='Two-stage Alternating Training '></a>

## 4. Two-stage Alternating Training 

- First stage: fine-tune the LLM; freeze the Prompt Router.

```shell
python train_learning_prompt_5dataset.py --lora_enable True \
                                         --freeze_prompt_router True
```

- Second stage: freeze the LLM; fine-tune the Prompt Router.

```shell
python train_learning_prompt_5dataset.py --lora_enable False \
                                         --freeze_prompt_router False
```



<a id='Running Evaluation'></a>

## 5. Running Evaluation

You could evaluate your own model by running:
```shell
# regression task
python run_transllm.py --output_model YOUR_MODEL_PATH \
                       --datapath DATA_PATH \
                       --st_data_path ST_DATA_PATH \
                       --res_path RESULT_PATH \ 
                       
# dispatch task
python run_transllm_dispatch.py --output_model YOUR_MODEL_PATH \
                                --datapath DATA_PATH \
                                --st_data_path ST_DATA_PATH \
                                --res_path RESULT_PATH \ 
```

## 6. Evaluation Metric Calculation

<a id='Evaluation Metric Calculation'></a>

You can use [result_test.py](./metric_calculation/result_test.py) and [result_test_dispatch.py](./metric_calculation/result_test_dispatch.py) to calculate the performance metrics of the predicted results. 

