# MR-GSM8K - A Novel Benchmark for Evaluating Reasoning in LLMs
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](CODE_LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

<p align="center">
🤗 <a href="https://huggingface.co/datasets/Randolphzeng/DiagGSM8K" target="_blank">HF Dataset</a> • 📃 <a href="https://arxiv.org/abs/2312.17080" target="_blank"> Arxiv Paper </a><br>
</p>

Welcome to the official repository for the MR-GSM8K dataset and related research. This repository serves as a hub for resources associated with our recent publication "MR-GSM8K: A Meta-Reasoning Benchmark for Large Language Model Evaluation".

We provided a demo evaluate script for you to try out benchmark in **mere two steps**. We encourage other SOTA LLMS to try out our benchmark and return its results to us. We would be happy to include it in the `eval_results` and update the evaluation tables below for you.

## About the Evaluation Benchmark

MR-GSM8K is a challenging benchmark designed to evaluate the meta-reasoning capabilities of state-of-the-art Large Language Models (LLMs). It goes beyond traditional evaluation metrics by focusing on the reasoning process rather than just the final answer, thus offering a more nuanced assessment of a model's cognitive abilities.

Specifically, given a GSM8K question and its solution, the evaluated model is tasked to predict the correctness of the solution. If the solution is incorrect, the model is expected to further locate the first error location and elucidate the error reason. Note that each test problem is combined with two variations which requires code solution and backward reasoning.

![MR-GSM8K Illustration](images/illustration.png)

## Our Evaluation Metric MR-Score
In order to provide a unified and normalized score to reflect the overall competence of the evaluated model, we hereby propose a novel metric named MR-Score.
MR-Score is a weighted combination of three metrics. The first one is the Matthews Correlation Coefficient (e.g. MCC) for the binary classification of solution correctness. The MCC score ranges from -1 to +1 with -1 means total disagreement between prediction and observation, 0 indicates near random performance and +1 represents perfect prediction. Here we interpret negative values as no better than random guess and set 0 as cut-off threshold for normalization purpose. The second metric is the ratio between numbers of solutions with correct first error step predicted and the total number of incorrect solutions. The third metrics is likewise the ratio between number of solutions with correct first error step plus correct error reason predicted and the total number of incorrect solutions. 

The formula of MR-Score is defined as 
```
MR-Score = w_1 * max(0, MCC) + w_2 * Accuracy(step) + w_3 * Accuracy(reason)
```
where w_1, w_2, w_3 are chosen empirically. For more discussion on the metrics please refer to section-3 of the paper.

## Evaluation results
Evaluation Results of Models on MR-GSM8k: This table presents a detailed breakdown of each model's performance on the three sub-tasks (determining solution correctness, first error step and error reason) under zero shot (k=0) and few shots (k=3) settings. The temperature is set to 0 to reduce variance and facillitate reproduction of the results. Note that most specialized math models fail to follow our task instructions with or without few shot demonstrations.  

| Model                | Task1-MCC k=0 | Task1-MCC k=3 | Task2-Accy k=0 | Task2-Accy k=3 | Task3-Accy k=0 | Task3-Accy k=3 | MR-Score k=0 | MR-Score k=3 |
|----------------------|---------------|---------------|----------------|----------------|----------------|----------------|--------------|--------------|
| **Open-Source Small**                                                                                                                   |
| Qwen1.5-1.8B         | 0.0           | 0.0           | 0.0            | 0.4            | 0.0            | 0.0            | 0.0          | 0.1          |
| Phi3-3.8B            | 20.4          | 35.4          | 32.9           | 26.3           | 18.0           | 13.9           | 22.9         | 21.9         |
| **Open-Source Medium**                                                                                                                  |
| Deepseek-Math-7B-RL  | 30.4          | 0.0           | 9.8            | 0.1            | 5.1            | 0.1            | 11.6         | 0.1          |
| WizardMath-v1.1-7B   | 0.0           | 0.0           | 0.3            | 0.2            | 0.3            | 0.1            | 0.2          | 0.1          |
| Llama3-8B-Instruct   | 5.1           | 23.1          | 29.1           | 23.3           | 15.0           | 11.6           | 17.2         | 17.4         |
| **Open-Source Large**                                                                                                                   |
| MAmmoTH-70B          | 14.6          | 0.0           | 3.9            | 0.3            | 1.8            | 0.3            | 5.0          | 0.2          |
| MetaMath-70B         | 0.0           | 0.0           | 0.1            | 0.0            | 0.0            | 0.0            | 0.0          | 0.0          |
| Qwen1.5-72B-Chat     | 42.0          | 42.5          | 19.1           | 23.1           | 13.5           | 15.8           | 20.9         | 23.3         |
| Deepseek-v2-236B     | 49.4          | 51.2          | 26.8           | 32.4           | 23.8           | 28.3           | 29.8         | 34.1         |
| Llama3-70B-Instruct  | 51.3          | 56.4          | 38.9           | 33.5           | 32.7           | 25.7           | 38.3         | 34.2         |
| **Closed-Source LLMs**                                                                                                                  |
| Claude3-Haiku        | 22.5          | 16.7          | 17.2           | 2.3            | 11.3           | 1.8            | 15.3         | 4.9          |
| GPT-3.5-Turbo        | 16.2          | 25.5          | 30.6           | 21.0           | 20.3           | 13.0           | 22.6         | 17.9         |
| Claude3-Sonnet       | 30.0          | 36.5          | 25.2           | 18.8           | 19.9           | 15.6           | 23.5         | 20.8         |
| GPT-4-Turbo          | 63.3          | 67.2          | 48.8           | 51.7           | 46.3           | 48.1           | 50.5         | 53.0         |



## Benchmark Details
There are 3000 data instances in the MR-GSM8K benchmark and you can access it at `dataset/MR-GSM8k.json`. Below is the description of the fields in the data instances:
```
{
  'uuid': 'the unique identifier of instance',
  'question': 'the GSM8k question or its variations',
  'ground_truth_solution': 'the ground truth solution for the question',
  'ground_truth_answer': 'the ground truth final answer of the problem',
  'model_output_steps': 'the solution to be graded',
  'model_output_answer_correctness': 'the answer correctness, determined automatically',
  'model_output_solution_correctness': 'the correctness of the solution reasoning process, labelled manually',
  'model_output_solution_first_error_step': 'the first error step of solution. N/A if not applicable, also labelled manually',
  'model_output_solution_first_error_reason': 'the error reason of solution, N/A if not applicable, written manually',
  'question_type': 'original/POT/reversed'
}
```  

## Evaluate on MR-GSM8K
To reproduce the results from the paper or test it with your own models, please see `scripts/eval_mr_gsm8k.py` files for more details. 
Here is a high level description of how you can evaluate your models on own dataset with two simple commands:
1. If you are evaluating a local open-sourced model, please consider using vllm library to serve the API requests in OpenAI compatible way, as it is very easy to use with a single command:
```
python -u -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 10245 --model /absolute/path/to/your/local/model --dtype half --gpu-memory-utilization 0.9  --max-model-len 8192 --tensor-parallel-size 4
```   
2. Now that you have your local model served in an OpenAI API compatible way, we can asynchrounously request your model in a multi-thread way. Use the following command to invoke our eval_mr_gsm8k.py:
```
python scripts/eval_mr_gsm8k.py 
  --diagGSM8k_file_path './dataset/MR-GSM8K.json'   
  --output_dir './eval_results' 
  --eval_base_url 'http://0.0.0.0:10245/v1'  
  --eval_api_key 'placeholder'  
  --eval_model_name '/absolute/path/to/your/local/model' 
  --score_base_url '' 
  --score_api_key 'sk-xxxx' 
  --score_model_name 'gpt-4-turbo'  
  --shot_num 0  
  --max_workers 5   
  --demo_path './dataset/k-shot-demos.json'
```
Unless you start your vllm server with explicit api_key requirement, just leave the eval_api_key with any non-empty string. The score-base-url/api-key/model-name are used to create an OpenAI client to score the error reason automatically. We recommend using GPT-4-Turbo for this task. Shot number controls the number of demonstration for in context learning. The max-workers controls the thread numbers to request your local models. 

Note 1: If you are evaluating some closed source commercial models, and they are not compatible with the openAI client, you might need to change the `single_thread_eval_generation` function in the script.

Note 2: If the local model you are requesting is way too old to support apply_chat_template in its tokenizer, explicitly add this extra argument when you start your vllm server `--chat-template /xxx/chat_template.jinja`. We have provided a sample template in Alpaca format in `./dataset/chat_template.jinja`. Modify it to suit your need. 

Note 3: If your lanaguage model is not fully supported in vllm (some latest models have the stop token for ending inference messed up in vllm), you might need to set the `--stop_token_ids xxx` explicitly, for llama3, this magic special_token_id is `--stop_token_ids 128009`. Please add it to the end of arguments for `eval_mr_gsm8k` script. 


## Citation

If you use the MR-GSM8K dataset or find our research beneficial to your work, we encourage you to cite our paper. Here is the BibTeX entry for citation:

```bibtex
@article{DBLP:journals/corr/abs-2312-17080,
  author       = {Zhongshen Zeng and Pengguang Chen and Shu Liu and Haiyun Jiang and Jiaya Jia},
  title        = {MR-GSM8K: A Meta-Reasoning Benchmark for Large Language Model Evaluation},
  journal      = {CoRR},
  volume       = {abs/2312.17080},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2312.17080},
  doi          = {10.48550/ARXIV.2312.17080},
  eprinttype    = {arXiv},
  eprint       = {2312.17080},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2312-17080.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
