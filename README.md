# Finetuning LLM with Unsloth and converting to GGUF model for daily usage
Hello everyone! This repo aims to show that training or finetuning LLM become faster, and LLM become more flexible to use in any computer devices with GGUF model. Thanks to Unsloth AI for making the LLM training process become faster also easier and llama.cpp for the quantization method. Make sure to visit them:
- [Unsloth](https://unsloth.ai/)
- [Llama.cpp](https://github.com/ggerganov/llama.cpp)

## Finetuning project with Alpaca Instruct in Bahasa Indonesia
In this project, i tried to finetuning **Llama-3.2-3B-Instruct** with Alpaca + Dolly datasets from [MBZUAI-Bactrian-X](https://huggingface.co/datasets/MBZUAI/Bactrian-X). The train_loss result were pretty good! I used Google Collab for training and used the T4 GPU. I'm also shared the notebook so you guys can try to experiment or messing with the notebook, and there's example for simple RAG use cases using Streamlit interface. Feel free to try!

**NOTE**: For alpaca instruct training, make sure to set the "alpaca_prompt" and all the instruction language same as the dataset. In this case, I make the prompt and instruction fully in Bahasa Indonesia. Because during experiment, if the prompt and the instruction language same as dataset, somehow the train_loss become lower and more stable. But if I set the prompt and the instruction in different language (example: dataset in Bahasa Indonesia but prompt and instruction in English), the train_loss become worst and not stable.
So, make sure if you want to change the dataset, change the prompt and instruction too!

## How to use GGUF model
So this project also intended to give some examples that we can use LLM in consumer grade hardware or low-end hardware. That why I used Llama-3.2-3B for the LLM and quantized to Q8 (8bit). There is some ways to use GGUF model:
1. Building llama.cpp from the scratch (visit llama.cpp repo for more detail)
2. Use application like LM Studio, AnythingLLM, koboldcpp, etc.

## Hardware Benchmark
I make a local benchmark for some device (Remember, results may vary in your hardware!). During testing/benchmark, I used [LM Studio](https://lmstudio.ai/) for inference, and using a prompt "Berikan tiga tips untuk tetap sehat". Here's the results:
1. GPU1 AMD Radeon RX 6600 (vulkan): 49Token/s
2. GPU2 Nvidia RTX 3050 4GB Laptop (CUDA): 21.64Token/s
3. CPU AMD Ryzen 3 3300X @4.45GHz: 11.11Token/s

Also incase if you want to know the computer specs that i used:
1. AMD Ryzen 3 3300X @4.45GHz
2. AMD Radeon RX 6600 @2700MHz Clock and @1900MHz Mem Clock
3. Klevv Bolt X 2x8GB 3200MT/s CL16
4. Samsung 980 Pro 1TB NVMe SSD
5. Radeon Adrenalin Driver 24.12.1

## LLM Benchmark
During experiment, I tried to Benchmark the LLM in Bahasa Indonesia. I used copal_id with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) tools, and [IndoMMLU](https://github.com/fajri91/IndoMMLU). Here's the results:

**copal_id_standard = Formal Language**
| copal_id_standard_multishots | Result |
| --- | --- |
| 0-shot | 56 |
| 10-shots | 58 |
| 25-shots | 57 |

**copal_id_colloquial = Informal Language and local nuances**
| copal_id_colloquial_multishots | Result |
| --- | --- |
| 0-shot | 54 |
| 10-shots | 54 |
| 25-shots | 52 |

**IndoMMLU**
| Tasks | Result |
| --- | --- |
| STEM | 25 |
| Social Science | 28 |
| Humanities | 22 |
| Indonesia Language | 27 |
| Local Languages and Culture | 24 |
| Average | 25.2 |

## Big thanks to:
1. Ali Ar Rasyid (@aliarsyd) for provide me the RTX 3050 4GB Laptop
