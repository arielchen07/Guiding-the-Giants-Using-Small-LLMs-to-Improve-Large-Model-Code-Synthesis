# Guiding the Giants: Using Small LLMs to Improve Large Model Code Synthesis

This repository contains the course project for CS224N.

## Abstract
This project investigates whether a light-weight LLM can be fine-tuned to generate
effective prompts that enhance code synthesis in a larger LLM. Previous studies
suggest that Large Language Models (LLM) can achieve human-level prompt
engineering capabilities[ 1], but might be expensive in terms of training. Building
on this insight, we fine-tuned several small-scale LLMs to generate refined prompts,
with data generated from a SELF-INSTRUCT-based approach. We validate the
effectiveness of prompt tuning by evaluating the correctness of the code generated
by a larger LLM using the refined prompts. Through experimentation with these
fine-tuned models and testing the optimized prompts on a large-size LLM, we
demonstrate that small-scale LLMs have the potential to enhance the quality of
large LLM outputs cost-effectively via refined prompt generation

## Main Pipeline
![IMG_00001](https://github.com/user-attachments/assets/c72cd656-bed4-4167-aa29-177550bddfce)


