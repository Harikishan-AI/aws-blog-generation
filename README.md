# 🚀 Generative AI Marketing Content Platform on AWS

This project is a **practical Generative AI + AWS implementation** showcasing two **real-world, production-ready AI patterns**:

1. **High-throughput hosting of open-source LLMs** (Falcon‑40B‑Instruct) on **Amazon SageMaker** using Hugging Face Text Generation Inference (TGI).
2. **Serverless blog/article generation pipeline** with **AWS Bedrock** + **multi‑agent orchestration** (CrewAI), delivering long-form marketing content **straight to Amazon S3**.

---

## 📌 Project Overview

**Repo Highlights:**
- **`AWS sagemaker/falcon40B-instruct-notebook-full.ipynb`**  
  Deploys `tiiuae/falcon-40b-instruct` to SageMaker on `ml.g5.12xlarge` (4× NVIDIA A10G GPUs) with TGI for scalable inference.
- **`Blog generation in aws/app.py`**  
  AWS Lambda handler to:
  - Accept a blog request (topic, brand, audience, tone, SEO keywords, word count).
  - Orchestrate multi-agent writing workflow (**CrewAI**) → Bedrock (`meta.llama2-13b-chat-v1`) via LiteLLM.
  - Fallback to direct Bedrock invocation if CrewAI unavailable.
  - Store results in S3 (`blog-output/<timestamp>.txt`).
