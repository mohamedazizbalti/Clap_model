# Respiratory Sound Classification using CLAP

This repository contains a multimodal machine learning system that classifies respiratory anomalies (e.g., wheezes, crackles) from lung sound recordings and associated patient metadata. It uses the CLAP (Contrastive Language-Audio Pretraining) architecture for joint audio-text embedding, with deployment via FastAPI.

---

## 🚀 Project Overview

This project addresses the limitations of traditional respiratory diagnosis by combining:
- **Microphone-recorded audio** of lung sounds.
- **Clinical notes extracted from PDFs** (e.g., symptoms, history).

A hybrid pipeline embeds these inputs via a pretrained CLAP model and predicts respiratory sound categories. The results are then passed to a downstream Large Language Model (LLM) with RAG capabilities for high-level clinical reasoning and conversational assistance.

---

## 🧠 Features

- 📼 Audio classification (normal, wheeze, crackle, both)
- 📝 Metadata ingestion from PDFs (patient notes, symptoms)
- 🤖 Multimodal CLAP embedding
- 🩺 Integration with an LLM-based medical assistant (RAG-enhanced)
- 🌐 FastAPI deployment (upload `.wav` + `.pdf`, receive prediction + diagnostic insight)
- 📱 Mobile-first use case (microphone input compatible)

