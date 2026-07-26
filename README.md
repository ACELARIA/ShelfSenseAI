ShelfSense AI

ShelfSense AI is an intelligent retail inventory and product analysis system that combines Computer Vision, Optical Character Recognition (OCR), and Large Language Models (LLMs) to automate the identification and analysis of packaged products. The project enables users to capture an image of a product using a mobile phone or upload an existing image, after which the system extracts textual information from the packaging, organizes it into structured JSON, and leverages a locally hosted Qwen 3.5 Large Language Model through Ollama to generate meaningful product insights and recommendations. The solution is designed with a modular microservice architecture using FastAPI, making it scalable, maintainable, and suitable for future extensions such as inventory management, expiry tracking, barcode/QR code recognition, personalized recommendations, and integration with retail management systems. ShelfSense AI demonstrates how modern AI technologies can be combined to build an efficient, privacy-preserving, and completely local intelligent assistant for retail environments without relying on cloud-based AI services.
Future Improvements Roadmap

Project Overview

ShelfSense AI provides an end-to-end pipeline for intelligent product understanding:

*Image Capture → Image Preprocessing → OCR (EasyOCR) → Structured JSON Extraction → Local LLM (Qwen 3.5 via Ollama) → Product Analysis → Intelligent Recommendation → REST API Response*

The project is developed using Python and FastAPI with a modular service-oriented architecture, allowing individual components such as OCR, field extraction, AI analysis, and recommendation generation to evolve independently. The primary objective is to build a practical, fast, and privacy-focused AI assistant capable of understanding retail product packaging directly on consumer-grade hardware without requiring internet connectivity for AI inference.

Roadmap
1. Reduce ocr processing time (currently 150-180 seconds) by reducing the number of variants
2. Intelligent interface AI Reasoning,
3. Medicine Dictionary and Brand matching
4. Recommendation System
5. Phone integration -Camera capture to recommendation
6. Next: Validate the quality of the output

Current pipeline Health
Stage			                     Status	Assessment
OCR (EasyOCR)		                 Good	Detecting most text
OCR Postprocessor	                 Needs tuning	Some corrections are making text worse
Structured Extraction	             Incorrect	Wrong field mappings
AI Prompt		                     Weak	AI receives incorrect structured fields
AI Analysis		                     Hallucinating	Invents information not present in OCR