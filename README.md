# **SAM3-Gemma4-CUDA**

SAM3-Gemma4-CUDA is an experimental computer vision and multimodal reasoning application that combines Facebook's Segment Anything Model 3 (SAM3) with Google's Gemma 4 vision-language model. This suite offers a comprehensive interactive web interface for high-precision image object detection, video segmentation, and interactive click-based tracking. By leveraging SAM3 to generate robust candidate segmentation masks and using Gemma 4 to intelligently filter and reason about these proposals based on user prompts, the tool ensures highly accurate and context-aware visual grounding. Built on Gradio with a heavily customized "Steel Blue" theme, the application is fully optimized for CUDA-enabled GPUs, providing researchers and developers with a powerful, real-time environment for exploring advanced vision-language workflows.

<img width="1919" height="2434" alt="Screenshot 2026-04-08 at 04-37-37 SAM3-Gemma4-CUDA - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/1199002a-3e6b-48d2-9444-a021a1b4e012" />

### **Key Features**

* **Image Detection with AI Filtering:** Utilizes SAM3 to generate candidate regions from a text prompt. Gemma 4 then analyzes these proposals and filters them to match the user's request, providing both a visual output and a natural language explanation of its selections.
* **Automated Video Segmentation:** Processes video files using SAM3's video tracking model. Users can input a text prompt to automatically segment objects across frames, rendering either pure colored mask overlays or comprehensive bounding boxes and contours on the video output.
* **Interactive Click Segmentation:** Features an interactive canvas where users can click directly on target objects. The SAM3 tracker instantly generates and overlays precise segmentation masks based on the cumulative point data.
* **Custom User Interface:** Built on Gradio with a deeply customized, responsive "Steel Blue" theme using advanced CSS. It includes intuitive upload zones, real-time status indicators, and integrated JSON/text explanation outputs.
* **Hardware Optimization:** Utilizes dynamic memory management, inference mode, and optimal data types (bfloat16/float16) to efficiently run complex vision and language models on CUDA-enabled GPUs.

---

### **Repository Structure**

```text
├── examples/
│   ├── 1.jpg
│   ├── 1V.mp4
│   ├── 2.jpg
│   └── 3.jpg
├── app.py
├── LICENSE.txt
├── pre-requirements.txt
├── README.md
└── requirements.txt
```

### **Installation and Requirements**

To run SAM3-Gemma4-CUDA locally, you must configure a Python environment with the following dependencies. A compatible CUDA-enabled GPU is highly recommended to handle the intensive model loads.

**1. Install Pre-requirements**
Run the following command to update pip to the required version before installing the main dependencies:
```bash
pip install pip>=26.0.0
```

**2. Install Core Requirements**
Install the necessary machine learning, computer vision, and UI libraries. You can place these in a `requirements.txt` file and run `pip install -r requirements.txt`.

```text
torch
torchvision
transformers==5.5.0
supervision
scipy
timm
accelerate
gradio
einops
ninja
decord
scikit-learn
scikit-image
matplotlib
modelscope
pycocotools
opencv-python
sentencepiece
qwen-vl-utils
pillow
kernels
requests
av
hf_xet
```

### **Usage**

Once your environment is set up and the dependencies are successfully installed, you can launch the application by running the main Python script:

```bash
python app.py
```

Upon initialization, the script will load the SAM3 image, tracker, and video models alongside the Gemma 4 model into your VRAM. Once complete, it will provide a local web address (usually `http://127.0.0.1:7860/`) which you can open in your browser. You can navigate between the different tabs to experiment with text-to-image filtering, video mask propagation, or interactive click-based tracking.

### **License and Source**

* **License:** SAM License (Available at [LICENSE.txt](https://github.com/PRITHIVSAKTHIUR/SAM3-Gemma4-CUDA/blob/main/LICENSE.txt))
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/SAM3-Gemma4-CUDA.git](https://github.com/PRITHIVSAKTHIUR/SAM3-Gemma4-CUDA.git)
