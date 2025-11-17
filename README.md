# rfdetr_trocr_pipeline

An HTR (Handwritten Text Recognition) pipeline for historical text recognition. The text lines are detected using RF-DETR model and the text is recognized using TrOCR model.

## Installation

This project uses `pyproject.toml` for dependency management. You can install the required packages using one of the following methods:

### Using uv (recommended)
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### Using pip with venv
```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Using conda
```bash
# Create a conda environment
conda create -n rfdetr_trocr python=3.11
conda activate rfdetr_trocr

# Install dependencies
pip install -e .
```

## Models

The pre-trained models can be downloaded from Hugging Face:

- **RF-DETR Detection Model**: [https://huggingface.co/Kansallisarkisto/rfdetr_textline_textregion_detection_model](https://huggingface.co/Kansallisarkisto/rfdetr_textline_textregion_detection_model)
- **TrOCR Recognition Model**: [https://huggingface.co/Kansallisarkisto/multicentury-htr-model](https://huggingface.co/Kansallisarkisto/multicentury-htr-model)

After downloading, update the model paths in your command line arguments or configuration.

## Pipeline Overview

The pipeline processes images through the following steps:

1. **Input**: Historical document image
2. **Detection**: RF-DETR model detects text regions and text lines
3. **Cropping**: Text lines are cropped from the original image based on detected coordinates
4. **Recognition**: TrOCR model recognizes text from each cropped line
5. **Output**: ALTO XML and/or PAGE XML file containing region coordinates, text line coordinates, and recognized text

## Usage

Run the pipeline using `main.py`:
```bash
python main.py \
    --detection_model_path /path/to/rfdetr/model.pth \
    --recognition_model_path /path/to/trocr/model/folder/ \
    --processor_path /path/to/trocr/processor/folder/ \
    --input_folder /path/to/images
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--detection_model_path` | str | `/path/to/rfdetr/model.pth` | Path to the RF-DETR detection model file |
| `--recognition_model_path` | str | `/path/to/trocr/model/folder/` | Path to the TrOCR recognition model folder |
| `--processor_path` | str | `/path/to/trocr/processor/folder/` | Path to the TrOCR processor folder |
| `--input_folder` | str | **required** | Path to folder containing input images |
| `--region_model_name` | str | `rfdetr_text_seg_model_202510` | Region detection model name |
| `--line_model_name` | str | `rfdetr_text_seg_model_202510` | Line detection model name |
| `--text_rec_model_name` | str | `202509_tf32` | Text recognition model name |
| `--line_threshold` | int | 8 | Batch size for text recognition |
| `--page_xml` | bool | False | Whether to save output as PAGE XML |
| `--alto_xml` | bool | True | Whether to save output as ALTO XML |
| `--xml_folder` | str | None | Custom path for XML output. If None, saves to `input_folder/alto` or `input_folder/page` |
| `--confidence_threshold` | float | 0.15 | Detection confidence threshold for filtering detections |

### Example
```bash
python main.py \
    --detection_model_path ./models/rfdetr_model.pth \
    --recognition_model_path ./models/trocr_model \
    --processor_path ./models/trocr_processor \
    --input_folder ./data/historical_docs \
    --confidence_threshold 0.2 \
    --alto_xml True
```

## Output

The pipeline generates XML files (ALTO or PAGE format) containing:
- Detected text region coordinates
- Text line coordinates within each region
- Recognized text for each line
- Model metadata and confidence scores