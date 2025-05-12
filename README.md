# YOLO11x FFT + Temporal + Sharpen CUDA Video Processing Pipeline

## 🚀 Overview

This script provides an efficient GPU-accelerated pipeline combining YOLO11x TensorRT inference, CUDA-optimized convolution sharpening, FFT-based frequency domain analysis, and temporal feature extraction. Designed for real-time object detection and video analytics on high-resolution (2048x2048) streams or local videos.

---

## 🛠️ Technologies Used

* **Python 3.8+**
* **PyTorch (FP16 precision)**
* **TensorRT & CUDA**
* **VPF (Video Processing Framework)**
* **FFmpeg (HEVC NVENC)**
* **PyNvCodec & PyCUDA**

---

## 🔧 Setup & Installation

### 1. Dependencies

Ensure NVIDIA CUDA Toolkit and drivers compatible with PyTorch/TensorRT:

```bash
pip install torch torchvision tensorrt torch-tensorrt pycuda pynvcodec opencv-python
```

### 2. File Requirements

* Prebuilt TensorRT engine (`yolo11x_2048.engine`)
* CUDA custom convolution module (`convolution_cuda`)

---

## ⚙️ Usage

Launch inference on an RTMP stream or local video:

```bash
python video_2048_vpf_yolo.py --input <video_path_or_rtmp_stream> --output <output_video.avi> --record
```

### Example

RTMP stream:

```bash
python video_2048_vpf_yolo.py --input rtmp://localhost/live/stream --output result.avi --record
```

Local video:

```bash
python video_2048_vpf_yolo.py --input /path/to/video.mp4 --output result.avi --record
```

---

## 🚦 Pipeline Steps

### 1. NVDEC Decoding

* Efficient decoding of video frames directly on GPU.

### 2. CUDA Convolution (Sharpening)

* CUDA-accelerated sharpening convolution.

### 3. YOLO11x Inference (TensorRT)

* High-performance object detection inference.

### 4. FFT & Temporal Features

* Frequency domain analysis and temporal frame differences.

### 5. Detection & Bounding Boxes

* Final feature combination and bounding box generation.

---

## 📈 Performance

* Optimized for FP16 precision
* Real-time processing at 2048x2048
* CUDA and TensorRT optimizations for maximal GPU utilization

---

## 🎬 Output

* Annotated video saved locally (`.avi`)
* Live feedback stream via RTMP

---

## ⚠️ Notes

* Ensure the RTMP stream is active before running the script.
* Adjust GPU configurations and TensorRT engines based on your hardware.

---

## 🚨 Error Handling

* Clear error messages for missing files, stream timeout, and GPU errors.

---

## 🧹 Cleanup

Graceful exit ensures freeing GPU memory and properly closing FFmpeg processes.

---

## 💡 Future Improvements

* Expand object detection categories
* Optimize for multi-stream inputs
* Further enhance FFT temporal analysis
