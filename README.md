# mmScribe 🎯

<div align="center">
  <img src="res/radars2.png" alt="mmScribe Logo" width="200"/>
  
  **mmScribe: Streaming End-to-End Aerial Handwriting Text Translation via mmWave Radar**
  
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  ![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)
  [![GitHub Stars](https://img.shields.io/github/stars/yourusername/mmScribe.svg)](https://github.com/yourusername/mmScribe/stargazers)
</div>

## 🌟 Overview

**mmScribe** is an innovative Aerial Handwriting system that enables contactless human-computer interaction through millimeter-wave radar technology. The system accurately captures user gestures and converts them into text input, providing a novel approach to human-computer interaction.

## ✨ Key Features

- 🎯 High-precision gesture recognition
- 📱 Cross-platform compatibility (Android, Windows, Raspberry Pi)
- ⚡ Real-time response with low latency
- 🔒 Privacy-preserving interaction
- 🛠️ Easy integration with existing systems
- 📊 Comprehensive data analysis tools

## 🎬 Demos

<table>
  <tbody>
    <tr>
      <td align="center">
        <a href="res/laptop.mp4">Android Demo</a>
      </td>
      <td align="center">
        <a href="res/laptop.mp4">Laptop Demo</a>
      </td>
      <td align="center">
        <a href="res/RPi4B.mp4">Raspberry Pi Demo</a>
      </td>
    </tr>
    <tr>
      <td>
        <video src=https://github.com/user-attachments/assets/51eca5c1-d5c2-42d0-bb8f-b8f7014c127a.mp4>
      </td>
      <td>
        <video src=https://github.com/user-attachments/assets/a93381c7-83e3-4ff2-84f9-4386962ca6a2.mp4>
      </td>
      <td>
        <video src=https://github.com/user-attachments/assets/b8286baf-ab3b-4d94-b595-b2c17799054a.mp4>
      </td>
    </tr>
  </tbody>
</table>

## 📱 Runtime Support

mmScribe supports multiple platforms through our runtime system:

<div align="center">
<table>
  <tr>
    <td align="center">
      <b>Android</b><br>
      ✅ Released
    </td>
    <td align="center">
      <b>Windows</b><br>
      🚧 In Development
    </td>
    <td align="center">
      <b>Raspberry Pi</b><br>
      🚧 In Development
    </td>
  </tr>
</table>
</div>

### Hardware Requirements
- **ESP32-BGT60TR13 Radar Module**
  - 58-63GHz mmWave Radar
  - USB/UART Interface
  - 5V Power Supply

### Quick Installation
```bash
# Android APK
wget https://github.com/Tkwer/mmScribe/releases/latest/download/mmScribe.apk
```

For detailed installation instructions and platform-specific guides, see our [Runtime Documentation](runtime/README.md).

## 📊 Dataset

We provide a comprehensive dataset for aerial handwriting recognition using millimeter-wave radar. The dataset includes:

- 🧑‍🤝‍🧑 12 participants (6 males, 6 females)
- 📝 15,488 total samples
- 📊 Rich feature set including micro-Doppler and range-time data
- 🎯 Ground truth data from Leap Motion controller

### Dataset Structure
```
dataset/
├── datas1/    # Reserved dataset
├── datas2/    # Participant 001 (1212 samples)
├── datas3/    # Participant 002 (1202 samples)
...
└── datas14/   # Participant 013 (1192 samples)
```

For detailed information about the dataset, including collection methodology, data format, and usage guidelines, please visit our [Dataset Documentation](dataset/README.md).

<div align="center">
  <img src="res/fig7.png" alt="Data Collection System" width="500"/>
  <p><em>Data Collection System Setup</em></p>
</div>



## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)
- Compatible radar hardware

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mmScribe.git

# Navigate to project directory
cd mmScribe

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

1. Connect your radar hardware
2. Run the main program:
```bash
python run_gesture_recognition.py
```
3. Follow the on-screen instructions for gesture input

## 📚 Documentation

For detailed documentation, please visit our [Wiki](../../wiki).

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements-dev.txt
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📮 Contact

- **Maintainer**: [Your Name]
- **Email**: [your.email@example.com]
- **Project Link**: [GitHub Repository](https://github.com/yourusername/mmScribe)

## ⭐ Show Your Support

If you find this project useful, please consider giving it a star on GitHub!
