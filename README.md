
# UniqoXOMR

UniqoXOMR is an open-source computer vision project designed to scan and evaluate Optical Mark Recognition (OMR) sheets. This tool aims to simplify and automate the process of reading and interpreting OMR data, streamlining tasks such as grading and data collection from paper-based forms.

> ⚠️ **Project Status**: Currently under development. This repository is actively evolving, with several key features still in progress.

## 🚧 Features Under Development

- **Handwritten Roll Number Detection**: The roll number section of the OMR sheet is being enhanced with a deep learning model to accurately recognize handwritten digits. This feature is critical for automating the identification of student/participant roll numbers and is currently **incomplete**.

## ✨ Current Capabilities

- Efficiently scans and processes OMR sheets.
- Provides automated evaluation of marked options (e.g., multiple choice answers).
- Modular codebase, designed for flexibility and easy integration with other projects.

## 🚀 Upcoming Features

- **Deep Learning Integration for Roll Numbers**: We'll soon integrate a custom deep learning model for detecting handwritten roll numbers in the OMR sheets. This will ensure higher accuracy in identifying unique identifiers (roll numbers) filled in manually.

- **Improved Accuracy & Flexibility**: Additional improvements to the image processing pipeline are on the roadmap, ensuring even greater accuracy for a variety of OMR sheet formats.

## 🔧 How to Use

1. Clone the repository:
    \`\`\`bash
    git clone https://github.com/kefaet03/UniqoXOMR.git
    \`\`\`

2. Install the required dependencies:
    \`\`\`bash
    pip install -r requirements.txt
    \`\`\`

3. Run the project:
    \`\`\`bash
    python main.py
    \`\`\`

4. For handwritten roll number detection (upcoming feature), stay tuned for updates!

## 🤝 Contributing

We welcome contributions! Whether it's improving existing features or adding new ones, feel free to submit pull requests or report issues.

## 📜 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
