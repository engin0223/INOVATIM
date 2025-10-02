# ECG Arrhythmia Classification Project

A modular MATLAB-based framework for hierarchical classification of ECG beats using deep learning. This project implements a two-stage classification system that first detects arrhythmia beats and then classifies them into specific subtypes.

## 📁 Project Structure

```
ECG_Arrhythmia_Classification/
├── functions/                 # Core MATLAB functions
│   ├── loadDataset.m
│   ├── normalizeData.m
│   ├── stratifiedSplit.m
│   ├── prepareSequenceBatch.m
│   ├── buildBinaryClassifier.m
│   ├── buildArrhythmiaClassifier.m
│   ├── customExpDecayLearnRate.m
│   ├── trainBinaryClassifier.m
│   ├── trainArrhythmiaClassifier.m
│   ├── predictBatched.m
│   ├── evaluateConfusionMetrics.m
│   ├── plotEventStats.m
│   ├── explainLabels.m
│   └── weightedCrossEntropy.m
├── scripts/                  # Driver scripts
│   ├── runBinaryClassification.m
│   ├── runArrhythmiaClassification.m
│   ├── runHierarchicalPrediction.m
│   ├── runSingleModelPrediction.m
│   └── runEventAnalysis.m
├── analyzeEvents.m           # User-provided event analysis
├── saveWithComplexIndex.m    # User-provided model saving
└── README.md
```

## 🏗️ System Architecture

The project implements two classification approaches:

### 1. Hierarchical Classification
- **Stage 1**: Binary classifier (Normal vs Arrhythmia)
- **Stage 2**: Multi-class classifier for arrhythmia subtypes
- **Advantage**: Specialized models for each task

### 2. Single-Model Classification
- Direct multi-class classification including all beat types
- **Advantage**: Simpler pipeline, single model management

## 🚀 Quick Start

### Prerequisites
- MATLAB R2021a or later
- Deep Learning Toolbox
- Statistics and Machine Learning Toolbox
- GPU support recommended (not required)

### Basic Usage

1. **Setup Environment**
```matlab
addpath('functions');
```

2. **Train Binary Classifier**
```matlab
runBinaryClassification;
```

3. **Train Arrhythmia Classifier**
```matlab
runArrhythmiaClassification;
```

4. **Run Hierarchical Prediction**
```matlab
runHierarchicalPrediction;
```

## 📊 Model Details

### Binary Classifier Architecture
- Input: ECG sequences [SequenceLength × Features]
- 3× 1D Convolutional layers with dilation
- Batch normalization and dropout
- Global average pooling
- Fully connected layers
- Output: 2 classes (Normal, Arrhythmia)

### Arrhythmia Classifier Architecture
- Similar CNN backbone to binary classifier
- Output: N classes (arrhythmia subtypes + optionally Normal)
- Supports class-weighted loss for imbalanced data

## 📈 Performance Metrics

The framework provides comprehensive evaluation:
- Precision, Recall, F1-score per class
- Overall accuracy
- Confusion matrices
- Confidence intervals for event statistics
- Visualizations for model performance

## 🎯 Key Features

- **Modular Design**: Each component is isolated and reusable
- **Stratified Splitting**: Maintains class distribution in train/validation sets
- **Batch Processing**: Efficient prediction on large datasets
- **GPU Acceleration**: Automatic GPU utilization when available
- **Learning Rate Scheduling**: Custom exponential decay scheduler
- **Class Weighting**: Handles imbalanced datasets
- **Comprehensive Visualization**: Confusion matrices, event statistics, training progress

## 🔧 Configuration

### Hyperparameters
Key tunable parameters in driver scripts:
- `latentDim`: Feature dimension (64 for binary, 32 for arrhythmia)
- `hiddenChannels`: CNN filters (128)
- `dropoutRate`: Regularization (0.4 for binary, 0.25 for arrhythmia)
- `batchSize`: Training mini-batch size
- `learningRate`: Initial learning rate with decay scheduling

### Data Preprocessing
- Z-score normalization per feature channel
- Automatic handling of zero-variance features
- Sequence formatting for CNN input

## 📁 Data Format

### Input Data
- `X`: [N × SequenceLength × Features] ECG signal data
- `Y`: [N × 1] Labels (char array or categorical)

### Supported Beat Types
The system handles MIT-BIH arrhythmia database labels including:
- `N`: Normal beat
- `V`: Premature ventricular contraction
- `S`: Supraventricular premature beat
- `F`: Fusion of ventricular and normal beat
- `Q`: Unclassifiable beat
- And other AAMI-standard beat types

## 🛠️ Customization

### Adding New Models
1. Create new architecture in `build*Classifier.m`
2. Update training wrapper in `train*Classifier.m`
3. Modify driver script for new configuration

### Custom Evaluation
- Extend `evaluateConfusionMetrics.m` for additional metrics
- Modify `plotEventStats.m` for custom visualizations
- Add new analysis functions following the modular pattern

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{ecg_arrhythmia_classification,
  title = {ECG Arrhythmia Classification Framework},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-repo}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the function documentation in each `.m` file
2. Review the example driver scripts
3. Open an issue on GitHub with:
   - MATLAB version
   - Error messages
   - Dataset information
   - Reproduction steps

## 🔄 Version History

- **v1.0** (Current): Initial release with hierarchical classification
- **Planned**: Real-time inference, model compression, additional architectures

---

*This project is for research purposes. Always consult healthcare professionals for medical diagnosis.*
