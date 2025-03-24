# 🤝 Contribution Guidelines

Welcome to mmScribe! 🎯

This repository is governed by the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## 📋 Where to Start

We welcome everyone who likes to contribute to mmScribe, especially in expanding our dataset for better handwriting  recognition across different scenarios and user groups.

You can contribute in multiple ways:
- 📊 Share your collected datasets
- 🐛 Report bugs and issues
- 💡 Suggest improvements
- 📝 Improve documentation
- 🔍 Help with data validation
- 🌍 Add support for new languages/gestures

## ⭐ Call for Dataset Contributions

We're actively seeking contributions to expand our handwriting recognition dataset. Your contributions help improve the system's accuracy and robustness across different:
- 👥 User demographics
- ✍️ Writing styles
- 🌏 Languages and scripts
- 📱 Device configurations

### 📊 Dataset Requirements

1. **Data Format**
   ```python
   # Data shape: T*163
   data[:,:128]    # Micro-Doppler time features
   data[:,128:160] # Range-time features
   data[:,160:162] # X-Z coordinate position (Leap Motion)
   data[:,163]     # Reserved
   ```

2. **Required Metadata**
   - 👤 Participant demographics (anonymized)
   - 📐 Hardware setup details
   - ⚙️ Collection parameters
   - 📝 Writing task descriptions

3. **Quality Standards**
   - ✅ Clear signal quality
   - ✅ Proper calibration
   - ✅ Complete metadata
   - ✅ Proper anonymization

## 🚀 How to Contribute Data

### 1. Data Collection
```bash
# Clone the repository
git clone https://github.com/yourusername/mmScribe.git

# Set up the data collection system
cd mmScribe/DataCaptureSystem
pip install -r requirements.txt

# Start collection
python start_collection.py
```

### 2. Data Validation
- Run our validation scripts
- Check signal quality
- Verify metadata completeness
- Ensure proper formatting

### 3. Submit Your Contribution

1. **Fork and Clone**
   ```bash
   git clone git@github.com:<your Github name>/mmScribe.git
   cd mmScribe
   git remote add upstream https://github.com/original/mmScribe.git
   ```

2. **Create Dataset Branch**
   ```bash
   git checkout -b dataset/<your_institution_name>
   ```

3. **Add Your Data**
   - Place data in `dataset/contributions/<your_institution_name>/`
   - Include README with collection details
   - Add metadata files

4. **Submit Pull Request**
   - Target the `data` branch
   - Include detailed description
   - Fill out contribution checklist

## ✨ Contribution Checklist

```markdown
### Dataset Information
- [ ] Number of participants: ___
- [ ] Total samples: ___
- [ ] Languages/scripts: ___
- [ ] Hardware configuration documented
- [ ] Collection parameters provided

### Quality Assurance
- [ ] Validation scripts passed
- [ ] Signal quality verified
- [ ] Metadata complete
- [ ] Privacy requirements met

### Documentation
- [ ] Collection methodology described
- [ ] Hardware setup documented
- [ ] Special considerations noted
- [ ] License terms accepted
```

## 🎯 Data Usage and Attribution

- All contributed data will be released under our academic license
- Contributors will be properly credited in:
  - 📚 Research papers
  - 🌐 Project documentation
  - 🏷️ Model attributions

## 💫 Benefits of Contributing

1. 🏆 Recognition in the mmScribe community
2. 🔬 Early access to research findings
3. 👥 Collaboration opportunities
4. 📊 Access to expanded dataset
5. 🎓 Academic collaboration possibilities

## 🤔 Questions?

- 📧 Email: [research@mmscribe.org](mailto:research@mmscribe.org)
- 💬 Join our [Discord](https://discord.gg/mmscribe)
- 🌟 Create an [Issue](https://github.com/yourusername/mmScribe/issues)

## 📝 License

By contributing to mmScribe, you agree that your contributions will be licensed under its MIT License, except for hardware-related contributions which fall under our academic license. 