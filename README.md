# SpecSwin3D

SpecSwin3D is a deep learning project for hyperspectral image band reconstruction, band importance analysis, and visualization. It provides tools for preprocessing, training, evaluation, and visualization of models, with a focus on spectral band selection and reconstruction using Swin-UNETR architectures.

## Project Structure

```
SpecSwin3D/
├── data_preprocessing/
│   ├── input-label.py
│   ├── restack_bands_to_16.py
│   └── restack_bands_to_16_repeated.py
├── evaluation/
│   ├── band_metrics_summary.py
│   ├── comprehensive_band_model_evaluation.py
│   └── evaluate_models.py
├── strategies/
│   ├── correlation_strategy.txt
│   ├── mutual_info_strategy.txt
│   ├── spectral_physics_strategy.txt
│   └── variance_strategy.txt
├── train/
│   ├── calculate_band_importance.py
│   ├── denormalize_utils.py
│   ├── fine_tune_all_strategies_219_bands.py
│   └── train_SpecSwin3D_16.py
├── visualize/
│   ├── analyze_checkpoints_structure.py
│   └── visualize_model_output.py
```

## Main Components

- **data_preprocessing/**: Scripts for preparing and stacking hyperspectral data and labels.
- **train/**: Training scripts, band importance calculation, and utility functions for normalization/denormalization.
- **evaluation/**: Scripts for evaluating model performance and summarizing band metrics.
- **visualize/**: Tools for visualizing model outputs and checkpoint structures.
- **strategies/**: Predefined band selection strategies based on different importance metrics.

## Key Scripts

- `train_SpecSwin3D_16.py`: Main training script for the Swin-UNETR model with 16 input bands.
- `calculate_band_importance.py`: Analyze and rank the importance of spectral bands using various methods (variance, correlation, mutual information, spectral physics).
- `denormalize_utils.py`: Utilities for denormalizing predictions and batches.
- `evaluate_models.py`: Evaluate trained models on test data.
- `visualize_model_output.py`: Visualize model predictions and outputs.

## Usage

1. **Data Preprocessing**
   - Use scripts in `data_preprocessing/` to prepare your input and label data.

2. **Band Importance Analysis**
   - Run `calculate_band_importance.py` to generate band importance rankings and strategies.

3. **Training**
   - Train models using `train_SpecSwin3D_16.py`.
   - You can specify a custom band selection strategy using the `--strategy custom --custom_txt <strategy_file>` arguments.

4. **Evaluation**
   - Use scripts in `evaluation/` to assess model performance and summarize results.

5. **Visualization**
   - Visualize outputs and checkpoints using scripts in `visualize/`.

## Example Training Command

```
python train_SpecSwin3D_16.py --strategy custom --custom_txt strategies/variance_strategy.txt --batch_size 12
```

## Data

The raw hyperspectral dataset used in this project is from the AVIRIS mission ([https://aviris.jpl.nasa.gov/](https://aviris.jpl.nasa.gov/)).

**Note:** The processed dataset used in this project is not publicly available. For access to the dataset, please contact the author:

- Tang Sui (tsui5@wisc.edu)

## Requirements
- Python 3.7+
- PyTorch
- numpy, matplotlib, scikit-learn, tqdm, seaborn, etc.

Install dependencies with:
```
pip install -r requirements.txt
```
or use the provided `environments.txt` for environment setup.

## License

This project is licensed for academic and research use only. For commercial or other uses, please contact the author.

## Citation
If you use SpecSwin3D in your research, please cite the relevant papers and this repository.

---

For more details, see the code and comments in each script.
