# Data Setup Instructions

## Required Data Files

Place these course data files in the project root directory:

- `train.csv` (259MB) - Training dataset with customer churn data
- `test.csv` (45MB) - Test dataset for final predictions
- `VariableDefinitions.csv` - Feature descriptions in French/English
- `SampleSubmission.csv` - Submission format template

## File Sources

These files should be obtained from your AI7101 course materials or LMS.

## Usage

The `real_data_churn_analysis.ipynb` notebook will automatically load these files when they are placed in the project root directory.

## Note

Large CSV files are not stored in git due to GitHub file size limits. The notebook includes data validation to ensure files are present before processing.