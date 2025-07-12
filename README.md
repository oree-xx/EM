# West Africa Debt Analysis

This project provides a comprehensive analysis of debt and economic indicators for West African countries using World Bank data. It includes automated data fetching, risk classification, sustainability metrics, and interactive visualizations.

## Features

- Fetches debt and economic data for 16 West African countries from the World Bank.
- Calculates debt-to-GDP, debt per capita, and sustainability metrics.
- Classifies countries into debt risk levels (Critical, High, Medium, Low).
- Generates interactive dashboards and visualizations using Plotly.
- Saves processed data and visualizations as HTML and CSV files.

## Requirements

- Python 3.7+
- [wbgapi](https://pypi.org/project/wbgapi/)
- pandas
- numpy
- plotly

Install dependencies with:
```sh
pip install wbgapi pandas numpy plotly
```

## Usage

Run the main analysis script:
```sh
python Africa.py
```

This will:
- Fetch and process the data
- Display interactive visualizations
- Save HTML files for each visualization
- Save the processed data as `comprehensive_debt_analysis.csv`

## Files

- `Africa.py`: Main analysis and visualization script.
- `Interactive_data.py`: Example script for interactive data exploration.
- `comprehensive_debt_analysis.csv`: Output data file (generated after running).
- `*.html`: Output visualization files.

## Customization

You can modify the list of countries or indicators in `Africa.py` by editing the `self.countries` and `self.indicators` dictionaries in the `WestAfricaDebtAnalyzer` class.

## License

This project is for educational and
