

# Installation and Usage Guide for RF Model Evaluation Toolkit

This guide provides instructions on how to set up and run the RF Model Evaluation Toolkit, a Dash-based web application for visualizing and analyzing RF data.

## Installation

To run the RF Model Evaluation Toolkit, you need Python installed on your system along with several libraries. Follow these steps to install the necessary dependencies:

1. **Install Python**: Ensure Python is installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

2. **Set up a Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
## Install Required Libraries

Use pip to install the required Python libraries:

```bash
pip install dash pandas numpy plotly scipy
```

## Running the Application

After installing the dependencies, you can run the application:

1. **Start the Application**:
   Navigate to the directory containing your Dash app script (`cam_visualization_df.ipynb`) and run it

## Access the Application
Open a web browser and go to http://127.0.0.1:8050/. This is the default address for Dash apps.


## Application Features

The RF Model Evaluation Toolkit includes several interactive features for analyzing RF data:

- **Main Spectrogram Display**: Shows the RF data in a spectrogram format. You can zoom in on specific areas to view details.

- **Heatmap Overlay**: A checkbox allows you to overlay frequency CAM data on the spectrogram for additional analysis.

- **Reset App Button**: Resets the view to the original state.

- **Additional Plots**:
  - **Model Accuracy**: Displays the model's accuracy across the selected data range.
  - **Frequency Magnitude**: Shows the Power Spectral Density (PSD) of the selected IQ data range.
  - **Time Domain**: Visualizes the real part of the IQ data over the selected range.
  - **Constellation Plot**: Provides a scatter plot of the In-phase vs Quadrature components of the IQ data.

- **Interactive Zoom and Pan**: All plots support interactive zooming and panning for detailed analysis.
