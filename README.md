# Combined Visual Encoding of Absolute(Part) and Fractional(Part-to-Whole) Values

<!-- ![Four visualization designs showing three data points with absolute values A=[77, 127, 42], fractional values F=[0.31, 0.65, 0.42], and respective whole valuesW=[252, 195, 100].](4_design_choices.PNG "Teaser Figure") -->

<img src="4_design_choices.PNG" alt="Teaser Figure" title="Four visualization designs showing three data points with absolute values A=[77, 127, 42], fractional values F=[0.31, 0.65, 0.42], and respective whole valuesW=[252, 195, 100]." style="width: 40%;">

This an initial design space exploration for visualizations that visualize both absolute and fractional values. It is an prototype implementation of an designs discussed in the following poster research paper:

*Exploring Designs for Combined Visual Encoding of Absolute and Fractional Values*. Madhav Poddar, and Fabian Beck. EuroVis 2024. DOI: 10.2312:evp.20241082

The visualizations were developed using the [Bokeh library](https://bokeh.pydata.org/en/latest/) in Python.

## Directly viewing the designs

Open "main.html" to see all the different visualizations.

## Generating the designs yourself

Prerequisites: conda needs to be installed.

To run the application simply clone the repository, go to the main project folder and start with the following commands:

```cmd
conda env create --name PandPtoWenv --file=p_and_p2w_env.yml
conda activate PandPtoWenv
```

This will create the conda environment and activate it.

Next, to run the code: 

```cmd
python main.py
```

If you want to visualize a different sample dataset or provide a different dataset, please make modifications in the file "main.py". 

## Project structure

| Source File Name         | Description                                                                                    |
|--------------------------|------------------------------------------------------------------------------------------------|
| main.py                  | The main file where you specify the dataset to visualize.                                      |
| generate_vis.py          | Defines the different visual encodings.                                                        |

