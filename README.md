

# A Fitting parameter assigner for Fityk

## Overview
The `assigner.py` script is designed to manage and assign peaks fitted by Fityk. The script itself doesn't contain the ain logic, which for the most part is within the `only_sess.py`. In that module live all needed functions to assign the peaks, read the `.fit` file and write it all to a csv.

##  Use
The intended use is through streamlit. The app is up directly on my streamlit account or you can run it locally. Make sure to install all necessary packages (for example using conda):
```bash
git clone https://github.com/albertopeinador/GIWAXS_Assinger.git
cd GIWAXS_Assigner
conda install --file requirements.txt
```
Then you can run the program with:
```bash
streamlit run asigner.py
```

The program itself should be easy to follow from there.

You will be prompted to upload your Fityk session file (`.fit` file). Next some boxes asking you for the peak names (see `list_of_peaks`) and the functions you used to fit your peaks (see `list_of_functions`). Currently only PseudoVoigt and Gaussian are accounted for in the csv output, so keep this in mind. Then, some graphs will appear.
Below, on the left you will see the first dataset it can easily assign, this will help you see if your `list_of_peaks` is what you expected. A bigger graph on the right will help you see all assignments so you can confirm the program worked correctly. You can chose what dataset to plot on this right hand side plot using the selection box above it.

There are some filters for the graphs in case you don't want to plot the data, the fitted model or the labels. Beware, the shown model only expects either `Linear`, `ExpDecay`, `PseudoVoigt` or `Gaussian` functions, more can be added in the future. The program will still work regardless of what baseline you use, but do keep in mind the model displayed will not contain anything that is not one of those functions. Therefore, the displayed model might look different from what it did in Fityk.
## Parameters

### Input Parameters
- Path to the FIT file containing model parameters.
- **`list_of_peaks`**: Names of the expected peaks in as many of the datasets as possible.
- **`assign_extra_peaks`**: Boolean flag to include additional, unassigned peaks in the output.
- **`list_of_functions`**: List of functions used to fit the peaks you want to be assigned.

### Output Parameters
- **`asigned_peaks`**: Dictionary mapping model names to their assigned peaks, each with q, height, and width parameters.
- **`mean_q`, `s_q`, `mean_h`, `s_h`, `mean_w`, `s_w`**: Statistical measures (means and standard deviations) for the assigned peaks.

## Inner Workings
The program first finds the datasets which contain the same number of peaks (only counting functions from `list_of_functions`) as the number of peaks given in `list_of_peaks` in order. From this first assignations, it extracts information on the peaks (mean position, height and width as well as the standard deviation of these parameters). Using this information it the fins optimal assignation for the rest of the datasets using the Hungarian Algorithm. Currently, the cost function it uses to find the optimal assignenment is given by $`\frac{<q>}{\sigma(q)}`$, but other functions could be implemented (incorporating peak height or witdth for example).