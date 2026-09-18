# A practical guide to dimensionality reduction

This course aims at showcasing how to perform dimensionality reduction in practice. **Please go to the notebooks folder and open the notebook 1_feature_transformation.ipynb**. Your mentor will give you further instructions.

**To run the notebook, it is recommended to use Google Colab (Google account required). For that, just follow the link within the notebook and ignore the rest of this README file.**

Alternatively, you can set up an environment on your own machine as follows. Due to time restrictions, mentors cannot provide help with installation issues during the course.

The course requires **Python 3.12**.

Setup steps (**not needed for Colab users**):

- Clone or download the repo
- Run the following commands in a terminal

With conda (e.g. from miniforge):
```
conda create -n dimred python=3.12
conda activate dimred
pip install -r requirements.txt
```

Or with venv, if you do not have conda:
```
python3.12 -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
- Check that the installation worked by opening verify_install.ipynb and running
  both cells, using the same kernel you will use for the course notebooks. It
  installs nothing and needs no internet. Every problem it finds prints the
  command that fixes it.
- Open the notebooks from the notebooks folder in numbered order, starting with
  1_feature_transformation.ipynb, and select the environment you created above as
  the kernel (dimred if you used conda, .venv if you used venv)

## License

This repository contains both source code and teaching materials, which are licensed separately:

- **Code** (notebooks, scripts, and other software) is licensed under the MIT License. See the `LICENSE` file.
- **Teaching materials** (slides, figures, and written explanations) are licensed under the Creative Commons Attribution 4.0 License. See the `LICENSE-CONTENT` file.

[https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)

