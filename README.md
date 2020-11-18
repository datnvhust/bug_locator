# How to run:
1. Clone this repository:
    ```
    git clone https://github.com/datnvhust/bug_locator.git
    ```
    Download the datasets file from [here](http://www.mediafire.com/file/5x0vjnno666ynst/data.zip/file), and unzip it in the root directory of the cloned repository.
    
2. Check the path of datasets in the `datasets.py` module and change the value of the `DATASET` variable to choose different datasets (values can be `aspectj`, `swt`, and `zxing`) (suggestion: `zxing`).
    Run the main module:
    ```
    python preprocessing.py
    python label.py
    python rvsm_similarity.py
    python ranking.py
    python ranking_label_1.py
    ```
    All the modules are also independently runnable if it was needed to run them one by one.
