# ISyE 6402 Module 5

## Usage Guide

### Course Wiki
We host introductory materials as wiki pages. Please checkout all information provided [here](https://github.gatech.edu/ISyE6402TimeSeriesAnalysis/Module5/wiki).

### Announcement and Q&A
We organize software-related announcements and Q&A in the [Discussion](https://github.gatech.edu/ISyE6402TimeSeriesAnalysis/Module5/discussions) page. Please only post ***software-related*** questions here. All other questions should be directed to ***Piazza***.

- [`Announcement`](https://github.gatech.edu/ISyE6402TimeSeriesAnalysis/Module5/discussions/categories/announcement): we highlight updates such as notebook changes or package changes as announcements. 
- `Q&A`: we encourage software-related questions posted in GitHub
    - Please use the ***correct category*** when posting questions.


### Update Course Materials
Make sure to do the following to update materials 

```bash
# navigate to ISyE6402Main
cd <path>/ISyE6402Main

# update materials
git pull --recurse-submodules
git submodule update --remote
```

After the update, if `Module5/` is empty, please checkout the `main` branch
```bash
cd Module5/
git checkout main
```

### Setup Environment
#### Local Environment
Currently, the notebooks are only tested with Google Colab (latest version)

#### Cloud Environment
If you prefer running the notebooks in cloud (e.g., Google Colab), please upload the notebook into your cloud environment, and make sure the data files are also uploaded under the root directory.
