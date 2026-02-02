# Environment set-up
Step 1
# Creating the environment, with the right python version in a bash terminal
conda create -n name_env python=3.11
Step 2
# Installing the meridian library
pip install --upgrade "google-meridian[colab,and-cuda] @ git+https://github.com/google/meridian.git"
This will automatically install all required libraries
Step 3
If demanded, install ipykernel
Step 4
# Install last missing libraries

pip install plotly
# to be able to plot results the way we want them

# installing needed libraries to run seasonnality functions
pip install vacances_scolaires_france

pip install openpyxl

## MMM-utilisation guide/toolbox
Every single code is already on gitlab under this project - tmilcent / MMMiX · GitLab
This is a guide if you ever have to start all over again from the start
### Step 1 : Data Preparation
Encoding Market data
Dataframe’s columns
Required
KPI - nb of recruitements per week
Time - weekly basis, ideally over the last 2 years (minimum)
Spend (for all medias)
Nb of impressions (for most digital channels)
Reach (for TV or OOH, goes hand in hand with Frequency)
Frequency (for TV or OOH, goes hand in hand with Reach)
Optional
Geo - in our case, to be replaced by the acquisition channel
Revenue per KPI - We take the margin over 48 months. To adapt for each acquisition channel
Population - in our case, to be put to 1 (as we don’t have the exact spend allocation per acquisition channel
Control (to help the model learn on seasonnality)
Non-media treatment (such as the release of a Freebox or else)
Example
week
recrutements_hebdo
geo
Revenue per KPI
population
jour ferie
back2school
sortie_box
spend_media1
spend_media2
spend_media3
nb_impressions_media1
nb_impressions_media2
reach_media3
frequency_media3

As you can see, as we cannot ventilate the spendings/nb of impressions per media across the different channels, which make the hierarchical modelling option (the one with the geo columns) a bit less robust than expected.
Recommandation
Use as geo the following acquisition channels - each time, do it for mobile and landline :
Web
Boutiques
Appels Entrants
We didn’t take Appels Sortants into account (too hard to predict for the model)
Encoding Market Data
For each media, format needed is the following :
For normal media
date
nb_impressions_media1
spend_media1
For media with reach and frequency
date
reach_media2
frequency_media2
spend_media2

## To be mentioned to marketing : complicated excels are not the solutions, see above to facilitate our work
However, here is how it has been encoded on vscode in the MMMiX project, after putting the market data excel to the format mentioned above
For Google and Meta, they sent the data separately, and it’s been encoded as well to match the needs of the model, see below the functions used, that you can store in a utils.py python file :
Change the links, paths and table names in blue in the code with the ones corresponding to the paths of your files.