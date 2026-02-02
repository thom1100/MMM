import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import sys
import os
PATH = Path().resolve().parent

sys.path.append(str(PATH)) # pour ajouter le chemin au sys.path et pouvoir importer la fonction
from utils import add_exogenous_variables


def data_concu():
# Retrieving spending data of competitors
    df22 = pd.read_excel('/home/tmilcent/.xlsx', sheet_name = 'ALL MEDIA', skiprows = 8)
    df23 = pd.read_excel('/home/tmilcent/.xlsx', sheet_name = 'ALL MEDIA', skiprows = 8)
    df24 = pd.read_excel('/home/tmilcent/MMMiX/.xlsx', sheet_name = 'ALL MEDIA', skiprows = 8)
    df25 = pd.read_excel('/home/tmilcent/MMMiX/.xlsx', sheet_name = 'ALL MEDIA', skiprows = 8)
    # Taking care of each year individually
    df_22 = excel_work(df22)
    df_23 = excel_work(df23)
    df_24 = excel_work(df24)
    df_25 = excel_work(df25)
# Last step, concatenating vertically, as we made sure that each column existed in each excel
    return pd.concat([df_22, df_23, df_24, df_25], axis=0)

def excel_work(df):
    '''
    Goal : Treating excels with competitors data and get a format easily readable for the model
    -----------
    Arguments :
    - Dataframe filled via the excel with the competitors data for a year
    -----------
    Returns
    - dataframe with a spending column for each operator as well as a total column
    '''
    # Automatically treating the excel as sent by marketing - to be changed if format changes
    df.drop(columns = ['Unnamed: 0', 'Unnamed: 2'], inplace = True)
    # Dropping unused columns
    df.dropna(inplace=True)
    df.rename(columns = {'Unnamed: 1':'date'}, inplace = True)
    # Let's rename 'date' the right column
    df.set_index('date', inplace = True)
    df = df.T
    df = df[df.index != 'TOTAL']
    df.index = pd.to_datetime(df.index, dayfirst=True)

    for col in ['TOTAL', 'BOUYGUES TEL.B&YOU', 'BOUYGUES TEL.BBOX', 'BOUYGUES TELECOM', 'FREE', 'FREE FREEBOX', 'ORANGE', 'ORANGE LIVEBOX','ORANGE SOSH', 'SFR RED','SFR'] :
        if col not in df.columns:
            df[col] = 0

    return df[['TOTAL', 'BOUYGUES TEL.B&YOU', 'BOUYGUES TEL.BBOX', 'BOUYGUES TELECOM', 'FREE', 'FREE FREEBOX', 'ORANGE', 'ORANGE LIVEBOX','ORANGE SOSH', 'SFR RED','SFR']]



def encode_meta_fixe(df):
    '''
    Goal : Treating Meta landline data automatically accessed from their platform of control
    Treating individually landline vs mobile if wanting to treat the 2 products separately
    -----------
    Arguments :
    - Dataframe filled via the excel with all the landline data of Meta for a year
    -----------
    Returns
    - dataframe with a spending and nb of impression columns for each type of meta media channel
    '''

# Distinguish the types of media we want to treat
    df = df[~df['Objective'].isin(['Store visits', 'Link clicks'])]
    # We get rid of Store visits and link clicks as they are not really relevant in our case
    df['Day'] = pd.to_datetime(df['Day'])
    df['date'] = df['Day'] - pd.to_timedelta(df['Day'].dt.weekday, unit='D')
    df_agg = df.groupby(['date', 'Objective']).agg(nb_impressions_fixe=('Impressions', 'sum'), spend_fixe=('Amount spent', 'sum')).reset_index()

    df_pivot = df_agg.pivot(index='date', columns='Objective', values=['nb_impressions_fixe', 'spend_fixe'])
    df_pivot.columns = [f"{col[0]}_{col[1].lower()}" for col in df_pivot.columns]

    df_pivot.reset_index(inplace=True)

    return df_pivot


def encode_meta_mobile(df):
    '''
    Goal : Treating Meta mobile data automatically accessed from their platform of control
    -----------
    Arguments :
    - Dataframe filled via the excel with all the landline data of Meta for a year
    -----------
    Returns
    - dataframe with a spending and nb of impression columns for each type of meta media channel
    '''

    df = df[~df['Objective'].isin(['Post engagement', 'Link clicks'])]
    df['Day'] = pd.to_datetime(df['Day'])
    df['date'] = df['Day'] - pd.to_timedelta(df['Day'].dt.weekday, unit='D')
    df_agg = df.groupby(['date', 'Objective']).agg(nb_impressions_mobile=('Impressions', 'sum'), spend_mobile=('Amount spent', 'sum')).reset_index()

    df_pivot = df_agg.pivot(index='date', columns='Objective', values=['nb_impressions_mobile', 'spend_mobile'])
    df_pivot.columns = [f"{col[0]}_{col[1].lower()}" for col in df_pivot.columns]

    df_pivot.reset_index(inplace=True)

    return df_pivot


df_fixe_2023 = pd.read_csv('/home/tmilcent/.csv', sep=',')
df_fixe_2024 = pd.read_csv('/home/tmilcent/.csv', sep=',')
df_fixe_2025 = pd.read_csv('/home/tmilcent/.csv', sep=',')

df_mobile_2023 = pd.read_csv('/home/tmilcent/.csv', sep=',')
df_mobile_2024 = pd.read_csv('/home/tmilcent/MMMiX/.csv', sep=',')
df_mobile_2025 = pd.read_csv('/home/tmilcent/MMMiX/.csv', sep=',')

def encode_all_meta():

    fixe_2023 = encode_meta_fixe(df_fixe_2023)
    fixe_2024 = encode_meta_fixe(df_fixe_2024)
    fixe_2025 = encode_meta_fixe(df_fixe_2025)

    mobile_2023 = encode_meta_mobile(df_mobile_2023)
    mobile_2024 = encode_meta_mobile(df_mobile_2024)
    mobile_2025 = encode_meta_mobile(df_mobile_2025)

    df_meta_2023 = pd.merge(fixe_2023, mobile_2023, how='outer', on='date')
    df_meta_2024 = pd.merge(fixe_2024, mobile_2024, how='outer', on='date')
    df_meta_2025 = pd.merge(fixe_2025, mobile_2025, how='outer', on='date')

# Concatenating all data for each year (both fixed and mobile)
    df_final = pd.concat([df_meta_2023, df_meta_2024, df_meta_2025], axis=0)
    df_final.fillna(0, inplace=True)

# Adding columns to get total spend and total nb of impressions for landline and mobile
    df_final['nb_impressions_outcomesalesconversions'] = df_final['nb_impressions_fixe_outcome sales'] + df_final['nb_impressions_mobile_outcome sales'] + df_final['nb_impressions_fixe_conversions'] + df_final['nb_impressions_mobile_conversions']
    df_final['nb_impressions_outcomeawareness'] = df_final['nb_impressions_fixe_outcome awareness'] + df_final['nb_impressions_mobile_outcome awareness']
    df_final['nb_impressions_outcomeengagement'] = df_final['nb_impressions_fixe_outcome engagement'] + df_final['nb_impressions_mobile_outcome engagement']

    df_final['spend_outcomesalesconversions'] = df_final['spend_fixe_outcome sales'] + df_final['spend_mobile_outcome sales'] + df_final['spend_fixe_conversions'] + df_final['spend_mobile_conversions']
    df_final['spend_outcomeawareness'] = df_final['spend_fixe_outcome awareness'] + df_final['spend_mobile_outcome awareness']
    df_final['spend_outcomeengagement'] = df_final['spend_fixe_outcome engagement'] + df_final['spend_mobile_outcome engagement']


    return df_final.groupby('date', as_index=False).sum()

df_snap = pd.read_excel('/home/tmilcent/MMMiX/Data/Paid-digital-Snapchat.xlsx')

def encode_snap():
    '''
    Goal : Treating Snap data
    -----------
    Arguments :
    - Dataframe filled via the excel with all the data of Snapchat for over a year
    -----------
    Returns
    - Dataframe with a spending and nb of impression columns for each type of snap media channel
    '''
    df_snap['date'] = df_snap['Début']
    df_snap['date'] = pd.to_datetime(df_snap['date'], format = 'mixed')
    df_snap['date'] = df_snap['date'] - pd.to_timedelta(df_snap['date'].dt.weekday, unit='D')

    df_agg = df_snap.groupby(['date', 'Objectif de la campagne']).agg(nb_impressions_snap = ('Impressions payées', 'sum'), spend_snap = ('Montant dépensé', 'sum')).reset_index()
    df_pivot = df_agg.pivot(index='date', columns='Objectif de la campagne', values=['nb_impressions_snap', 'spend_snap'])

    df_pivot.columns = [f"{col[0]}_{col[1].lower()}" for col in df_pivot.columns]
    df_pivot.fillna(0, inplace=True)

    df_pivot.reset_index(inplace = True)

    return df_pivot

df1 = pd.read_csv("/home/tmilcent/.csv")
df2 = pd.read_csv("/home/tmilcent/.csv")
df3 = pd.read_csv("/home/tmilcent/.csv")
df4 = pd.read_csv("/home/tmilcent/.csv")
df5 = pd.read_csv("/home/tmilcent/.csv")
df6 = pd.read_csv("/home/tmilcent/.csv")
df7 = pd.read_csv("/home/tmilcent/.csv")
df8 = pd.read_csv("/home/tmilcent/.csv")
df9 = pd.read_csv("/home/tmilcent/.csv")
df10 = pd.read_csv("/home/tmilcent/.csv")

def encode_all_google():

    '''
    Goal : Treating Google data
    -----------
    Arguments :
    - Dataframe filled via the excel with the data of Google for a year or less
    -----------
    Returns
    - Dataframe with a spending and nb of impression columns for each type of snap media channel
    '''

    df_all = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10], axis=0)

    df_all['ReportDate'] = pd.to_datetime(df_all['ReportDate'])
    df_all['date'] = df_all['ReportDate'] - pd.to_timedelta(df_all['ReportDate'].dt.weekday, unit='D')

    df_agg = df_all.groupby(['date', 'CampaignType']).agg(nb_impressions=('Impressions', 'sum'), spend=('Cost', 'sum')).reset_index()

    df_pivot = df_agg.pivot(index='date', columns='CampaignType', values=['nb_impressions', 'spend'])
    df_pivot.columns = [f"{col[0]}_{col[1].lower()}" for col in df_pivot.columns]
    df_pivot.reset_index(inplace = True)

    return df_pivot.groupby('date', as_index=False).sum()
# Let's do the same for all the other data, that we compiled ourselved on an excel

file_path = Path(PATH, "MMMiX", "Data", "Free_Calendrier_CAMPAGNE TV_OFF_2025.xlsx")
df_tv = pd.read_excel(file_path, sheet_name='Calendrier_campagnes_reformat', skiprows=1)
df_tv = df_tv.dropna(subset=['date']) # on s'assure de ne pas avoir de na dans la colonne date
df_tv = df_tv.set_index('date') #pour permettre le join, on met la colonne date en index
df_tv = df_tv.fillna(0)
df_tv = df_tv.drop(columns = [
'Produit',
'Focus_produit_tv_fixe',
'Focus_produit_tv_mobile',
'Mix_format_tv_fixe',
'Mix_format_tv_mobile'
])
df_tv.index = pd.to_datetime(df_tv.index)

df_google = encode_all_google()
df_google = df_google.set_index('date')
df_google = df_google.fillna(0)

df_google.index = pd.to_datetime(df_google.index)
df_google = df_google[(df_google.index >= min(df_tv.index)) & (df_google.index <= max(df_tv.index))]

df_meta = encode_all_meta()
df_meta = df_meta.set_index('date')

#  We fix indexes, useful to join all tables
df_meta.index = pd.to_datetime(df_meta.index)
df_meta = df_meta[(df_meta.index >= min(df_tv.index)) & (df_meta.index <= max(df_tv.index))]

df_snap = encode_snap()
df_snap = df_snap.set_index('date')
df_snap.index = pd.to_datetime(df_snap.index)
df_snap = df_snap[(df_snap.index >= min(df_tv.index)) & (df_snap.index <= max(df_tv.index))]

df_concu = data_concu()
df_concu_weekly = df_concu.resample('W-MON').sum()

# Shortening some dataframes to avoid compiling all
df_concu_weekly = df_concu_weekly[(df_concu_weekly.index >= min(df_tv.index)) & (df_concu_weekly.index <= max(df_tv.index))]

df_tv_concu = df_concu_weekly.join(df_tv, how='outer')
df_tv_concu_google = df_tv_concu.join(df_google, how='outer')
df_tv_concu_google_snap = df_tv_concu_google.join(df_snap, how='outer')
df_media_channels = df_tv_concu_google_snap.join(df_meta, how='outer')
df_media_channels.reset_index(inplace = True)
df_media_channels.rename(columns={"index": "date"}, inplace=True)



def seasonnality(df_recrutements_hebdo):
# on utilise la fonction d'Alexis F définie dans utils pour ajouter les variables saisonnières

    '''
    Goal : Get the seasonnality data, for the weeks in which we are interested
    -----------
    Arguments :
    - Dataframe with a column 'date' and a column 'recruitments'
    -----------
    Returns
    - Dataframe with all seasonality informations, per week, such as holidays, back2school period, etc
    '''

    df = add_exogenous_variables(df_recrutements_hebdo)
    return df.drop(
        columns=[
        "rattrapage_churn",
    ]
)

def get_bdd(channel_considered=str, landline_or_mobile=str):

    '''
    Goal : Get all needed data to have a basic dataset useful in Meridian
    -----------
    Arguments :
    - acquisition channel considered : String, choose between [Web, Appel Entrant, Boutiques]
    - landline or mobile : String, choose between [landline, mobile]-----------
    Returns
    - Dataframe with columns date, recruitements, geo, and all seasonnality columns, depending on the channel and product considered
    '''

    #Final push to create the dataframe with all media impressions,
    df_media = pd.read_csv("df_media.csv")
    df_recrutements_hebdo, model_name = pd.read_csv("df_recrutements_hebdo.csv")
    df_recrutements_hebdo = df_recrutements_hebdo.set_index('date')
    df_media = df_media.set_index('date')
    df_recrutements_hebdo.index = pd.to_datetime(df_recrutements_hebdo.index)
    df_media.index = pd.to_datetime(df_media.index)
    df_recrutements_hebdo = df_recrutements_hebdo[(df_recrutements_hebdo.index >= min(df_media.index)) & (df_recrutements_hebdo.index <= pd.to_datetime("2025-05-19"))]
    df = seasonnality(df_recrutements_hebdo)

    if landline_or_mobile == 'mobile':
        if channel_considered == 'Appel Entrant':
            revenue_per_kpi = 462.0
        elif channel_considered == 'Boutiques':
            revenue_per_kpi = 470.0
        elif channel_considered == 'Web':
            revenue_per_kpi = 357.0
    elif landline_or_mobile == 'landline':
        if channel_considered == 'Appel Entrant':
            revenue_per_kpi = 354.0
        elif channel_considered == 'Boutiques':
            revenue_per_kpi = 288.0
        elif channel_considered == 'Web':
            revenue_per_kpi = 348.0
    df_final = df.join(df_media, how = 'outer')

    df_final = df_final.reset_index()
    # Last cosmetic change to avoid wrong data, for instance already planned campaigns
    df_final = df_final[df_final['date'] <= max(df_recrutements_hebdo.index)]

    # Adding a column to have the 'geography', in our case corresponding to the acquisition channel
    df_final['geo'] = channel_considered + landline_or_mobile
    # Adding the revenue_per_kpi column to assess the impact of each recruitment
    df_final['revenue_per_kpi'] = revenue_per_kpi

    return df_final.fillna(0), model_name

def get_all_bdd(channel_considered, landline_or_mobile):

    '''
    Goal : Get all needed data to have a basic dataset useful in Meridian, for all acquisition channel
    -----------
    Arguments :
    - acquisition channel considered : String, choose between [Web, Appel Entrant, Boutiques]
    - landline or mobile : String, choose between [landline, mobile]-----------
    Returns
    - Dataframe with columns date, recruitements, geo, population and all seasonnality columns, depending on the channel and product considered
    '''

    df_final = pd.DataFrame()
    for product in landline_or_mobile:
        for channel in channel_considered:

            df, model_name = get_bdd(channel, product)
            df_final = pd.concat([df_final, df], axis=0)
    df_final['population'] = 1

    return df_final
#Training the model
# https://developers.google.com/meridian/docs/basics/input-data?hl=fr
# https://developers.google.com/meridian/docs/basics/model-spec?hl=fr
# Initializing Meridian Model

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import arviz as az

import IPython

from meridian import constants
from meridian.data import load
from meridian.data import test_utils
from meridian.model import model
from meridian.model import spec
from meridian.model import prior_distribution
from meridian.analysis import optimizer
from meridian.analysis import analyzer
from meridian.analysis import visualizer
from meridian.analysis import summarizer
from meridian.analysis import formatter

# If utils exists with the needed encoded functions
from utils import *
# check if GPU is available
from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\\n'.format(ram_gb))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))

import matplotlib.pyplot as plt
import plotly.express as px
import os

# Loading the data

df = get_all_bdd(safe_get(config, "channel_considered", default = "Boutiques"),safe_get(config, "Landline_Mobile", default = "landline"))
df.head()

# Indicating to the model what columns it shall use and loading them
coord_to_columns = load.CoordToColumns(
    time='date',
    geo='geo',
    population='population',
    kpi='recrutements_hebdo',
    revenu_per_kpi= 'revenu_per_kpi',
    controls= ['jour_ferie_wo_pentecote',
                'jour_ferie'
                'lundi_pentecote',
                'jour_ferie_dimanche',
                "start_of_month",
                "september",
                "may",
                'pont',
                'ferie_prolonge',
                'school_holiday',
                'end_of_year',
                'pre_end_of_year',
                'rentree_scolaire',
                'xmas_new_year',
                'back_to_school',
                'adjacent_back_to_school',
                'BOUYGUES_TEL_B_YOU',
                'BOUYGUES_TEL_BBOX',
                'BOUYGUES_TELECOM',
                'ORANGE',
                'ORANGE_LIVEBOX',
                'ORANGE_SOSH',
                'SFR_RED',
                'SFR',
                "sortie_ultra"
                ],
    non_media_treatments= [
                            ],
    media= [
            'nb_impressions_campaign_type_demand_gen',
            'nb_impressions_campaign_type_search',
            'nb_impressions_campaign_type_video',
            'nb_impressions_campaign_type_search',
            'nb_impressions_metadisplay',
            'nb_impressions_metavideo',
        ],
    media_spend=[
                'spend_campaign_type_demand_gen',
                'spend_campaign_type_search',
                'spend_campaign_type_video',
                'spend_campaign_type_pmax',
                'spend_metadisplay',
                'spend_metavideo',
                ],
    reach=['reach_tvfixe',
                'reach_tvmobile',
                'reach_ooh'],
    frequency=['frequency_tvfixe',
                    'frequency_tvmobile',
                    'frequency_ooh']
    rf_spend=['spend_tv_fixe',
                'spend_tv_mobile',
                'spend_ooh']
)


# Rajouter media_spend_to_channel, reach_to_channel etc
media_to_channel={
    'nb_impressions_metadisplay':'meta display',
    'nb_impressions_metavideo':'meta video',
    'nb_impressions_campaign_type_demand_gen' : 'google demande gen',
    'nb_impressions_campaign_type_display': 'google display',
    'nb_impressions_campaign_type_pmax':'google pmax',
    'nb_impressions_campaign_type_search':'google search', #on le met de cote pour le moment car va avoir tendance à cannibaliser les ventes
    'nb_impressions_campaign_type_video':'google video',
}
media_spend_to_channel={
    'spend_metadisplay':'meta display',
    'spend_metavideo':'meta video',
    'spend_metamixed':'meta mixed',
    'spend_campaign_type_demand_gen' : 'google demande gen',
    'spend_campaign_type_display': 'google display',
    'spend_campaign_type_pmax':'google pmax',
    'spend_campaign_type_search':'google search',
    'spend_campaign_type_shopping':'google shopping',
    'spend_campaign_type_video':'google video',
}
reach_to_channel={
            'reach_tv_fixe':'tv_fixe',
            'reach_tv_mobile':'tv_mobile',
            'reach_ooh':'ooh'
            }
frequency_to_channel={
            'frequency_tv_fixe':'tv_fixe',
            'frequency_tv_mobile':'tv_mobile',
            'frequency_ooh':'ooh'
    }
rf_spend_to_channel={
            'spend_tv_fixe':'tv_fixe',
    'spend_tv_mobile':'tv_mobile',
            'spend_ooh':'ooh'
    }

loader = load.DataFrameDataLoader(
    df=df, # Attention, le csv doit être au même endroit que le script python
    kpi_type='non_revenue',
    coord_to_columns=coord_to_columns,
    media_to_channel=media_to_channel,
    media_spend_to_channel=media_spend_to_channel,
    reach_to_channel=reach_to_channel,
    frequency_to_channel= frequency_to_channel,
    rf_spend_to_channel= rf_spend_to_channel,
)
# Implementing the priors
# 🚧 You need the same numbers of priors as there is of media channels !
# Latence et saturation des canaux média  |  Meridian  |  Google for Developers
alpha_prior = tfp.distributions.Uniform(
low= [0.0001, 0.0001, 0.0001, 0.0001, 0.0001,0.0001],
high=[0.4, 0.3, 0.3, 0.6, 0.3, 0.4],
name=constants.ALPHA_M
)

# Elasticité cumulée, ie expected per euro spent during the period
ec_priors = tfp.distributions.TruncatedNormal(
    loc= [0.1, 0.2, 0.2,0.3, 0.4, 0.2],
    scale=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    low = 0.001,
    high= 10.0,
    name=constants.EC_M
)

# Intensity of media effect
# Google → strong direct effet, TV → harder to measure its effect through time
beta_prior = tfp.distributions.LogNormal(
    loc= [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
scale=[0.17, 0.2, 0.2, 0.13, 0.17, 0.17]
, # pour jouer sur l'incertitude
    name=constants.BETA_M
)

alpha_prior_rf = tfp.distributions.Uniform(
low= [0.5, 0.5, 0.001],
high=[1.0, 1.0, 0.7],
name=constants.ALPHA_M
)

ec_priors_rf = tfp.distributions.TruncatedNormal(
    loc= [0.4, 0.4, 0.3],
    scale=[2.0, 2.0, 2.0],
    low = 0.001,
    high= 10.0,
    name=constants.EC_M
)

beta_prior_rf = tfp.distributions.LogNormal(
    loc= [0.0, 0.0, 0.0],
scale=[0.095, 0.09, 0.11]
, # pour jouer sur l'incertitude
    name=constants.BETA_M
)

knot_prior = tfp.distributions.Normal(
    loc = 0.0,
    scale = 0.2,
    name=constants.KNOT_VALUES)
# How much the baseline of each geo is allowed to changed compared to the one with the most population
baseline_prior = tfp_distributions.Normal(
		loc = 0.0,
		scale = 0.01,
		name = constants.BASELINE_GEO
)
prior = prior_distribution.PriorDistribution(
    knot_values=knot_prior,
    tau_g_excl_baseline=baseline_prior,
    alpha_m=alpha_prior,
    alpha_rf = alpha_prior_rf,
    beta_m=beta_prior,
    beta_rf = beta_prior_rf,
    ec_m=ec_priors,
    ec_rf = ec_priors_rf
)
# Running and training the model
# Important notice : Need to restart the Kernel if running it on a notebook!!! If not it will cause the model to fail 😃
#Running the model with the chosen priors

model_spec = spec.ModelSpec(knots = 20,
														prior=prior,
                            media_prior_type='coefficient',
                            rf_prior_type = 'coefficient',
                            baseline_geo = None,
                            max_lag= 18)

mmm =model.Meridian(
    input_data=data,
    model_spec=model_spec
)

mmm.sample_prior(500)
mmm.sample_posterior(n_chains=10, n_adapt=2000, n_burnin=500, n_keep=1000, seed=1)
# When using MLFlow, you need to
# create a new class to use the Meridian model :

class MeridianWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, input_data, model_spec):
        self.model_class = model  # model.Meridian (la classe, pas une string)
        self.input_data = input_data
        self.model_spec = model_spec
        self.meridian_model = self.model_class(input_data=self.input_data, model_spec=self.model_spec)

    def predict(self, context, model_input):
        return self.meridian_model.predict(model_input)

    def __getattr__(self, name):
        # Délègue à meridian_model si défini
        meridian_model = self.__dict__.get("meridian_model", None)
        if meridian_model is not None:
            return getattr(meridian_model, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
# Create a config.yaml file to help you store all priors

# La combinaison des deux permet de choisir le canal d'acquisition auquel on s'intéresse, ainsi que le type de produits (landline ou mobile).
# All si on veut tous les recrutements d'un type de produits
# channel_considered:
#   - "Appel Entrant"
#   - "Boutiques"
#   - "Web"
# Landline_Mobile:
#   - "landline"
#   - "mobile"

# Variables de contrôle, qui permettent d'aider le modèle à prédire la saisonnalité ou baseline - ie à quel niveau de recrutements on serait sans les médias
# coord:
#   controls:
#     - "jour_ferie_wo_pentecote"
#     - "jour_ferie"
#     - "lundi_pentecote"
#     - "jour_ferie_dimanche"
#     - "start_of_month"
#     - "september"
#     - "may"
#     - "cyber_attack"
#     - "pont"
#     - "ferie_prolonge"
#     - "school_holiday"
#     - "end_of_year"
#     - "pre_end_of_year"
#     - "rentree_scolaire"
#     - "xmas_new_year"
#     - "back_to_school"
#     - "adjacent_back_to_school"
#     - "BOUYGUES_TEL_B_YOU"
#     - "BOUYGUES_TEL_BBOX"
#     - "BOUYGUES_TELECOM"
#     - "ORANGE"
#     - "ORANGE_LIVEBOX"
#     - "ORANGE_SOSH"
#     - "SFR_RED"
#     - "SFR"
    # - "sortie_ultra"

# Variables de contrôle qui peuvent à certains égards agir comme une publicité - typiquement les sorties de box pour le landline
#   non_media_treatments:
    # - "sortie_ultra"

# On indique au modèle les médias auxquels on s'intéresse, plus particulièrement les colonnes du dataframe qu'on va utiliser
# Attention à bien distinguer les medias pour lesquels on a des impressions, et ceux pour lesquels on a fréquence et répétitions, à indiquer à un autre endroit
# media:
#   impressions:
#     - 'nb_impressions_radioaudio'
#     - 'nb_impressions_campaign_type_demand_gen'
#     - 'nb_impressions_campaign_type_video'
#     - 'nb_impressions_campaign_type_pmax'
#     - 'nb_impressions_campaign_type_search'
#     - 'nb_impressions_snapawarenessengagement'
#     - 'nb_impressions_snapsales'
#     - 'nb_impressions_outcomesalesconversions'
#     - 'nb_impressions_outcomeawareness'
#     - 'nb_impressions_outcomeengagement'
#   spend:
#     - 'spend_radioaudio'
#     - 'spend_campaign_type_demand_gen'
#     - 'spend_campaign_type_video'
#     - 'spend_campaign_type_pmax'
#     - 'spend_campaign_type_search'
#     - 'spend_snapawarenessengagement'
#     - 'spend_snapsales'
#     - 'spend_outcomesalesconversions'
#     - 'spend_outcomeawareness'
#     - 'spend_outcomeengagement'
# # Indiquer ici fréquence et répétition
#   reach:
#     - 'reach_tv'
#     - 'reach_ooh'
#     - 'reach_radio'
#   frequency:
#     - 'frequency_tv'
#     - 'frequency_ooh'
#     - 'frequency_radio'
#   rf_spend:
#     - 'spend_tv'
#     - 'spend_ooh'
#     - 'spend_radio'

# # Enfin, il faut indiquer au modèle à quel média sont rattachés les impressions et spend
#   impressions_to_channel:
#     nb_impressions_radioaudio: 'radio audio'
#     nb_impressions_campaign_type_demand_gen: 'google demande gen'
#     nb_impressions_campaign_type_video: 'google video'
#     nb_impressions_campaign_type_pmax: 'pmax'
#     nb_impressions_campaign_type_search: 'search'
#     nb_impressions_snapawarenessengagement : 'snapchat awareness and engagement'
#     nb_impressions_snapsales : 'snapchat sales'
#     nb_impressions_outcomesalesconversions: 'meta sales'
#     nb_impressions_outcomeawareness: 'meta awareness'
#     nb_impressions_outcomeengagement: 'meta engagement'
#   spend_to_channel:
#     spend_radioaudio: 'radio audio'
#     spend_campaign_type_video: 'google video'
#     spend_campaign_type_pmax: 'pmax'
#     spend_campaign_type_search: 'search'
#     spend_campaign_type_demand_gen: 'google demande gen'
#     spend_snapawarenessengagement : 'snapchat awareness and engagement'
#     spend_snapsales : 'snapchat sales'
#     spend_outcomesalesconversions: 'meta sales'
#     spend_outcomeawareness: 'meta awareness'
#     spend_outcomeengagement: 'meta engagement'
#   reach_to_channel:
#     reach_tv: 'tv'
#     reach_ooh: 'ooh'
#     reach_radio: 'radio'
#   frequency_to_channel:
#     frequency_tv: 'tv'
#     frequency_ooh: 'ooh'
#     frequency_radio: 'radio'
#   rf_spend_to_channel:
#     spend_tv: 'tv'
#     spend_ooh: 'ooh'
#     spend_radio: 'radio'

# # C'est ici qu'on fixe les priors, pour chaque média, et dans le même ordre dans lequel ils sont indiqués dans graph - media_and_non_media_channels (les non medias exceptés)

# # ordre medias : google demand gen, google search, google pmax, google video, meta display, meta video, tv fixe, tv mobile, ooh
# priors:
#   baseline:
#     loc: 0.0
#     scale: 0.6   # baseline ~dominante mais flexible (≈60–75%)
#   media:
#     alpha:   # persistance (mémoire). plus haut = dure plus longtemps
#       low:  [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
#       high: [
#         0.60, # Radio
#         0.30, # Google Video (awareness, persistant mais pas excessif)
#         0.30, # Google Pmax (perf, court)
#         0.25, # Google Search (perf, court)
#         0.50, # Google Demand Gen
#         0.45, # Snap awareness & engagement
#         0.25, # Snap sales
#         0.25, # Meta sales
#         0.40, # Meta awareness
#         0.45  # Meta engagement
#       ]

#     ec:      # dépense à 50% de l’effet max. plus haut = il faut plus de budget pour avoir des résultats
#       loc: [
#         0.35, # Radio
#         0.40, # Google Video (cher à activer)
#         0.20, # Pmax
#         0.15, # Search (réactif)
#         0.5, # Demand Gen
#         0.40, # Snap A&E
#         0.15, # Snap sales
#         0.15, # Meta sales
#         0.4, # Meta awareness
#         0.3  # Meta engagement
#       ]
#       scale: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
#       low: 0.001
#       high: 10.0

#     beta:    # intensité brute (LogNormal, loc=0 => médiane neutre)
#       loc: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#       scale: [
#         0.09, # Radio
#         0.08, # Google Video (shrinkage vers 0, pour éviter de surestimer)
#         0.12, # Pmax
#         0.13, # Search (on laisse monter si les données le soutiennent)
#         0.11, # Demand Gen
#         0.1, # Snap A&E
#         0.13, # Snap sales
#         0.13, # Meta sales
#         0.1, # Meta awareness
#         0.1  # Meta engagement
#       ]

#   media_rf:
#     alpha:
#       low:  [0.0001, 0.0001, 0.0001]   # TV, OOH, Radio
#       high: [
#         0.95, # TV (mémoire forte)
#         0.40, # OOH (mémoire modérée)
#         0.60  # Radio
#       ]
#     ec:
#       loc: [
#         0.25, # TV (réactif)
#         0.7, # OOH
#         0.35  # Radio
#       ]
#       scale: [2.0, 2.0, 2.0]
#       low: 0.001
#       high: 10.0
#     beta:
#       loc: [
#         0.50, # TV (a priori fort)
#         0.00, # OOH
#         0.00  # Radio
#       ]
#       scale: [
#         0.20, # TV (liberté de monter)
#         0.08, # OOH
#         0.1  # Radio
#       ]

#   knot:
#     loc: 0.0
#     scale: 0.2

#   max_lag: 16

# on utilise les variables suivantes pour notre fonction custom qui trace la layer curve
# graph:
#   media_and_non_media_channels:
#     - 'radioaudio'
#     - 'google demand gen'
#     - 'google search'
#     - 'google pmax'
#     - 'google video'
#     - 'snap awareness and engagement'
#     - 'snap sales'
#     - 'meta sales'
#     - 'meta awareness'
#     - 'meta engagement'
#     - 'tv'
#     - 'ooh'
#     - 'radio'
#   nb_media_channels: 13
#   colors:
#     radioaudio: '#ff7f0e'
#     google demand gen: '#1f77b4'
#     google search: '#2ca02c'
#     google pmax: '#d62728'
#     google video: '#76c893'
#     snap awareness and engagement: 'lightgray'
#     snap sales: '#c5b0d5'
#     meta sales: '#9467bd'
#     meta awareness: '#8c564b'
#     meta engagement: '#bcbd22'
#     tv: '#7f7f7f'
#     ooh: '#17becf'
#     radio: '#ff7f0e'
#   geo_name: [
#     'Appel Entrantlandline',
#     'Appel Entrantmobile',
#     'Boutiqueslandline',
#     'Boutiquesmobile',
#     'Weblandline',
#     'Webmobile'
#   ]

# opti: # WIP
#   fixed_budget: True
#   budget: 28100000
#   pct_of_spend: None
#   spend_constraint_lower: None
#   spend_constraint_upper: None
#   target_roi: None
#   target_mroi: None
#   kpi: False
#   start_date : '2024-05-20'
#   end_date : '2025-05-19'
#   frequency : False

# Visualization and Optimization
# Visualization
# Integrated in Meridian model
# media_summary = visualizer.MediaSummary(mmm)
# media_summary.plot_contribution_waterfall_chart()
# image.png
# model_fit = visualizer.ModelFit(mmm)
# model_fit.plot_model_fit()
# image.png
# media_summary = visualizer.MediaSummary(mmm)
# media_summary.plot_channel_contribution_area_chart('quarterly')
# image.png
# media_summary.plot_channel_contribution_bump_chart()
# image.png
# media_summary.plot_roi_vs_effectiveness()
# image.png
# media_summary.plot_roi_bar_chart(include_ci=False)
# image.png
# media_effect = visualizer.MediaEffects(mmm)
# media_effect.plot_adstock_decay()
# image.png
# media_effect.plot_response_curves(confidence_level=0.5)
# image.png
# model_diagnostics.plot_prior_and_posterior_distribution('beta_m')

#Possible as well for ['beta_rf', 'alpha_m', 'alpha_rf', 'ec_m', 'ec_rf', 'knots_value', 'baseline_priors'] and other priors

# Comparing spending and contribution_per_kpi

from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(specs=[[{"secondary_y": True}]])

# Easily understandable color palette
colors = {
		'radioaudio': '#ff7f0e',
    'google_demand_gen': '#1f77b4',
    'google_search': '#17becf',
    'google_pmax': '#2ca02c',
    'google_video': '#76c893',
    'snap awareness and engagement':'lightgray',
    'snap sales':'#c5b0d5',
    'meta sales':'#9467bd',
    'meta_conversion': '#e377c2',
    'meta_awareness': '#9467bd',
    'meta engagement':'#bcbd22',
    'tv': '#8c564b',
    'ooh': '#7f7f7f',
    'ooh_fixe':'#d55e00',
    'radio':'#ff7f0e',
}

# Google Demand Gen
fig.add_trace(go.Scatter(
    name='Google Demand Gen - Contribution',
    x=df_baseline['date'],
    y=df_media_contrib['google demand gen'],
    mode="lines",
    line=dict(color=colors['google_demand_gen'])),
    secondary_y=False)

fig.add_trace(go.Scatter(
    name='Google Demand Gen - Spend',
    x=df_baseline['date'],
    y=df_baseline['spend_campaign_type_demand_gen'],
    mode="lines",
    line=dict(color=colors['google_demand_gen'], dash='dot')),
    secondary_y=True)

# Google Display
fig.add_trace(go.Scatter(
    name='Google Search - Contribution',
    x=df_baseline['date'],
    y=df_media_contrib['google search'],
    mode="lines",
    line=dict(color=colors['google_display'])),
    secondary_y=False)

fig.add_trace(go.Scatter(
    name='Google Search - Spend',
    x=df_baseline['date'],
    y=df_baseline['spend_campaign_type_search'],
    mode="lines",
    line=dict(color=colors['google_display'], dash='dot')),
    secondary_y=True)

# Google Pmax
fig.add_trace(go.Scatter(
    name='Google Pmax - Contribution',
    x=df_baseline['date'],
    y=df_media_contrib[
 'google pmax'],
    mode="lines",
    line=dict(color=colors['google_pmax'])),
    secondary_y=False)

fig.add_trace(go.Scatter(
    name='Google Pmax - Spend',
    x=df_baseline['date'],
    y=df_baseline['spend_campaign_type_pmax'],
    mode="lines",
    line=dict(color=colors['google_pmax'], dash='dot')),
    secondary_y=True)

# Google Video
fig.add_trace(go.Scatter(
    name='Google Video - Contribution',
    x=df_baseline['date'],
    y=df_media_contrib[
 'google video'],
    mode="lines",
    line=dict(color=colors['google_video'])),
    secondary_y=False)

fig.add_trace(go.Scatter(
    name='Google Video - Spend',
    x=df_baseline['date'],
    y=df_baseline['spend_campaign_type_video'],
    mode="lines",
    line=dict(color=colors['google_video'], dash='dot')),
    secondary_y=True)

# Meta Conversions
fig.add_trace(go.Scatter(
    name='Meta Conversions- Contribution',
    x=df_baseline['date'],
    y=df_media_contrib[
 'meta conversions'],
    mode="lines",
    line=dict(color=colors['meta_display'])),
    secondary_y=False)

fig.add_trace(go.Scatter(
    name='Meta Conversions- Spend',
    x=df_baseline['date'],
    y=df_baseline['spend_conversions'],
    mode="lines",
    line=dict(color=colors['meta_conversions'], dash='dot')),
    secondary_y=True)

# Meta Sales
fig.add_trace(go.Scatter(
    name='Meta Sales- Contribution',
    x=df_baseline['date'],
    y=df_media_contrib[
 'meta sales'],
    mode="lines",
    line=dict(color=colors['meta sales'])),
    secondary_y=False)

fig.add_trace(go.Scatter(
    name='Meta Conversions- Spend',
    x=df_baseline['date'],
    y=df_baseline['spend_sales'],
    mode="lines",
    line=dict(color=colors['meta sales'], dash='dot')),
    secondary_y=True)

# Meta Awareness
fig.add_trace(go.Scatter(
    name='Meta Awareness- Contribution',
    x=df_baseline['date'],
    y=df_media_contrib[
 'meta awareness'],
    mode="lines",
    line=dict(color=colors['meta awareness'])),
    secondary_y=False)

fig.add_trace(go.Scatter(
    name='Meta Awareness- Spend',
    x=df_baseline['date'],
    y=df_baseline['spend_awareness'],
    mode="lines",
    line=dict(color=colors['meta awareness'], dash='dot')),
    secondary_y=True)

# Meta Engagement
fig.add_trace(go.Scatter(
    name='Meta Engagement- Contribution',
    x=df_baseline['date'],
    y=df_media_contrib[
 'meta engagement'],
    mode="lines",
    line=dict(color=colors['meta engagement'])),
    secondary_y=False)

fig.add_trace(go.Scatter(
    name='Meta Engagement- Spend',
    x=df_baseline['date'],
    y=df_baseline['spend_engagement'],
    mode="lines",
    line=dict(color=colors['meta engagement'], dash='dot')),
    secondary_y=True)

# Snap Sales
fig.add_trace(go.Scatter(
    name='Snap sales- Contribution',
    x=df_baseline['date'],
    y=df_media_contrib[
 'snap sales'],
    mode="lines",
    line=dict(color=colors['snap sales'])),
    secondary_y=False)

fig.add_trace(go.Scatter(
    name='Snap sales- Spend',
    x=df_baseline['date'],
    y=df_baseline['spend_snapsales'],
    mode="lines",
    line=dict(color=colors['snap sales'], dash='dot')),
    secondary_y=True)

# Snap Awareness and engagement
fig.add_trace(go.Scatter(
    name='Snap awareness and engagement- Contribution',
    x=df_baseline['date'],
    y=df_media_contrib[
 'snap awareness and engagement'],
    mode="lines",
    line=dict(color=colors['snap awareness and engagement'])),
    secondary_y=False)

fig.add_trace(go.Scatter(
    name='Snap awareness and engagement- Spend',
    x=df_baseline['date'],
    y=df_baseline['spend_snapawarenessengagement'],
    mode="lines",
    line=dict(color=colors['snap awareness and engagement'], dash='dot')),
    secondary_y=True)

# TV
fig.add_trace(go.Scatter(
    name='TV- Contribution',
    x=df_baseline['date'],
    y=df_media_contrib[
 'tv'],
    mode="lines",
    line=dict(color=colors['tv'])),
    secondary_y=False)

fig.add_trace(go.Scatter(
    name='TV- Spend',
    x=df_baseline['date'],
    y=df_baseline['spend_tv'],
    mode="lines",
    line=dict(color=colors['tv'], dash='dot')),
    secondary_y=True)

# Radio
fig.add_trace(go.Scatter(
    name='Radio - Contribution',
    x=df_baseline['date'],
    y=df_media_contrib[
 'radio'],
    mode="lines",
    line=dict(color=colors['radio'])),
    secondary_y=False)

fig.add_trace(go.Scatter(
    name='Radio - Spend',
    x=df_baseline['date'],
    y=df_baseline['spend_radio'],
    mode="lines",
    line=dict(color=colors['radio'], dash='dot')),
    secondary_y=True)

# Radioaudio
fig.add_trace(go.Scatter(
    name='Radio audio- Contribution',
    x=df_baseline['date'],
    y=df_media_contrib[
 'radioaudio'],
    mode="lines",
    line=dict(color=colors['radio audio'])),
    secondary_y=False)

fig.add_trace(go.Scatter(
    name='Radio audio- Spend',
    x=df_baseline['date'],
    y=df_baseline['spend_radioaudio'],
    mode="lines",
    line=dict(color=colors['radioaudio'], dash='dot')),
    secondary_y=True)

# OOH
fig.add_trace(go.Scatter(
    name='OOH - Contribution',
    x=df_baseline['date'],
    y=df_media_contrib[
 'ooh'],
    mode="lines",
    line=dict(color=colors['ooh'])),
    secondary_y=False)

fig.add_trace(go.Scatter(
    name='OOH - Spend',
    x=df_baseline['date'],
    y=df_baseline['spend_ooh'],
    mode="lines",
    line=dict(color=colors['ooh'], dash='dot')),
    secondary_y=True)

# Layout
fig.update_layout(
    title="Meridian Prediction Decomposition vs Media Spend (dotted)",
    xaxis_title="Date",
    yaxis_title="Contribution / Spend",
    width=1300,
    height=700
)

fig.show()
# Custom layer curves chart, enabling a split by region

def create_media_contrib_geo(mmm, list_of_channel, nb_of_channel, geo_names):
    '''
    Goal : Being able to create a layer chart per region
    ---------
    Arguments :
    - mmm : meridian model with good enough metrics
    - list of channel : list with all the media considered
    - nb_of_channel : len(list_of_channel)
    - geo_names : list of the acquisition channels considered
    ---------
    Returns
    - df_media_contrib : for the geo, dataframe with the contribution (in kpi) per media channel throughout the period
    - df_baseline : Dataframe with a column date and a column baseline
    - df_actual : Actual recruitements made on the period
    - df_expected : Expected recruitements predicted by the model
    '''

    # geo_name = None if we want all acquisition channels
    percentiles = {}
    incremental_outcome= Analyzer(mmm).incremental_outcome(use_kpi=True, aggregate_times=False, aggregate_geos=False, selected_geos=geo_names)
    samples = tf.reshape(incremental_outcome, (-1, 125, nb_of_channel)) # Le 125 correspond au nombre de points de données existant
    samples_np = samples.numpy()

    for i, channel in enumerate(list_of_channel):
        percentiles[channel] = np.mean(samples_np[:, :, i], axis=0)
    weeks = np.arange(samples_np.shape[1])  # 0 to nb of weeks
    df_media_contrib = pd.DataFrame({'Week': weeks})

    # Add percentiles to DataFrame
    for channel in list_of_channel:
            df_media_contrib[channel] = percentiles[channel]

    model_fit = visualizer.ModelFit(mmm)
    df = model_fit._transform_data_to_dataframe(selected_geos=geo_names)
    df_actual = df[df['type']== 'actual'][['time', 'mean']]
    df_expected = df[df['type'] =='expected'][['time', 'mean']]
    df_baseline = df[df['type'] =='baseline'][['time', 'mean']]

    return df_media_contrib, df_baseline, df_actual, df_expected
fig_media_contrib = go.Figure()

# Add all components
fig_media_contrib.add_trace(
    go.Scatter(
        name='Baseline',
        x=df_baseline['time'],
        y=df_baseline['mean'],
        stackgroup="one",
        line = dict(color='orange')
    )
)
fig_media_contrib.add_trace(
    go.Scatter(
        name = 'Actual',
        x = df_baseline['time'],
        y = df_actual['mean'],
            mode = "lines",
            line = dict(color='red', width=2)
        )
    )
fig_media_contrib.add_trace(
    go.Scatter(
        name = 'Expected',
        x = df_baseline['time'],
        y = df_expected['mean'],
            mode = "lines",
            line = dict(color='blue', width=2)
        )
    )

colors = {
    'radioaudio': '#17becf',
    'google demand gen': '#1f77b4',
'google search': '#2ca02c',
'google pmax': '#d62728',
'google video': '#76c893',
'snap awareness and engagement': 'lightgray',
'snap sales': '#c5b0d5',
'meta conversions': '#e377c2',
'meta outcome sales': '#9467bd',
'meta outcome awareness': '#7f7f7f',
'meta outcome engagement': '#bcbd22',
'tv':'#8c564b',
'ooh': '#ff7f0e',
'radio': '#17becf'
}

media_list = [
    'radioaudio',
        'google demand gen',
        'google search',
        'google pmax',
        'google video',
        'snap awareness and engagement',
        'snap sales',
        'meta conversions',
        'meta outcome sales',
        'meta outcome awareness',
        'meta outcome engagement',
        'tv',
        'ooh',
        'radio',
                    ]

for media in media_list:
    fig_media_contrib.add_trace(
        go.Scatter(
            name=media.title().replace('_', ' '),  # Mise en forme du nom
            x=df_baseline['time'],
            y=df_media_contrib[media],
            stackgroup="one",
            line=dict(color=colors[media])
        )
    )

fig_media_contrib.update_layout(
    title="Meridian Prediction Decomposition compared to Actual",
    xaxis_title="Date",
    yaxis_title="Value",
    width=1300,
    height=700
)

fig_media_contrib.show()
# Running the optimization
budget_optimizer = optimizer.BudgetOptimizer(mmm)
# Different scenarios available
# The following optimization parameters are assigned default values based on the model input data:
# Flighting pattern. This is the relative allocation of a channel's media units across geos and time periods. By default, the historical flighting pattern is used. The default can be overridden by passing new_data.media. The flighting pattern is held constant during optimization and does not depend on the overall budget assigned to the channel.
# Cost per media unit. By default, the historical spend divided by historical media units is used. This can optionally vary by geo or time period or both depending on whether the spend data has geo and time dimensions. The default can be overridden by passing new_data.spend. The cost per media unit is held constant during optimization and does not depend on the overall budget assigned to the channel.
# Center of the spend box constraint for each channel. By default, the historical percentage of spend within selected_geos and between start_date and end_date is used. This can be overridden by passing pct_of_spend.
# Total budget to be allocated (for fixed budget scenarios only). By default, the historical spend within selected_geos and between start_date and end_date is used. This can be overridden by passing budget.
#Option 1 : fixed budget with constraints on media_spend

pct_of_spend = {
    'Channel0': 0.2,
    'Channel1': 0.2,
    'Channel2': 0.2,
    'Channel3': 0.1,
    'Channel4': 0.2,
    'Channel5': 0.1,
}
spend_constraint = {
    'Channel0': 0.3,
    'Channel1': 0.2,
    'Channel2': 0.3,
    'Channel3': 0.3,
    'Channel4': 0.3,
    'Channel5': 0.2,
}

optimization_results = budget_optimizer.optimize(
				use_posterior=True,
				selected_times=('2023-01-16', '2024-01-15'),
	      budget=58_000_000,
	      pct_of_spend=build_channel_args(**pct_of_spend),
	      spend_constraint_lower=build_channel_args(**spend_constraint),
	      spend_constraint_upper=build_channel_args(**spend_constraint),
	      use_optimal_frequency=False # because of the way radio data was built, better to use historical frequency use
)

#Option 2 : flexible budget, with a target ROI

optimization_results = budget_optimizer.optimize(
      selected_times=('2023-01-16','2024-01-15'),
      fixed_budget=False,
      spend_constraint_lower=0.5,
      spend_constraint_upper=0.5,
      target_roi=1,
)
# When using MLFlow, you need to create a new class :
class BudgetOptimizerWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, fitted_model, opti_config):
        """
        fitted_model : instance déjà prête du modèle MMM (déjà fit)
        opti_config  : dict des paramètres pour BudgetOptimizer.optimize
        """
        self.fitted_model = fitted_model
        self.opti_config = opti_config

    def predict(self, context, model_input):
        """
        Ici, model_input est ignoré si on rejoue toujours la même optimisation.
        """
        budget_optimizer = optimizer.BudgetOptimizer(self.fitted_model)
        optimization_results = budget_optimizer.optimize(**self.opti_config)
        return optimization_results
# Visualize optimization
# 1. Using MLflow model, which is supposed to have stored both unoptimized and optimized data
run_id = '5aa8fd29709f430491de48c29e9f9710' #run_id of your optimizatoin run

artifact_path = 'diagnostics/optimized data.csv' # retrieving optimized data
local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=artifact_path
            )
df_optimized = pd.read_csv(local_path)

artifact_path = 'diagnostics/unoptimized data.csv' # retrieving nonoptimized data
local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=artifact_path
            )

df_nonoptimized = pd.read_csv(local_path)

# ROI, CPIK and total spend for non optimized and optimized budget
roi_unoptimized = df_nonoptimized['incremental_outcome'].sum()/((df_nonoptimized['spend']/6).sum())
# Dividing spend by 6 as the model was trained on 6 different geographies, hence summing 6 times the real budget in our case
print(roi_unoptimized)
cost_per_kpi_unoptimized = (df_nonoptimized['spend']/6).sum()/df_nonoptimized['incremental_kpi'].sum()
print(cost_per_kpi_unoptimized)
inc_kpi_unoptimized = df_nonoptimized['incremental_kpi'].sum()
print(inc_kpi_unoptimized)

roi_optimized = df_optimized['incremental_outcome'].sum()/(df_optimized['spend']/6).sum()
print(roi_optimized)
cost_per_kpi_optimized = (df_optimized['spend']/6).sum()/df_optimized['incremental_kpi'].sum()
print(cost_per_kpi_optimized)

print(df_nonoptimized['spend'].sum()/6)
print(df_optimized['spend'].sum()/6)
# Using both unoptimized and optimized datasets, building the ‘casual graphs’ used when running optimization
import plotly.express as px

df_plot.sort_values(by='delta_spend', inplace=True, ascending=True)

# Ajouter une colonne couleur en fonction du signe
df_plot["color"] = df_plot["delta_spend"].apply(
    lambda x: "Positif" if x >= 0 else "Négatif"
)
df_plot["spend"] = df_plot["delta_spend"]/1e6

# Définir le mapping de couleurs
color_map = {
    "Positif": "lightskyblue",  # bleu ciel
    "Négatif": "lightcoral"     # rouge clair
}

# Créer le bar chart
fig = px.bar(
    df_plot,
    x="channel",
    y="delta_spend",
    color="color",
    color_discrete_map=color_map,
    title="Delta Spend par Canal",
    text = "spend"
)

fig.update_traces(
    texttemplate="%{text:,.1f} M€",
    textposition="outside"
)

fig.update_layout(
    xaxis_title="Canaux d'acquisition",
    yaxis_title="Delta Spend",
    width=2000,
    height=1000,
    # plot_bgcolor="white",
    # xaxis=dict(showgrid=False),
    # yaxis=dict(showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor="black"),
    showlegend=False  # cacher la légende si pas nécessaire
)

fig.show()

import plotly.graph_objects as go

df_plot.sort_values(by='delta_outcome', ascending=True,inplace=True)
labels = ["Unoptimized outcome"] + df_plot["channel"].tolist() + ["Optimized outcome"]
values = [unoptimized] + df_plot["delta_outcome"].tolist() + [0]  # dernière valeur sera "total"

# Définir le type de chaque barre
measure = ["absolute"] + ["relative"]*len(df_plot) + ["total"]

# Définir les couleurs pour delta positive/négative
base_color = "blue"
increase_color = "lightskyblue"
decrease_color = "lightcoral"

text_values = [""] + [f"{v/1e6:+.2f} M€" for v in df_plot["delta_outcome"]] + [""]

fig = go.Figure(go.Waterfall(
    name="Outcome",
    orientation="v",  # vertical
    measure=measure,
    x=labels,
    y=values,
    text=text_values,
    decreasing={"marker":{"color": decrease_color}},
    increasing={"marker":{"color": increase_color}},
    totals={"marker":{"color": "blue"}}
))

fig.update_layout(
    title="Waterfall Outcome per Media",
    yaxis_title="Outcome (M€)",
    xaxis_title="",
    width=2000,
    height=1000,
    template="plotly_white"
)

fig.show()

x_order = ['TV', 'OOH', 'RADIO', 'META_salesconversions', 'GOOGLE_search', 'GOOGLE_pmax', 'META_awareness', 'GOOGLE_video', 'SNAP_sales', 'GOOGLE_demand_gen', 'META_engagement', 'SNAP_awarenessengagement']

df_plot_optimized = df_optimized.copy()
df_plot_optimized['spend'] = df_optimized['spend']/6
total = df_plot_optimized["spend"].sum()

df_plot_optimized["share_pct"] = df_plot_optimized["spend"] / total *100
# Texte affiché = "xx% (xxx €)"

df_plot_optimized['channel'] = df_plot_optimized['channel'].replace({
    'tv':'TV',
    'ooh':'OOH',
    'radioaudio':'RADIO',
    'meta sales':'META_salesconversions',
    'search':'GOOGLE_search',
    'pmax':'GOOGLE_pmax',
    'meta awareness':'META_awareness',
    'google video':'GOOGLE_video',
    'snapchat sales':'SNAP_sales',
    'google demande gen':'GOOGLE_demand_gen',
    'meta engagement':'META_engagement',
    'snapchat awareness and engagement':'SNAP_awarenessengagement'
})

# aggregate: sum all numeric columns for each channel
df_plot_optimized = (
    df_plot_optimized
    .groupby("channel", as_index=False)
    .sum(numeric_only=True)   # keeps only numeric columns summed
)
df_plot_optimized["label"] = df_plot_optimized.apply(lambda row: f"{row['share_pct']:.1f}% ({row['spend']/1e6:,.1f} M€)", axis=1)

def map_group(channel):
    if channel.startswith("META_"):
        return "META"
    elif channel.startswith("GOOGLE_"):
        return "GOOGLE"
    elif channel.startswith("SNAP_"):
        return "SNAP"
    elif channel in ["TV", "OOH", "RADIO"]:
        return channel
    else:
        return "AUTRE"

df_plot_optimized["group"] = df_plot_optimized["channel"].map(map_group)

color_map = {
    "META": "#1f77b4",
    "GOOGLE": "#ff7f0e",
    "SNAP": "#2ca02c",
    "TV": "#9467bd",
    "OOH": "#8c564b",
    "RADIO": "#e377c2"
}

df_plot_optimized = df_plot_optimized[df_plot_optimized['group'].isin(["META", "GOOGLE", "SNAP", "tv", "ooh", "RADIO"])]

fig = px.bar(
    df_plot_optimized,
    x="channel",
    y="share_pct",
    color="group",
    text=df_plot_optimized["label"],
    color_discrete_map=color_map,
    title="Répartition des dépenses média optimisées sur la période",
    category_orders={"channel": x_order}
)

fig.update_traces(
    textposition="outside",
    hovertemplate="%{x}<br>%{y:,.0f} M€<br>%{customdata:.1f}%<extra></extra>",
    customdata=df_plot_optimized["share_pct"]  # pour tooltips propres
)

fig.update_layout(
    yaxis_title="Part of total spendings (%)",
    xaxis_title="Canal",
    template="plotly_white",
    height=600,
    width=1400,
    uniformtext_minsize=8,
    # uniformtext_mode="hide"
)

fig.show()
# 2. If you have time/a quick running time and access to optimization_results easily

# Maybe the most important plot
optimization_results.plot_response_curves()
# Careful : the spend on the x-axis corresponds to the actual spend * nb of acquisition channels considered —> to be taken into account when fixing the spendings of each media channel
# Comparison non-optimized vs optimized
optimization_results.plot_incremental_outcome_delta()

optimization_results.plot_spend_delta()

optimization_results.plot_budget_allocation(False) # Non optimized spend

optimization_results.plot_budget_allocation(True) # Optimized spend

filepath = 'optimization'
optimization_results.output_optimization_summary(
    'optimization_output.html', filepath
)

IPython.display.HTML(filename='optimization/optimization_output.html')