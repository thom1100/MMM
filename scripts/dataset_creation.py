import pandas as pd

def create_final_df(col:str):
    media_variables = pd.read_excel("../data/Dataset UK L'Oreal Paris Haircare - HEC Training.xlsx", sheet_name = "A&P Variables")
    target_variables = pd.read_excel("../data/Dataset UK L'Oreal Paris Haircare - HEC Training.xlsx")
    control_variables = pd.read_excel("../data/Dataset UK L'Oreal Paris Haircare - HEC Training.xlsx", sheet_name = "Commercial Variables")


    controls_target_variables = control_variables.merge(target_variables, on=["Starting Week", "Year"], how="left", validate="one_to_one").sort_values("Starting Week", ascending = True)
    controls_target_variables.rename(columns= {"Starting Week": "Starting week"}, inplace = True)
    media_tv = media_variables[media_variables["growth_driver_l4"]=="traditional_tv"]
    media_engagements = media_variables[media_variables["metric"]=="engagements"]
    media_shopper_experience = media_variables[media_variables["growth_driver_l4"]=="testers_and_merchandising"]
    media_variables = media_variables[media_variables["metric"]=="impressions"]

    media_impressions = media_variables.groupby(["Starting week", col]).agg(
        spend = ("investment (in pound)", "sum"),
        impressions=("execution", "sum")
    ).reset_index()

    media_impressions[col] = media_impressions[col].str.lower()

    df_impressions = (
        media_impressions
        .pivot_table(
            index='Starting week',
            columns=col,
            values=['spend', 'impressions'],
            aggfunc='sum',
            fill_value=0
        )
    )
    df_impressions.columns = [
        f"{metric}_{media}"
        for metric, media in df_impressions.columns
    ]

    # optional: reset index
    df_impressions = df_impressions.reset_index().sort_values("Starting week", ascending = True)
    media_tv = media_tv.groupby(["Starting week", col]).agg(
        spend=("investment (in pound)", "sum"),
        grp = ("execution", "mean")
    )
    df_grp = (
        media_tv
        .pivot_table(
            index='Starting week',
            columns=col,
            values=['spend', 'grp'],
            aggfunc='sum',
            fill_value=0
        )
    )
    df_grp.columns = [
        f"{metric}_{media}"
        for metric, media in df_grp.columns
    ]

    df_grp = df_grp.reset_index().sort_values("Starting week", ascending = True)
    media_engagements = media_engagements.groupby(["Starting week", col]).agg(
        spend=("investment (in pound)", "sum"),
        engagement = ("execution", "sum")
    )
    df_engagement = (
        media_engagements
        .pivot_table(
            index='Starting week',
            columns=col,
            values=['spend', 'engagement'],
            aggfunc='sum',
            fill_value=0
        )
    )
    df_engagement.columns = [
        f"{metric}_{media}"
        for metric, media in df_engagement.columns
    ]

    df_engagement = df_engagement.reset_index().sort_values("Starting week", ascending = True)
    # let's merge everything

    final_df = pd.concat([controls_target_variables, df_impressions, df_engagement, df_grp], axis = 1)
    return final_df