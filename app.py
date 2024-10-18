# coding=utf-8
# import libraries
import requests
import base64
import subprocess
import pandas as pd
import streamlit as st
from annoy import AnnoyIndex
import os
import math
import warnings
from unidecode import unidecode

warnings.simplefilter("ignore")

# variables
all_name = "All"
photo_profile_dir = "profile_photo/"

for file in os.listdir():
    if file.endswith(".png"):
        os.remove(file)

# load data

st.set_page_config(
    page_title="Similar Player Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv("data/datafm20.csv")
    df["Name"] = df["Name"].apply(lambda name: unidecode(name))
    # df["positions_list"] = df["positions"].apply(lambda x: x.split(","))
    
    # df["contract"] = df["contract"].astype(int)
    #df = df[df.Value > 1000000]

    return df


df = load_data()
league_list = list(df["Division"].unique())
player_list = list(df["Name"].unique())

#default_leagues = [
#    "Spanish First Division",
#    "Italian Serie A",
#    "Ligue 1 Conforama",
#    "English Premier Division",
#    "Bundesliga",
#    "Eredivisie",
#    "Sky Bet Championship"
#]

default_leagues = league_list

default_positions = ['AM (R)', 'AM (L)', 'AM (C)']

positions_list = list(df['Best Pos'].unique())


show_columns = [
    "Name",
    "Club",
    "Division",
    "Age",
    "Position",
    "Wage",
    "Value"
]

default_columns_to_compare = list(df.columns[17:])

possible_columns_to_compare = list(default_columns_to_compare)


################################################################
# css style
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Annoy (Approximate Nearest Neighbors Oh Yeah) is a C++ library with Python bindings to search for points in space that are close to a given query point. It also creates large read-only file-based data structures that are mmapped into memory so that many processes may share the same data.
# https://github.com/spotify/annoy

##################################################################
# sidebar filters
st.sidebar.title(":pick: Filters")

st.sidebar.title("Target:")

target_player_name = st.sidebar.selectbox("Player:", [""] + player_list)

target_player_name = target_player_name.strip()

st.sidebar.title("Similar Player Conditions:")

leagues = st.sidebar.multiselect(
    "League:", [all_name] + league_list, default=default_leagues
)

positions = st.sidebar.multiselect(
    "Position:", positions_list, default=default_positions
)

age = st.sidebar.slider("Age:", min_value=15, max_value=50, value=27)

transfer_fee = 1000000 * float(
    st.sidebar.text_input("Maximum Transfer Fee (€M):", "100")
)
wage = 1000 * float(st.sidebar.text_input("Maximum Wage (€K):", "200"))

columns_to_compare = st.sidebar.multiselect(
    "KPIs:", possible_columns_to_compare, default=default_columns_to_compare
)

top_K = st.sidebar.slider("K Top Similar Players", min_value=0, max_value=20, value=10)

is_scan = st.sidebar.button("Detect")

st.sidebar.image(
    "agent.jpg",
    caption="https://www.wikihow.com/Become-a-Football-Agent",
    use_column_width=True,
)
st.sidebar.header("Contact Info")
st.sidebar.info("hadisotudeh1992[at]gmail.com")

##############################################################################
# if detect button is clicked, then show the main components of the dashboard


def filter_positions(row, positions):
    for p in positions:
        if p in (row["Position"]):
            return True
    return False





def create_table(data, width=100, class_="", image_height=95, image_width=95):
    if len(class_) > 0:
        table = f'<table class="{class_}" style="text-align: center; width:{width}%">'
    else:
        table = f'<table style="text-align: center; width:{width}%">'

    # create header row
    header_html = "<tr>"
    for col in data.columns:

        if col == "Value":
            header_html = header_html + "<th>Value (€M)</th>"
        elif col == "player_hashtags":
            header_html = header_html + "<th>Description</th>"
        else:
            header_html = header_html + f"<th>{col.capitalize()}</th>"
    header_html = header_html + "<tr>"

    all_rows_html = ""
    for row_index in range(len(data)):
        row_html = "<tr>"
        row = data.iloc[row_index]
        for col in data.columns:
            if row[col] == None:
                row_html = row_html + "<td></td>"
            elif col == "positions":
                row_html = row_html + f'<td>{", ".join(eval(row[col]))}</td>'
            else:
                row_html = row_html + f"<td>{row[col]}</td>"
        row_html = row_html + "</tr>"
        all_rows_html = all_rows_html + row_html

    table = table + header_html + all_rows_html + "</table>"
    st.markdown(table, unsafe_allow_html=True)


# @st.cache(allow_output_mutation=True)
def scan(target_player, leagues, positions, transfer_fee, wage, age):
    df = load_data()

    target_player_KPIs = target_player[columns_to_compare].to_numpy()[0]

    df = df.loc[df["Name"] != target_player_name]
    df = df[df["Age"] <= age]
    if all_name not in leagues:
        df = df[df["Division"].isin(leagues)]
    df = df[(df["Value"] <= transfer_fee) & (df["Wage"] <= wage)]

    df["filter_positions"] = df.apply(
        lambda row: filter_positions(row, positions), axis=1
    )
    search_space = df.loc[df["filter_positions"] == True]
    search_space.reset_index(drop=True, inplace=True)

    # search_space["label"] = pd.Series(list(clf.fit_predict(X)))
    # search_space["score"] = pd.Series(list(clf.score_samples(X)))
    # search_space.sort_values(by=["score"], inplace=True)

    # calculate ANNOY
    annoy = AnnoyIndex(len(columns_to_compare), "euclidean")
    search_space_array = search_space[columns_to_compare].to_numpy()

    for i in range(search_space_array.shape[0]):
        annoy.add_item(i, search_space_array[i, :])
    annoy.build(n_trees=1000)

    indices = annoy.get_nns_by_vector(target_player_KPIs, top_K)
    return pd.concat([search_space.iloc[index : index + 1, :] for index in indices])


#@st.cache(allow_output_mutation=True)
def calc_target_player(target_player_name):
    target_player = df.loc[df["Name"] == target_player_name]
    return target_player


if is_scan:
    target_player = calc_target_player(target_player_name)
    target_player_age = target_player["Age"].iloc[0]
    target_player_teams = target_player["Club"].iloc[0]
    st.title("Target Player:")
    st.markdown(
        f" **{target_player_name}** - **{target_player_teams}**"
    )
    result = scan(target_player, leagues, positions, transfer_fee, wage, age)
    st.markdown(f"**Top _{top_K}_ most similar players are**:")
    result["Value"] = result["Value"].apply(lambda v: str(v / 1000000))
    create_table(result[show_columns])
else:
    st.title(":male-detective: Similar Player Detector")
    st.subheader(
        "This app makes use of [EA SPORTS™ FIFA 2020](https://sofifa.com) KPIs to search for similar players to a given one."
    )
    st.subheader(
        "It first applies filters such as league, age, and market value on players. Then, each remaining player is considered as a vector of their KPIs and afterwards [Annoy (Approximate Nearest Neighbors Oh Yeah)](https://github.com/spotify/annoy) is used to search for players (points) in space that are close to a given query."
    )
    st.image("annoy.jpg", caption="https://github.com/spotify/annoy")
