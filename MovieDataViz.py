# Dash imports
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


# Plotly imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Other imports
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
import re
import time

# Data preprocessing
# reading the datasets
movies_rating = pd.read_csv('tmdb_5000_credits.csv')
movies_detail = pd.read_csv('tmdb_5000_movies.csv')

# dropping off unwanted columns
drop_cols=['homepage','overview','tagline']
movies_detail.drop(columns=drop_cols,inplace=True)
movies_detail=movies_detail.dropna()
movies_rating=movies_rating.dropna()

# merging both the datasets
merged_movies_df= pd.merge(movies_detail,movies_rating, left_on='id', right_on='movie_id', how='left').drop(columns='id', axis=1)
merged_movies_df["release_year"]=merged_movies_df["release_date"].apply(lambda x : int(x.split("-")[0]))
merged_movies_df.drop(columns='release_date',inplace=True)

# parsing json data
def json_load(colNames,df):
    for name in colNames:    
        df[name]=df[name].apply(lambda i : json.loads(i))
        value='name'
        if name =="production_countries":
            id_name="iso_3166_1"
        elif name =="spoken_languages":
            id_name="iso_639_1"
        elif name =="cast":
            id_name="gender"
        elif name =="crew":
            id_name="job"
        else:
            id_name='id'
            
        df[name+"_ids"]=df[name].apply(lambda x :','.join([str(i.get(id_name)).strip() for i in x]))
        df[name+"_names"]=df[name].apply(lambda x : ','.join([i.get(value).strip() for i in x]))

        
# transforming json data into individual columns        
json_columns=["genres","keywords","production_companies","production_countries","spoken_languages","cast"]        
json_load(json_columns,merged_movies_df)
merged_movies_df.drop(columns=json_columns,inplace=True)

# pre-processing data for word cloud based on keywords
# method to obtain frequency of each word in a column
def get_frequency(name,df):
    melted = df[name+"_names"].str.split(",", expand=True).reset_index().melt(id_vars="index", value_name=name+"_names")
    return melted.groupby(name+"_names").count().sort_values(by='index',ascending=False).drop(index='')

# method to generate wordcloud
def generate_wordcloud(frequency_dict):
    mask=np.array(Image.open("camera.png"))
    wordcloud=WordCloud(background_color="white",mask=mask)
    wordcloud.generate_from_frequencies(frequency_dict)
    return wordcloud

# generating keywords word cloud
keywords_dict = get_frequency("keywords",merged_movies_df).iloc[:,1].to_dict()
fig_keywords_cloud = px.imshow(generate_wordcloud(keywords_dict))
fig_keywords_cloud.update_layout({'title': "Word Cloud based on keywords of movies",'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}, 'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}, 'hovermode' : False})

# pre-processing data for distribution of genres
genres_dict = get_frequency("genres",merged_movies_df).iloc[:,1].to_dict()
figure_gen_bar=px.bar(y=list(genres_dict.keys()),x=list(genres_dict.values()),orientation='h')
figure_gen_bar.update_layout({'title': "Genre wise distribution of films",
    'xaxis_title' : "No of films",
    'yaxis_title' : "Genres"
})

# pre-processing data for choropleth based on no of films produced by each country
# map type 
countries = get_frequency("production_countries",merged_movies_df).drop(columns="index")
countries.rename(columns={'variable' : 'NoofMovies'},inplace=True)
countries = countries[1:]
countrywise_film_map = px.choropleth(countries, locations=countries.index,
                    color="NoofMovies", 
                    locationmode='country names',
                    hover_name=countries.index, 
                    color_continuous_scale=px.colors.sequential.Agsunset_r)
countrywise_film_map.update_layout({'title': "Distribution of films by Production countries (excluding USA)"})

# globe type 
film_countrywise = dict(
    type = 'choropleth',
    locations = countries.index,
    locationmode='country names',
    colorscale = 'agsunset_r',
    autocolorscale = False,
    z=countries.values.flatten())
countrywise_film_globe = go.Figure(data=[film_countrywise])

# change projection type to orthographic for spherical globe
countrywise_film_globe.update_geos(projection_type="natural earth")
countrywise_film_globe.update_layout(height=600, margin={"r":0,"l":0,"b":0})

countrywise_film_globe.update_layout({'title': "Distribution of film by Production countries excluding USA", 'yaxis_title' : "No of films"})

# bubble chart representation of the profit based on directors
# retrieving director_data from crew
id_name="job"
name="crew"
value='name'
merged_movies_df[name]=merged_movies_df[name].apply(lambda i : json.loads(i))
merged_movies_df["Director"]=merged_movies_df[name].apply(lambda x : ','.join([i.get(value).strip() for i in x if i.get(id_name)=="Director"]))         
merged_movies_df.drop(columns=name,inplace=True)
top_directors = pd.pivot_table(merged_movies_df, values="movie_id", 
                               index=["Director"],aggfunc=len).sort_values(by='movie_id',ascending=False).drop(index='')[0:5]

# getting year_wise director data
director_data=pd.DataFrame()
for i in top_directors.index:
    temp =merged_movies_df[merged_movies_df["Director"]==i]
    revenue=temp.groupby(['release_year'])['revenue'].agg('sum')
    budget=temp.groupby(['release_year'])['budget'].agg('sum')
    new_df = pd.concat([revenue,budget],axis=1)
    new_df["Director"]=i
    director_data=pd.concat([director_data, new_df],axis=0)

director_bubblechart = px.scatter(director_data, x=director_data.index, y="budget",
                                size="revenue", color="Director",
                                hover_name="Director", log_x=True, size_max=60)

director_bubblechart.update_layout({'title': "Profitability of top directors over the years", 'yaxis_title' : "Budget",'xaxis_title' : "Year"})

# pre-processing data for year wise average profit of various production houses
avg_profit = (merged_movies_df['revenue']-merged_movies_df['budget']).mean()

warner_bros = merged_movies_df[merged_movies_df['production_companies_names'].str.contains("Warner Bros")]
avg_profit_wb = (warner_bros['revenue'] - warner_bros['budget']).mean()

walt_disney = merged_movies_df[merged_movies_df['production_companies_names'].str.contains("Walt Disney Pictures")]
avg_profit_wd = (walt_disney['revenue'] - walt_disney['budget']).mean()

marvel_studios = merged_movies_df[merged_movies_df['production_companies_names'].str.contains("Marvel Studios")]
avg_profit_ms = (marvel_studios['revenue'] - marvel_studios['budget']).mean()

wb_fig = go.Figure()

wb_fig.add_trace(go.Indicator(
    mode = "number+gauge+delta", value = avg_profit_wb,
    delta = {'reference': avg_profit},
    domain = {'x': [0.25, 1], 'y': [0.08, 0.25]},
    title = {'text': 'Warner Bros'},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 700000000]},
        'threshold': {
            'line': {'color': "black", 'width': 2},
            'thickness': 0.75,
            'value': avg_profit},
        'steps': [
            {'range': [0, 200000000], 'color': "gray"},
            {'range': [200000000, 400000000], 'color': "lightgray"}],
        'bar': {'color': "black"}}))

wb_fig.add_trace(go.Indicator(
    mode = "number+gauge+delta", value = avg_profit_wd,
    delta = {'reference': avg_profit},
    domain = {'x': [0.25, 1], 'y': [0.4, 0.6]},
    title = {'text': 'Walt Disney Pictures'},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 700000000]},
        'threshold': {
            'line': {'color': "black", 'width': 2},
            'thickness': 0.75,
            'value': avg_profit},
        'steps': [
            {'range': [0, 200000000], 'color': "gray"},
            {'range': [200000000, 400000000], 'color': "lightgray"}],
        'bar': {'color': "black"}}))

wb_fig.add_trace(go.Indicator(
    mode = "number+gauge+delta", value = avg_profit_ms,
    delta = {'reference': avg_profit},
    domain = {'x': [0.25, 1], 'y': [0.7, 0.9]},
    title = {'text': 'Marvel Studios'},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 700000000]},
        'threshold': {
            'line': {'color': "black", 'width': 2},
            'thickness': 0.75,
            'value': avg_profit},
        'steps': [
            {'range': [0, 200000000], 'color': "gray"},
            {'range': [200000000, 400000000], 'color': "lightgray"}],
        'bar': {'color': "black"}}))

wb_fig.update_layout({'title': " How promising are the top production houses ?"})
# wb_fig.update_layout(height = 400 , margin = {'t':0, 'b':0, 'l':0})

# pre-processing data for average rating of directors
jc_avg_rating = merged_movies_df[merged_movies_df['Director'].str.contains("James Cameron")]['vote_average'].mean()
ss_avg_rating = merged_movies_df[merged_movies_df['Director'].str.contains("Steven Spielberg")]['vote_average'].mean()
cn_avg_rating = merged_movies_df[merged_movies_df['Director'].str.contains("Christopher Nolan")]['vote_average'].mean()
ap_avg_rating = merged_movies_df[merged_movies_df['Director'].str.contains("Alexander Payne")]['vote_average'].mean()
ml_avg_rating = merged_movies_df[merged_movies_df['Director'].str.contains("Mike Leigh")]['vote_average'].mean()

directors=['James Cameron', 'Steven Spielberg', 'Christopher Nolan', 'Alexander Payne', 'Mike Leigh']

director_rating_fig = go.Figure([go.Bar(x=directors, 
                        y=[jc_avg_rating, ss_avg_rating, cn_avg_rating, ap_avg_rating, ml_avg_rating])])

director_rating_fig.update_layout(
    title="Rating of Directors",
    xaxis_title="Directors",
    yaxis_title="Ratings",
    yaxis=dict(range=[5, 8.5])
)

# pre-processing data for comaprison between actors
dicaprio = merged_movies_df[merged_movies_df['cast_names'].str.contains("Leonardo DiCaprio")]
tom_hanks = merged_movies_df[merged_movies_df['cast_names'].str.contains("Tom Hanks")]
johnny_depp = merged_movies_df[merged_movies_df['cast_names'].str.contains("Johnny Depp")]

dicaprio = dicaprio.groupby('release_year').agg({'revenue': 'sum'})
tom_hanks = tom_hanks.groupby('release_year').agg({'revenue': 'sum'})
johnny_depp = johnny_depp.groupby('release_year').agg({'revenue': 'sum'})

years = [year for year in range(1995, 2016)]
dicaprio_values = [value[0] for value in dicaprio.values.tolist()]
tom_hanks_values = [value[0] for value in tom_hanks.values.tolist()]
johnny_depp_values = [value[0] for value in johnny_depp.values.tolist()]

actor_fig = go.Figure()

actor_fig.add_trace(go.Bar(x=years,
                y=dicaprio_values,
                name='Leonardo DiCaprio',
                marker_color='rgb(55, 83, 109)'
            ))

actor_fig.add_trace(go.Bar(x=years,
                y=tom_hanks_values,
                name='Tom Hanks',
                marker_color='rgb(26, 118, 255)'
            ))

actor_fig.add_trace(go.Bar(x=years,
                y=johnny_depp_values,
                name='Johnny Depp',
                marker_color='rgb(0, 206, 235)'
            ))

actor_fig.update_layout(
    title='Collective revenue of films based on actor',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='USD (billions)',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)

# pre-processing data for distribution of movies by languages
melted = movies_detail["original_language"].str.split(",", expand=True).reset_index().melt(id_vars="index", value_name="original_language")
lang_dict = melted.groupby("original_language").count().sort_values(by='index',ascending=False).iloc[:,1].to_dict()
lang_labels = ['French', 'Spanish', 'Chinese', 'German', 'Hindi', 'Japanese', 'Italian', 'Chinese (Singapore)', 'Russian', 'Korean', 'Portuguese', 'Danish', 'Others']
lang_count = list(lang_dict.values())
display_count = lang_count[1:13]
other_langs = sum(lang_count[13:])
display_count.append(other_langs)
 
movie_lang_fig = {
    'data':[go.Pie(labels=lang_labels, values=display_count, hole=.3, textinfo='none')],
    'layout' : {'title' : 'Film distribution by lanaguage (excluding English)'}
 }

# pre-processing data for distribution of film across years
yearwise_films=pd.pivot_table(merged_movies_df,index="release_year",values="movie_id",aggfunc=len)
yearwise_films.columns=["noOfFilms"]
df = yearwise_films.reset_index()

movie_year_fig=px.line(df, x="release_year", y="noOfFilms", title='Over the years')
movie_year_fig.update_xaxes(rangeslider_visible=True)
movie_year_fig.update_layout(title = 'No of films vs year',
                xaxis={'title': 'Years'},
                yaxis={'title': 'No of films'},

                )

# Graphs
keywords_word_cloud = dcc.Graph(
    id = 'Keyword-word-cloud',
    figure = fig_keywords_cloud
)

genres_histogram = dcc.Graph(
    id = 'Genres-histogram',
    figure = figure_gen_bar
)

films_count_choropleth_map = dcc.Graph(
    id = 'Films-count-choropleth-map',
    figure = countrywise_film_map
)

films_count_choropleth_globe = dcc.Graph(
    id = 'Films-count-choropleth-globe',
    figure = countrywise_film_globe
)

director_profit_bubble = dcc.Graph(
    id = 'Director-profit-bubble',
    figure = director_bubblechart
)

wb_avg_profit = dcc.Graph(
    id = 'Wb_avg_profit',
    figure = wb_fig
)

director_rating_bar = dcc.Graph(
    id = 'director_avg_rating',
    figure = director_rating_fig
)

actor_multi_bar = dcc.Graph(
    id = 'actor_multi_bar',
    figure = actor_fig
)

movie_lang_pie = dcc.Graph(
    id = 'movie_lang_pie',
    figure = movie_lang_fig
)

movie_year_line = dcc.Graph(
    id = 'movie_year_line',
    figure = movie_year_fig
)


# CSS
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Custom css
colors = {
    'background': '#111111',
    'text': '#7FDBFF',
}

# Dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(style={
    'backgroundColor': colors['background'],
    'color': colors['text'],
    'padding': '5%',
    'padding-left': '5%',
    'min-height': '100vh'
    }, 
    children=[
        html.H1(children='Start! Camera! Action!'),
    
        html.Div(children=['''
            Dashboard analysing 5000 movies over the years
        ''',
        
        html.Div(children=[
            html.P(style={
                'color': 'white',
                'margin-top': '20px'
                },
                children='Please select a visualization from the given tabs:'
            ),
            dcc.Tabs(id='tabs', value='tab-1', style={'color':'black'}, children=[
                
                dcc.Tab(label='Regionwise analysis', value='tab-1', children=[
                    html.Div(style={
                            'color': 'black',
                            'background': 'white',
                            'padding': '40px'
                            },
                        children=[
                            html.P(style={
                                'color': 'black',
                                'padding-top': '20px',
                                'padding-left': '40px'
                                },
                                children='Please select the type of map:'
                            ),
                            dcc.Dropdown(id='drop-down',
                                style={
                                'color': 'black',
                                'width': '65%',
                                'margin-top': '10px',
                                'margin-left': '20px'
                                },
                                options=[
                                    {'label': 'Globe', 'value': 'globe'},
                                    {'label': 'Standard Map', 'value': 'map'},
                                ],
                                value='globe'
                            ),
                            
                            html.Div(children=[
                                html.Div(id='choropleth')    
                            ])
                    ])
                ]),

               
                dcc.Tab(label='Over the years', value='tab-2', children=movie_year_line),
                dcc.Tab(label='Linguistic Distribution', value='tab-3', children=movie_lang_pie),
                dcc.Tab(label='Genre Analysis', value='tab-4', children=genres_histogram),               
                dcc.Tab(label='Keywords wordcloud', value='tab-5', children=keywords_word_cloud),
                dcc.Tab(label='Profitability of directors', value='tab-6', children=director_profit_bubble),
                dcc.Tab(label='Top Director Rating', value='tab-7', children=director_rating_bar),               
                dcc.Tab(label='Actors Revenue comparison', value='tab-8', children=actor_multi_bar),
                dcc.Tab(label='Production house performance', value='tab-9', children=wb_avg_profit),
            ]),
        ]),
        
    ])
])


@app.callback(
    Output(component_id='choropleth', component_property='children'),
    [Input(component_id='drop-down', component_property='value')]
)
def display_choropleth(data):
    if data == 'globe':
        return films_count_choropleth_globe
    elif data == 'map':
        return films_count_choropleth_map


if __name__ == '__main__':
    app.run_server(debug=False)