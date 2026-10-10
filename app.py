from datetime import date
from pathlib import Path
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).resolve().parent
MODELS = BASE / 'models'
DATA = BASE / 'data' / 'Sri_Lanka_Paddy_Production.csv'
GREEN = '#237653'
st.set_page_config(page_title='Paddy Insights | Sri Lanka', page_icon='🌾', layout='wide')
st.markdown('''
<style>
.block-container {max-width:1240px; padding-top:2rem; padding-bottom:3rem;}
h1,h2,h3 {letter-spacing:-.035em;}
.hero {padding:2rem 2.2rem; border-radius:20px; margin-bottom:1.5rem; background:linear-gradient(115deg,#123e30,#237653); color:white;}
.hero .eyebrow {font-size:.78rem; font-weight:700; letter-spacing:.16em; color:#cdebd6;}
.hero h1 {color:white; font-size:clamp(2rem,4vw,3rem); margin:.4rem 0; padding:0;}
.hero p {color:#e0efe5; max-width:680px; margin:.5rem 0 0; line-height:1.65;}
[data-testid="stMetric"] {padding:1rem 1.2rem; border:1px solid #dce8df; border-radius:14px;}
[data-testid="stMetricValue"] {font-size:1.8rem;}
.stButton > button,.stFormSubmitButton > button {border-radius:10px; min-height:2.8rem;}
button[data-baseweb="tab"] {font-weight:600; padding-left:1rem; padding-right:1rem;}
@media(max-width:640px) {.block-container {padding-top:1rem;} .hero {padding:1.4rem;}}
</style>
<div class="hero"><div class="eyebrow">SRI LANKA · PADDY ANALYTICS</div>
<h1>🌾 Paddy Insights</h1><p>Explore district-level paddy production, build a production scenario,
and compare five machine learning models in one place.</p></div>
''', unsafe_allow_html=True)

@st.cache_resource
def load_models():
    meta = json.loads((MODELS / 'metadata.json').read_text(encoding='utf-8'))
    return meta, {name: joblib.load(MODELS / filename) for name, filename in meta['models'].items()}

@st.cache_data
def load_data():
    return pd.read_csv(DATA)

@st.cache_data
def load_results():
    return pd.read_csv(MODELS / 'metrics.csv'), pd.read_csv(MODELS / 'test_predictions.csv')

if not (MODELS / 'metadata.json').exists():
    st.error('Trained models are missing. Run `python train_models.py`, then refresh this page.')
    st.stop()
try:
    meta, models = load_models()
    metrics, test_predictions = load_results()
    data = load_data()
except (FileNotFoundError, ImportError, ValueError) as exc:
    st.error('Could not load the data or models. Install the requirements and run `python train_models.py`.')
    with st.expander('Technical details'):
        st.code(str(exc))
    st.stop()

ranked = metrics.sort_values('RMSE').reset_index(drop=True)
best = ranked.iloc[0]
a,b,c,d = st.columns(4)
a.metric('Records used', f"{meta['rows']:,}")
b.metric('Districts', len(meta['districts']))
c.metric('Training years', f"{meta['year_min']}–{meta['year_max']}")
d.metric('Models compared', len(models))
st.write('')
tab1,tab2,tab3 = st.tabs(['🌱 Production estimator','📊 Model performance','📚 Data & methodology'])

def style_chart(fig):
    fig.update_layout(template='plotly_white', font=dict(color='#274438'),
                      margin=dict(l=15,r=15,t=35,b=20), paper_bgcolor='rgba(0,0,0,0)',
                      legend=dict(orientation='h',y=-.2,title_text=''))
    return fig

with tab1:
    st.subheader('Build your production scenario')
    st.caption('Choose a location and season, then enter the cultivated area to estimate production.')
    inputs,output = st.columns([1.1,1],gap='large')
    with inputs:
        with st.form('production_scenario'):
            st.markdown('#### Location & season')
            a,b = st.columns(2)
            district = a.selectbox('District',meta['districts'])
            season = b.selectbox('Season',meta['seasons'])
            st.markdown('#### Cultivation details')
            a,b = st.columns(2)
            year = a.number_input('Year',min_value=int(meta['year_min']),max_value=2100,
                                  value=min(2100,max(int(meta['year_min']),date.today().year)),step=1)
            extent = b.number_input('Sown extent (ha)',min_value=0.,value=50000.,step=1000.,
                                    format='%.1f',help='Area sown in hectares; 1 ha = 10,000 m².')
            selected_model = st.selectbox('Prediction model',list(models),
                index=list(models).index(best['Model']),help='Defaults to the model with the lowest test RMSE.')
            st.caption(f"Training data covers {meta['year_min']}–{meta['year_max']}. Later years are unvalidated extrapolations.")
            submitted = st.form_submit_button('Estimate production →',type='primary',use_container_width=True)
        if submitted:
            if extent <= 0:
                st.error('Enter a sown extent greater than zero to create a scenario.')
            else:
                row = pd.DataFrame([dict(District=district,Season=season,Year=int(year),Sown_Extent_Hectares=float(extent))])
                with st.spinner('Estimating production…'):
                    value = float(models[selected_model].predict(row)[0])
                st.session_state['production_result'] = dict(production_thousand_metric_tons=value,
                    district=district,season=season,year=int(year),extent_hectares=float(extent),model=selected_model)
    with output:
        with st.container(border=True):
            st.markdown('#### Your production estimate')
            result = st.session_state.get('production_result')
            if result is None:
                st.info('Your estimate will appear here after you submit a scenario.')
                st.markdown('**Three steps to get started**')
                st.markdown('1. Choose a district and season.\n2. Enter the year and sown extent.\n3. Select a model and estimate production.')
            else:
                st.caption('Last submitted scenario')
                value = result['production_thousand_metric_tons']
                st.metric('Production · thousand metric tons',f'{value:,.2f}')
                st.caption(f'Approximately {value * 1000:,.0f} metric tons')
                st.divider()
                st.write(f"**{result['district']} · {result['season']} {result['year']}**")
                st.write(f"Sown extent: **{result['extent_hectares']:,.1f} ha**")
                st.write(f"Model: **{result['model']}**")
                if result['year'] > meta['year_max']:
                    st.warning('This year is beyond the training period. Future-year accuracy has not been validated.')
                if value < 0:
                    st.warning('The estimate is negative and not physically meaningful. Review the scenario inputs.')
                st.download_button('Download estimate (CSV)',pd.DataFrame([result]).to_csv(index=False),
                    file_name='paddy_production_estimate.csv',mime='text/csv',use_container_width=True)
        st.caption('Educational model estimate, not an official production forecast. Results remain visible until you submit another scenario.')

with tab2:
    st.subheader('Compare model performance')
    st.caption('All models use the same random 20% holdout. Error units: thousand metric tons.')
    a,b,c = st.columns(3)
    a.metric('Lowest RMSE model',str(best['Model']))
    b.metric('Best RMSE ↓',f"{best['RMSE']:.2f}")
    c.metric('Best model R² ↑',f"{best['R2']:.3f}")
    st.write('')
    left,right = st.columns(2,gap='large')
    with left:
        st.markdown('#### Prediction errors')
        errors = ranked.melt(id_vars='Model',value_vars=['RMSE','MAE'],var_name='Metric',value_name='Error')
        fig = px.bar(errors,x='Model',y='Error',color='Metric',barmode='group',height=390,
                     color_discrete_map={'RMSE':GREEN,'MAE':'#bba34f'},labels={'Error':'Error (thousand metric tons)'})
        st.plotly_chart(style_chart(fig),use_container_width=True)
        st.caption('Lower RMSE and MAE indicate smaller prediction errors.')
    with right:
        st.markdown('#### Actual vs. predicted')
        chart_model = st.selectbox('Inspect a model',ranked['Model'].tolist())
        plot_df = pd.DataFrame({'Actual':test_predictions['Actual'],'Predicted':test_predictions[chart_model]})
        plot_df['Absolute error'] = (plot_df['Actual']-plot_df['Predicted']).abs()
        fig = px.scatter(plot_df,x='Actual',y='Predicted',hover_data={'Absolute error':':.2f'},
                         color_discrete_sequence=[GREEN],opacity=.65,height=330,
                         labels={'Actual':'Actual (thousand metric tons)','Predicted':'Predicted (thousand metric tons)'})
        low = min(0,plot_df[['Actual','Predicted']].min().min())
        high = max(1,plot_df[['Actual','Predicted']].max().max()*1.05)
        fig.add_shape(type='line',x0=low,y0=low,x1=high,y1=high,line=dict(color='#9aab9f',dash='dash'))
        fig.update_xaxes(range=[low,high])
        fig.update_yaxes(range=[low,high])
        st.plotly_chart(style_chart(fig),use_container_width=True)
        st.caption('Points closer to the dashed line have more accurate predictions.')
    with st.expander('View all evaluation scores',expanded=True):
        st.dataframe(ranked.style.format({'R2':'{:.4f}','MAE':'{:.2f}','RMSE':'{:.2f}'}),hide_index=True,use_container_width=True)
        st.caption('R²: higher is better. MAE and RMSE: lower is better. Ranked by RMSE.')
        st.download_button('Download model scores',ranked.to_csv(index=False),file_name='model_scores.csv',mime='text/csv')
    st.info('Evaluation uses a random 80/20 split, not a chronological split. These scores do not establish performance on future seasons.')

with tab3:
    st.subheader('Understand the data behind the estimate')
    left,right = st.columns([1.2,1],gap='large')
    with left:
        with st.container(border=True):
            st.markdown('#### Source & coverage')
            st.markdown('[Department of Census and Statistics, Sri Lanka · Paddy Statistics](https://www.statistics.gov.lk/Agriculture/StaticalInformation/PaddyStatistics)')
            st.write('Source spreadsheets were consolidated by district, year and Maha/Yala season.')
            st.caption(f"{meta['original_rows']:,} original records · {meta['rows']:,} records used · {meta['year_min']}–{meta['year_max']}")
            st.markdown('**Inputs:** District, season, year and sown extent (hectares).')
            st.markdown('**Target:** Production (thousand metric tons).')
    with right:
        with st.container(border=True):
            st.markdown('#### Evaluation method')
            st.write(f"Split: {meta['split']}. All five models use the same split.")
            st.write('Random Forest, XGBoost, LightGBM, CatBoost and an RF + ANN voting ensemble. The ensemble averages two regression predictions.')
    with st.expander('Model inputs & limitations'):
        st.write('Harvested extent and average yield are excluded because they are closely tied to production and may be unknown when requesting a prediction.')
        st.write('This is an exploratory comparison. Rainfall, irrigation, fertilizer use and other influential factors are not included.')
    st.markdown('#### Explore source records')
    a,b = st.columns(2)
    filter_district = a.selectbox('Filter by district',['All districts',*meta['districts']])
    filter_season = b.selectbox('Filter by season',['All seasons',*meta['seasons']])
    filtered = data
    if filter_district != 'All districts':
        filtered = filtered[filtered['District'] == filter_district]
    if filter_season != 'All seasons':
        filtered = filtered[filtered['Season'] == filter_season]
    st.caption(f'{len(filtered):,} matching records · Preview shows up to 100 records.')
    st.dataframe(filtered.head(100),hide_index=True,use_container_width=True)
    st.download_button('Download filtered data',filtered.to_csv(index=False),file_name='paddy_source_records.csv',mime='text/csv')

st.divider()
st.caption('🌾 Paddy Insights · Applied Machine Learning · Sri Lanka')
