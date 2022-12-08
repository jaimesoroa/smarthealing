from st_on_hover_tabs import on_hover_tabs
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import requests
import time as t
from bs4 import BeautifulSoup as bs
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
from PIL import Image
import plotly.express as px

base = 'https://en.wikipedia.org/'
icd9_wiki = 'https://en.wikipedia.org/wiki/List_of_ICD-9_codes'
r = requests.get(icd9_wiki)
links = bs(r.content, 'html.parser').find('div', class_ = 'mw-parser-output')

# Scraping the diseases
def code_name(code):
    if code[0] in ['E', 'V', 'M']:
        return 'Sorry, no description.'
    elif code[0].isnumeric() == False:
        return 'Code does not exist.'
    else:
        # Get category to access specific link.
        i = None
        block_ends = [140,240,280,290,320,390,460,520,580,630,680,710,740,760,780,800,1000]
        blocks = [[i+1, s] for i, s in enumerate(block_ends)]
        for block in blocks:
            if float(code) < block[1]:
                i = block[0]
                break
        # Get url of icd9 specific page.
        try:
            link = links.ul.find_all('a')[i-1].get('href')
        except:
            return 'Code does not exist.'
        # Search for name of code.
        url = base + link
        r2 = requests.get(url)
        d = bs(r2.content, 'html.parser').find('div', class_ = 'mw-parser-output')
        found = False
        for a in d.find_all('a'):
            if found:
                return a.text
            if a.text == code:
                found = True
    
# ===============================================================================================================
# Page config            
st.set_page_config(
    page_title="Smart Healing",
    page_icon='⚕️',
    layout="wide",
    initial_sidebar_state="auto",
)

if "params" not in st.session_state:
    st.session_state['params'] = dict()

st.markdown('<style>' + open('smarthealing_app/website/style.css').read() + '</style>', unsafe_allow_html=True)

with st.sidebar:
        tabs = on_hover_tabs(tabName=['Dashboard', 'Prediction', 'Results'], 
                             iconName=['dashboard', 'money', 'economy'],
                             key="0")

# ===============================================================================================================
# Tab Dashboard
if tabs =='Dashboard':
    image = Image.open('smarthealing_app/website/Pngsmarthealing.png')
    c1, c2,  = st.columns([3, 1.5], gap='medium')
    with c1:
        st.title("Dashboard")
    with c2:
        st.image(image, caption=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")
    st.markdown("""---""")
    
    st.markdown(""" 
                The dataset consist of the information on leaves of more than 250K employees of many different companies
                """)
    selection = ['count','edad_mean', 'edad_median', 'duration_mean', 'duration_median']
    list_options = ['1 - Number of Leaves','2 - edad_mean', 'edad_median', 'duration_mean', 'duration_median']
    page = st.selectbox('Select graph: ', selection)
    # Display the country content here
    page = selection[page[0]]
    st.header(f"In the top 10 most common causes of leaves, here is the relationship between {page.capitalize()} and the ICD9 code (cause of leave)")
    fun = pd.read_csv('smarthealing_app/website/fun_stuff.csv')
    more_common = fun.head(10)
    more_common = round(more_common,2)

    fig = px.bar(
        data_frame=more_common, 
        x='delete', 
        y=page, 
        custom_data=['count', 'icd9','edad_mean', 'edad_median', 'duration_mean', 'duration_median', 'description']
    )

    fig.update_traces(
        hovertemplate="<br>".join([
            "Count: %{customdata[0]}",
            "icd9: %{customdata[1]}",
            "Age (mean): %{customdata[2]}",
            "Age (median): %{customdata[3]}",
            "Duration (mean): %{customdata[4]}",
            "Duration (median): %{customdata[5]}",
            "Description: %{customdata[6]}"
        ])
    )
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3  = st.columns(3)
    c1, c2, c3  = st.columns(3)
    c1.write(' ')
    c2.caption('ICD9 codes, descriptions and amount of leaves')
    c3.write(' ')

# ================================================================================================================
# Tab where we input the data
elif tabs == 'Prediction':
    image = Image.open('smarthealing_app/website/Pngsmarthealing.png')
    c1, c2,  = st.columns([3, 1.5], gap='medium')
    with c1:
        st.title("Predicted Medical Leaves")
    with c2:
        st.image(image, caption=None, use_column_width=True, clamp=True, channels="RGB", output_format="auto")
        #displaying the image on streamlit app
    st.markdown("""---""")
    st.markdown("""
                #### Please, fill this form to get a prediction of how many days should the employee take of leave?
                """)
    # 1st Line
    # CNAE category
    cnae_df = pd.read_csv('smarthealing_app/website/data/cnae_list.csv', sep = '.-', dtype='string', engine='python') 
    cnae_select = st.selectbox('CNAE Category:', (cnae_df['Code'] + ' - ' + cnae_df['Description']).tolist())
    cnae = cnae_select[:4]

    # 2nd Line
    c1, c2 = st.columns(2)   
    with c1:
        contract_df = pd.read_csv('smarthealing_app/website/data/contract_list.csv', sep = ',', dtype='string') 
        contract = st.selectbox('Contract Type:', (contract_df['clave'] + ' - ' + contract_df['denominacion']).tolist())
        # Selecting just description from DF, idk why it wasn't working the simple way
        contract = contract[:3]  
    with c2:
        # ICD9 category
        valid_characters = ['M','V','E','m','b','e','1','2','3','4','5','6','7','8','9','0','.']
        idc9_select = st.text_input('ICD9 code: ', value=460)
        matched_list = [characters in valid_characters for characters in idc9_select]
        if all(matched_list): 
            icd9 = f"{idc9_select} - {code_name(idc9_select)}"
            st.write('You selected: ', icd9)
        else:
            st.warning('Please input a valid string..', icon="⚠️")
    
    # 3rd Line
    c1, c2, c3 = st.columns(3)
    with c1:
        # Day counter
        day_counter = st.number_input('Acumulated days of previous leave: ',min_value=0, max_value=5000000)
        
    with c2:
        # Number of workers 
        num_workers = st.number_input('Number of workers in the company: ',min_value=0, max_value=5000000)
    with c3:
        # Codigo Postal
        postal_df = pd.read_csv('smarthealing_app/website/data/postal_list.csv', sep = ',', dtype='string') 
        
        postal = st.selectbox('Postal Code', (postal_df['codigopostalid'] + ' - ' + postal_df['provincia'] + ' - ' + postal_df['poblacion']).tolist())
        # Selecting just description from DF, idk why it wasn't working the simple way
        postal = postal.split(' ')[0]
       
    c1, c2 = st.columns(2)
    with c1: 
        # Relapse, setback
        setback_select = st.radio('Is this a setback / relapse?', ('Yes', 'No'))
        if setback_select == 'Yes':
            setback = 1
        else:
            setback = 0
    with c2:
        # Multiple jobs
        multi_job_select = st.radio("The employee have multiple jobs?", ('Yes', 'No'))
        if multi_job_select == 'Yes':
            multi_job = 1
        else:
            multi_job = 0
    st.markdown("""---""")
    
    c1, c2 = st.columns(2)
    # Type of contract
    with c1:
          # Day of the week 
        options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_week = st.select_slider('Select day of the week the leave started: ',
            options=options)
        week_day = options.index(day_week) + 1 
    with c2:
        # Cotization group 
        coti_df = pd.read_csv('smarthealing_app/website/data/coti_list.csv', sep = ':', dtype='string') 
        contribution = st.selectbox('Select tax contribution group:',
            options=(coti_df['description'].to_list()))
        # Selecting just description from DF, idk why it wasn't working the simple way
        selection_coti = coti_df[coti_df['description']==contribution]['category'].iloc[0]
    
    # 2nd Line
    c1, c2, c3 = st.columns(3)
    # Years in the company
    with c1:
        d0 = st.date_input(
        "Working in the company since: ", date(2021, 7, 6), min_value=date(1920, 7, 6))
        d1 = date.today()
        d3 = d1 - d0
        result_year = round((d3.days/365),3)
    with c2:
        # How old is the worker
        a0 = st.date_input(
        "When was the employee born? ", date(2004, 7, 6), min_value=date(1910, 7, 6))
        a1 = date.today()
        a3 = a1 - a0
        age = round((a3.days/365),3)
    with c3:
        # Week
        week = st.number_input('Week of the year: ', min_value=0, max_value=5000000)
    st.markdown("""---""")
        
    if st.button(label = 'Get Prediction'):
        my_bar = st.progress(0)
        for percent_complete in range(100):
            t.sleep(0.008)
            my_bar.progress(percent_complete + 1)
        st.write('Data saved! 💾')
        
        params = {
            "cnae": int(cnae),
            "icd9": str(idc9_select),
            "recaida": int(setback),
            "numtreb": int(num_workers),
            "codipostal": int(postal),
            "ContadordiasBajasDNI": int(day_counter),
            "contracte": str(contract),
            "grupcoti": str(selection_coti),
            "pluriempleo": int(multi_job),
            "diasemana": int(week_day),
            "tiempo_en_empresa": float(result_year),
            "edad": float(age),
            "epiweek": int(week)
        }
        st.session_state["params"] = params.copy()
        # st.write(params)
        # API CALL
        
        url = "https://smarthealing-w5jxjldzkq-ew.a.run.app/predict"

        with st.spinner('Fetching your prediction...'):
            prediction = requests.get(url, params)
            regress_baja = prediction.json().get('regression_leave_duration')
            class_baja = prediction.json().get('classifier_leave_duration')
            if(class_baja == 0):
                st.success(f"### Predicted: *Short Leave (Shorter than 15 days)*")
            else:
                st.success(f"### Predicted duration: *Long Leave (Longer than 15 days)*")
            st.success(f"### Estimated duration: *Around {regress_baja} days*")
    
# ================================================================
elif tabs == 'Results':
    image = Image.open('smarthealing_app/website/Pngsmarthealing.png')
    c1, c2  = st.columns([3, 1.5], gap='medium')
    with c1:
        st.title("Results")
    with c2:
        st.image(image, caption=None, use_column_width=True, clamp=True, channels="RGB", output_format="auto")
    st.markdown("""---""")
    
    if len(st.session_state["params"]) > 0:
        params = st.session_state["params"]
        st.header(f"Given the parameters and prediction...")
        fun = pd.read_csv('smarthealing_app/website/fun_stuff.csv')
    
        more_common = fun.head(10)
        more_common = round(more_common, 2)

        col1, col2, col3 = st.columns(3)
        col1.metric("ICD9:", params['icd9'] , "1.2 °F")
        col2.metric("Age", params['edad'], "-8%")
        col3.metric("Day of the Week", params['epiweek'], "4%")

    else:
        st.write("No parameters yet, go to the prediction tab and input the data ⚠️")
        #displaying the image on streamlit app
        url_lottie = "https://assets4.lottiefiles.com/packages/lf20_4owMZE.json"
        def load_lottieurl(url: str):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()

        c1, c2, c3  = st.columns([1,2,1])
        c1.write(' ')
        lottie_json = load_lottieurl(url_lottie)
        with c2:
            st_lottie(lottie_json)
            st_lottie_spinner(lottie_json)
        c3.write(' ')
    