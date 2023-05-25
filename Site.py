import subprocess
import sys
import streamlit as st
import pandas as pd
from Model import data, NBdata

def install(package):
    subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)

install('pandas-profiling')
install('streamlit-pandas-profiling')

import pandas_profiling
from streamlit_pandas_profiling import st_profile_report



st.title('Автоматическое определение токсичных сообщений')
st.write(
    'Программа была обучена на дата-сете "Russian Language Toxic Comments": https://www.kaggle.com/datasets/blackmoon/russian-language-toxic-comments')

st.dataframe(data)

st.write("""
Модель, которую мы подобрали для данного датасета показала высокую точность, если сравнивать с решениями других людей.
""")
cr_matrix = [[''] * 4,
             [0.94, 0.92, 0.93, 948],
             [0.85, 0.88, 0.87, 494],
             [''] * 4,
             ['', '', 0.91, 1442],
             [0.89, 0.90, 0.90, 1442],
             [0.91, 0.91, 0.91, 1442],
             ]
cr_matrix = pd.DataFrame(cr_matrix,
                         columns=['precision', 'recall', 'f1-score', 'support'],
                         index=['', 'neutral', 'toxic', '', 'accuracy', 'macro avg', 'weighted avg'],
                         )
st.table(cr_matrix)


@st.cache_resource(experimental_allow_widgets=True)
def More_Info():
    profiled_data = data.profile_report()
    with st.expander("More info"):
        st_profile_report(profiled_data, height=500)


More_Info()

st.header('Попробовать написать комментарий и проверить его на токсичность')
with st.form('new_comment'):
    phrase = st.text_area("Введите комментарий:",
                          height=150,
                          )
    phrases = phrase.split("\n")
    submitted = st.form_submit_button("Предсказать токсичность")

if phrases[0] != '':
    pred = NBdata.predict(phrases)
    if len(pred) == 1:
        st.write('Модель, обученная на дата-сете, считает, что Ваш отзыв')
    else:
        st.write('Модель, обученная на дата-сете, считает, что Ваши отзывы в том же порядке:')

    for one in pred:
        if one == 'neutral':
            st.write(':green[не токсичный]')
        elif one == 'toxic':
            st.write(':red[токсичный]')
else:
    st.write('Напишите одно или несколько предложений с новой строки, чтобы проверить их на токсичность.')
