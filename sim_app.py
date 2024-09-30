import streamlit as st
import rfp_q_sim as rfp
import pandas as pd
from pinecone import Pinecone
import os

st.title('RFP Suggestions Page')


def get_all_metadata_filters(index):

    pinecone_metadata = [item.metadata for item in index.query(top_k=10000, vector=rfp.get_embedding("hello world"), namespace="rfp-response", include_metadata=True)['matches']]
    # pinecone_metadata = {key: value for key, value in pinecone_metadata.items() if key not in ['question', 'answer']}
    return {key: set([item[key] for item in pinecone_metadata]) for key in pinecone_metadata[0].keys() if key not in ['question', 'answer']}

# Function that runs the progress message element
def update_progress(message='working!', progress_element=None):

    state = 'running'
    if 'Success' in message or 'complete' in message or 'already' in message:
        state = 'complete'
        if 'Summary' in message:
            st.session_state.answer_is_summary = True
    elif 'could not' in message:
        state = 'error'

    if progress_element==None:
        st.status(f"Progress: {message}", state=state)
    else:
        progress_element.status(f"Progress: {message}", state=state)

def run_similarities_on_q_df(q_df, filter_dict={}, progress_element=None, progress_function=None):

    q_df = q_df.fillna("")

    st.session_state.finished_qs = rfp.run_similarities_on_q_df(q_df, filter_dict=filter_dict, progress_element=progress_element, progress_function=progress_function)

if 'filter_dict' not in st.session_state:
    st.session_state.pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    st.session_state.index = st.session_state.pc.Index('rfp-response-qs')
    st.session_state.metadata_filter_values = get_all_metadata_filters(st.session_state.index)
    st.session_state.metadata_filters = st.session_state.metadata_filter_values.keys()

    st.session_state.qs_to_check = 5

    st.session_state.list_qs_ready=False

    st.session_state.filter_dict = dict()
    st.session_state.finished_qs = None

st.sidebar.subheader('Filters: ')

for key, value in rfp.filter_cat_dict.items():
    filter_val = st.sidebar.selectbox(label=key, options=[''] + list(st.session_state.metadata_filter_values[value]))
    if filter_val != '':
        st.session_state.filter_dict[rfp.filter_cat_dict[key]] = filter_val

single_question, full_q_df, upload_new_q_as = st.tabs(['Single Question', 'List of Questions', 'Upload Answered Questions'])

with single_question:
    st.write('Enter a question to get suggestions for')
    question = st.text_input('Question')


    if question:
        # st.session_state.filter_dict = {key: value for key, value in st.session_state.filter_dict.items() if value != ''}

        st.write(st.session_state.filter_dict)

        suggestions = rfp.get_suggestions(question, filter_dict=st.session_state.filter_dict)
        st.write(suggestions)


with full_q_df:
    
    st.session_state.qs_to_check = st.number_input('Number of new questions to check', min_value=1, value=20)

    qs_to_check = st.data_editor(pd.DataFrame([''] * st.session_state.qs_to_check, columns=['question']), width=1000)

    progress_element = st.empty()

    st.write('Filtered by: ', st.session_state.filter_dict)

    st.button(
        'Get Suggestions', on_click=run_similarities_on_q_df, args=(qs_to_check, st.session_state.filter_dict, progress_element, update_progress)
        )

    if st.session_state.finished_qs is not None:


        st.dataframe(st.session_state.finished_qs)

    

with upload_new_q_as:

    st.write('Upload a csv of questions and answers to add to the database')
    st.write("The csv should have the following columns: [question, answer, rfp, project, tech, general, state, iso/region, month, year]")
    multi= '''<ul>These columns represent:  
                <li>the question;</li>
                <li>the answer;</li>
                <li>the rfp name;</li>
                <li>the project name;</li>
                <li>the technology (BESS, Solar, or Wind) <b>[case sensitive];</b></li>
                <li>whether the question is general (questions about EDPR more generally, not just this RFP or this Offtaker);</li>
                <li>the state;</li>
                <li>the iso/region;</li>
                <li>the month of the RFP due date;</li>
                <li>and the year of the RFP due date.</li>
            </ul>
             '''
    st.markdown(multi, unsafe_allow_html=True)


    st.download_button('Download Template', data=rfp.new_q_a_template.to_csv(index=False).encode('utf-8'), file_name='new_q_a_TEMPLATE.csv')

    uploaded_file = st.file_uploader("Choose a file", type=['csv'])

    upload_progress= st.empty()

    if uploaded_file:
        new_q_a_csv = pd.read_csv(uploaded_file)

        try:
            rfp.check_new_csv(new_q_a_csv)

            rfp.upload_new_qs(new_q_a_csv, upload_progress, update_progress)
        except AssertionError as e:
            st.error(e)