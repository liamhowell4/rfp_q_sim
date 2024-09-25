import os
from pinecone import Pinecone
import pandas as pd
import tiktoken
from openai import OpenAI
from tqdm.autonotebook import tqdm
import re

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

client = OpenAI()

# embedding model parameters
embedding_model = "text-embedding-3-large"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

encoding = tiktoken.get_encoding(embedding_encoding)

#turns off SettingWithoutCopy warning
pd.options.mode.chained_assignment = None

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
index = pc.Index('rfp-response-qs')

filter_cats = ['rfp', 'general', 'project', 'iso/region', 'state', 'tech']
filter_cat_dict = {'Project Name': 'project', 'ISO/Region': 'iso/region', 'State': 'state', 'Technology': 'tech', 'Include General (not project/RFP specific) questions': 'general', 'Specific RFP': 'rfp'}

q_a_template_line1 = [['What is the project name?', 'Las Camas Solar Park', 'cpa_rfo_2024', 'Las Camas', 'Solar', True, 'CA', 'CAISO', '7', '2024']]
new_q_a_template = pd.DataFrame(q_a_template_line1, columns=['question', 'answer', 'rfp', 'project', 'tech', 'general', 'state', 'iso/region', 'month', 'year'])


def get_embedding(text, engine=embedding_model):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=engine).data[0].embedding

#takes in a dataframe with at least an 'id' and 'text' column
def parallel_embedding_for_df(chunks_df, embed_function):

    section_data = list(zip(chunks_df['id'], chunks_df['question']))

    section_data_with_embeddings = []

    with ThreadPoolExecutor() as executor:
        
        future_to_section = {executor.submit(embed_function, section[1]): section for section in section_data}

        for future in concurrent.futures.as_completed(future_to_section):
            section = future_to_section[future]

            try:
                embedding = future.result()
                section_data_with_embeddings.append([section[0], embedding])
            except Exception as exc:
                print(f"{section[0]} generated an exception: {exc}")
            
    return pd.DataFrame(section_data_with_embeddings, columns=['id', 'embedding'])

def prepare_df_for_pinecone(df, id_column='id', embedding_column='embedding'):
        return [{'id': row[id_column], 'values': row[embedding_column], 'metadata': {f'{column}': row[column] for column in df.columns if column not in [
                id_column, embedding_column]}} for _, row in df.iterrows()]

def cut_up_list(input_list, chunk_size):
        """Yield successive chunks of chunk_size from input_list."""
        for i in range(0, len(input_list), chunk_size):
            yield input_list[i:i + chunk_size]

# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def check_new_csv(csv):
    assert list(csv.columns) == [
        'question', 'answer', 'rfp', 'project', 'tech', 'general', 'state', 'iso/region', 'month', 'year'
    ], 'Columns must be question, answer, rfp, project, tech, general, state, iso/region, in that order'

# function that takes in a new csv of questions, answers, and metadata and uploads it
def upload_new_qs(new_q_a_csv, progress_element=None, progress_function=None):

    if progress_element:
        progress_function(f"Processing {len(new_q_a_csv)} questions and answers", progress_element)

    new_q_a_csv = new_q_a_csv.dropna(subset=['question', 'answer']).reset_index(drop=True)

    new_q_a_csv['counter'] = new_q_a_csv.groupby('rfp').cumcount()

    # Add the 'id' column by combining 'rfp' and 'counter'
    new_q_a_csv['id'] = new_q_a_csv['rfp'] + '_' + new_q_a_csv['counter'].astype(str)

    # Drop the intermediate 'counter' column if not needed
    new_q_a_csv = new_q_a_csv.drop('counter', axis=1)

    ids_embeddings = parallel_embedding_for_df(new_q_a_csv, get_embedding)

    new_q_a_csv = new_q_a_csv.merge(ids_embeddings, on='id', how='inner')

    q_a_bank_pc = prepare_df_for_pinecone(new_q_a_csv)

    for sub_list in tqdm(list(cut_up_list(q_a_bank_pc, 30))):
        index.upsert(vectors=sub_list, namespace='rfp-response', async_req=True)

    if progress_element:
        progress_function(f"Success! All {len(new_q_a_csv)} questions and answers uploaded", progress_element)

def delete_rfp_from_pinecone_namespace(rfp_name, index=index, namespace='rfp-response'):
    index.delete(list(index.list(prefix=rfp_name, namespace=namespace))[0], namespace=namespace)

# Get the top k similar, already answered questions for a question based on a set of filters
def get_suggestions(question, index=index, k=5, filter_dict = {}, print_filter_list=True, include_metadata=False):

    question_embedding = get_embedding(question)

    filter_dict = {key: value if key in filter_cats else print(f'{key} is not filterable')for key, value in filter_dict.items()}

    if print_filter_list:
        print("Filter Dict: ", filter_dict)
        
    results = index.query(vector=question_embedding, top_k=k, namespace='rfp-response', include_metadata=True, filter=filter_dict)

    if not include_metadata:
        q_a = [[match['metadata']['question'], match['metadata']['answer'], match['metadata']['rfp']] for match in results['matches']]
    else:
        q_a = [[match['metadata']['question'], match['metadata']['answer'], match['metadata']['rfp'], 
                match['metadata']['general'], match['metadata']['project'], match['metadata']['isoregion'],
                match['metadata']['tech']] for match in results['matches']]
        

    if not len(q_a):
        return pd.DataFrame(columns=['question', 'answer', 'rfp_name'])

    elif q_a[0][0] == question and ('project' in filter_dict.keys() or filter_dict['general'] == True):

        q_a = [q_a[0]] + [['', ' ', ' ']] * 4

    return pd.DataFrame(q_a, columns=['question', 'answer', 'rfp_name'])

# function that takes in a dataframe of questions and runs the get_suggestions function on each question. Returns a dataframe of the top k suggestions for each q
# If the top suggestions is the question itself, it will return just that top suggestion
def run_similarities_on_q_df(q_df, filter_dict={}, progress_element=None, progress_function=None, answers_element=None):

    if progress_element:
        progress_function(f"Processing {len(q_df)} questions", progress_element)

    suggestions_list = []

    q_df = q_df[q_df['question'] != '']

    for question in q_df['question']:

        suggestion_row = {'question_actual': question}

        suggestions = [{f'{inner_key}_{outer_key}': inner_value for inner_key, inner_value in outer_value.items()} for outer_key, outer_value in 
                    get_suggestions(question, filter_dict = filter_dict).to_dict(orient='index').items()]

        for sugggestion in suggestions:
            suggestion_row.update(sugggestion)

        suggestions_list.append(suggestion_row)
    
    if progress_element:
        progress_function(f"Success, processed {len(q_df)} questions", progress_element)

    return pd.DataFrame(suggestions_list)

def to_snake_case(text):
    # Replace all non-alphanumeric characters with a space
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    # Replace uppercase letters with lowercase and precede them with an underscore if they are not at the start
    text = re.sub(r'(?<!^)(?=[A-Z])', '_', text).lower()
    # Replace spaces and underscores with a single underscore
    text = re.sub(r'[\s_]+', '_', text)
    return text