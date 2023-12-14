import streamlit as st
import requests
import pandas as pd
from pandas import json_normalize
from datetime import datetime, timedelta
from dateutil import parser as date_parser
import calendar
import pytz
from math import radians, cos, sin, asin, sqrt
import time
import json

from google.oauth2 import service_account

from langchain.chat_models import AzureChatOpenAI, ChatVertexAI
from langchain.llms import VertexAI

from langchain import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import ConversationChain, LLMChain

from pydantic import BaseModel, Field
from typing import List

## Streamlit related functions ##
def get_custom_css_modifier():
    css_modifier = """
<style>
/* remove Streamlit default menu and footer */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}
</style>
    """
    return css_modifier

## LLM related functions ##
#@st.cache_resource 
def get_llm_api_config_details():
    llm_choice = st.session_state.llm_choice

    if llm_choice == 'OpenAI':

        config = {
            'api_type' : st.secrets['api_type'],
            'api_base' : st.secrets['api_base'],
            'api_version' : st.secrets['api_version'],
            'api_key' : st.secrets['api_key']
        }
        
        return config
    
    else:
        google_api_cred = service_account.Credentials.from_service_account_info(
            info={
                "type": st.secrets['type'] ,
                "project_id": st.secrets['project_id'] ,
                "private_key_id": st.secrets['private_key_id'] ,
                "private_key": st.secrets['private_key'] ,
                "client_email": st.secrets['client_email'] ,
                "client_id": st.secrets['client_id'] ,
                "auth_uri": st.secrets['auth_uri'] ,
                "token_uri": st.secrets['token_uri'] ,
                "auth_provider_x509_cert_url": st.secrets['auth_provider_x509_cert_url'] ,
                "client_x509_cert_url": st.secrets['client_x509_cert_url'] ,
                "universe_domain": st.secrets['universe_domain'] 
            },
        )

        return google_api_cred

#@st.cache_resource 
def get_llm_instance(model='gpt-4'):   
    llm_choice = st.session_state.llm_choice
    config = get_llm_api_config_details()

    if llm_choice == 'OpenAI':
        #model = 'gpt-4' | 'gpt-4-32k' | 'gpt-35-turbo'
        llm = AzureChatOpenAI(
            deployment_name=model, 
            model_name=model, 
            openai_api_key = config['api_key'],
            temperature=0, 
            max_tokens=5000,
            n=1, 
            openai_api_base = config['api_base'],
            openai_api_version = config['api_version'],
        )

    elif llm_choice == 'Vertex-chat':
        
        llm = ChatVertexAI(
            model_name="chat-bison",
            max_output_tokens=1024,
            temperature=0,
            top_p=0.8,
            top_k=40,
            verbose=True,
            credentials = config,
            project=config.project_id,
            #location='us-central1'
        )

    else:
        
        llm = VertexAI(
            model_name="text-bison",
            max_output_tokens=1024,
            temperature=0,
            top_p=0.8,
            top_k=40,
            verbose=True,
            credentials = config,
            project=config.project_id,
            #location='us-central1'
        )

    return llm

def get_completion(string_prompt_value):
    llm_choice = st.session_state.llm_choice
    llm = get_llm_instance()

    if llm_choice in ['OpenAI', 'Vertex-chat']:
        final_prompt = string_prompt_value.to_messages()
        response = llm(final_prompt)
        return response.content

    else:
        final_prompt = string_prompt_value.to_string()
        response = llm(final_prompt)

        return response

## Get Simulated data functions ##
def getcoordinates(address):
    req = requests.get(f'https://www.onemap.gov.sg/api/common/elastic/search?searchVal={address}&returnGeom=Y&getAddrDetails=Y&pageNum=1')
    resultsdict = eval(req.text)
    if len(resultsdict['results'])>0:
        return resultsdict['results'][0]['LATITUDE'], resultsdict['results'][0]['LONGITUDE']
    else:
        pass

def getcoordinates_gmap(address):
    gmap_api_key = st.secrets['gmap_api_key']
    req = requests.get(f'https://maps.googleapis.com/maps/api/geocode/json?key={gmap_api_key}&address={address}, Singapore')
    resultsdict = json.loads(req.text)
    if len(resultsdict['results'])>0:
        return resultsdict['results'][0]['geometry']['location']['lat'], resultsdict['results'][0]['geometry']['location']['lng']
    else:
        pass

@st.cache_data
def get_facility_outlets():
    get_facility_outlets_url = 'https://www.onepa.gov.sg/pacesapi/FacilitySearch/GetFacilityOutlets'

    try:
        response = requests.get(get_facility_outlets_url)
        json_data = response.json()
    except Exception as e:
        json_data = json.load(open('getFacilityOutlets.json','r'))

    tmp_df = json_normalize(
        pd.DataFrame(
            json_data['data']['data']['outlets']
        ).explode('ccList', ignore_index=True)['ccList']
    ).explode('productList', ignore_index=True)

    replacement_address = {
        'Toa Payoh East Zone 2 RN': '310228', 
        'Toa Payoh East Zone 4 RN': '310258'
    }

    for l in replacement_address:
        coords = getcoordinates(replacement_address[l])

        tmp_df.loc[tmp_df['label'] == l, 'lat'] = coords[0]
        tmp_df.loc[tmp_df['label'] == l, 'lng'] = coords[1]

    tmp_df['lat'] = tmp_df['lat'].astype(float)
    tmp_df['lng'] = tmp_df['lng'].astype(float)

    facility_id_df = pd.concat([
        tmp_df[['label','lat','lng']],
        json_normalize(tmp_df['productList']),
    ], axis=1).drop_duplicates()

    return facility_id_df

@st.cache_data 
def get_simulated_data():
    df = pd.read_csv('facility_availability_2023_08_23.csv')
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y').dt.date
    df['price'] = df['price'].round(2)

    facility_outlet_df = get_facility_outlets()

    df = df.merge(facility_outlet_df, left_on='facility_id', right_on='id', how='left')
    df = df.rename(columns={'value': 'facility_type'})

    df = df.sort_values(by=['booking_url','startTime'])

    df = df.query("facility_type == 'BADMINTON COURTS'") #only focus on badminton courts for now
    
    #don't need this now, the simulated dataset has the booking URL.
    #df['booking_url'] = df.apply(lambda x: generate_pa_facility_booking_link(x['facility_id'], x['date']), axis=1) 

    return df


## Date formatter functions ##
def todayDate():
    today = datetime.now(pytz.timezone("Singapore"))
    return today.strftime('%d/%m/%Y')

# get day of week for a date (or 'today')
def dayOfWeek(date):
    today = datetime.now(pytz.timezone("Singapore"))
    if date == 'today':
        return calendar.day_name[today.weekday()]
    else:
        try:
            theDate = date_parser.parse(date)
        except:
            return 'invalid date format, please use format: dd/mm/yyyy'

        return calendar.day_name[theDate.weekday()]

def get_calendar_reference(today_str = None):
    
    if today_str:
        today = date_parser.parse(today_str, dayfirst=True)
    else:
        # Get today's date
        today = datetime.now(pytz.timezone("Singapore")).date()

    # Initialize the output string
    output = ""

    # Generate and append day of the week and date for the next 16 days
    for day in range(16):
        current_date = today + timedelta(days=day)
        day_of_week = current_date.strftime('%A')
        date_formatted = current_date.strftime('%d/%m/%Y')
        if current_date == today:
            date_formatted = f"{date_formatted} [TODAY]"
        output += f'- {day_of_week} {date_formatted}\n'

    return output

def get_valid_dates(today_str = None):
    if today_str:
        today = date_parser.parse(today_str, dayfirst=True)
    else:
        # Get today's date
        today = datetime.now(pytz.timezone("Singapore")).date()

    output = []

    # Generate and append day of the week and date for the next 16 days
    for day in range(16):
        current_date = today + timedelta(days=day)
        day_of_week = current_date.strftime('%A')
        date_formatted = current_date.strftime('%d/%m/%Y')
        output.append(date_formatted)

    return output

def get_interested_dates(date_str, is_simulation_mode = False):
    class InterestedDates(BaseModel):
        interested_dates: List[str] = Field(description='a list of dates in dd/mm/yyyy format that the client could be interested in when booking the facility. If this information is not found, output ["none"].')

    parser = PydanticOutputParser(pydantic_object=InterestedDates)
    format_instructions = parser.get_format_instructions()

    template = """
The calendar for the next 15 days are as deliminted by triple backticks:
```
{calendar_ref}
```

Today's date is {today_day_of_week} {today_date_str}.
Identify the closest date or dates `{date_str}` could be referring to from the calendar above in dd/mm/yyyy format. 
If there are no valid dates in the calendar above or if the date doesn't exist, do not make up an answer and just output ["none"].
{format_instructions}
    """

    if is_simulation_mode:
        df = get_simulated_data()
        simulated_date_range = sorted(df.date.unique())

        today_date_str = str(min(simulated_date_range))
        today_day_of_week = dayOfWeek(today_date_str)
        calendar_ref = get_calendar_reference(today_date_str)
        valid_dates_list = get_valid_dates(today_date_str)

    else:
        today_date_str = todayDate()
        today_day_of_week = dayOfWeek('today')
        calendar_ref = get_calendar_reference()
        valid_dates_list = get_valid_dates()

    prompt = PromptTemplate.from_template(
        template=template, 
    ).format_prompt(
        today_day_of_week = today_date_str,
        today_date_str = today_day_of_week,
        calendar_ref = calendar_ref,
        date_str = date_str,
        format_instructions = format_instructions
    )

    response = get_completion(prompt)

    output = parser.parse(response)
    output.interested_dates = [d for d in output.interested_dates if d in valid_dates_list]

    return output


## Get nearby facilities functions ##
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r


def get_nearby_facilities(user_interested_location, facility_type = 'BADMINTON COURTS'):
    '''
    1. geocode location
    2. filter "value" to facility of choice
    3. get top x distinct id based on closest haversine distance
    '''
    #To get rid of duplicated facility_id entries in full dataframe
    nearest_id_set = set()
    nearest_label_set = set()
    
    user_coords = getcoordinates(user_interested_location)
    
    if user_coords:
        facility_id_df = get_facility_outlets()

        facility_id_df['dist_from_user'] = facility_id_df[['lat','lng']].apply(
            lambda x: haversine(x['lat'], x['lng'], float(user_coords[0]),float(user_coords[1])),
            axis=1
        )

        sorted_df = facility_id_df[facility_id_df['value'] == facility_type].sort_values('dist_from_user')


        #currently fetching the top 3 closest facilities. Can consider getting all facilities within a radius

        for index, row in sorted_df.iterrows():
            nearest_id_set.add(row.id)
            nearest_label_set.add(row.label)
            if len(nearest_id_set) >=3:
                break
    

    # consider returning label too, since that contains the name of the CC
    print(nearest_id_set)
    return nearest_id_set, nearest_label_set


## Parsing user input functions ##
def understand_user_intent(user_input_str, chat_history):
    class UserInput(BaseModel):
        type_of_request: str = Field(description='the type of request that the client is making based on the latest message. This will be one of the following values: conversational_message, request_question')
        crux_message: str = Field(description="A summary sentence of the human's request.")
        humans_language: str = Field(description="Identified language the human is comfortable with. This will be one of the following values: English, Chinese, Malay, Tamil")
            
    parser = PydanticOutputParser(pydantic_object=UserInput)
    format_instructions = parser.get_format_instructions()
    
    template = """
You are reviewing a conversation between a Badminton Court booking assistant and a human. 
Analyze the <latest human input> to infer ultimate request the human is seeking. If information is unavailable, reference the <chat history> to understand the human's intent.

<Chat history>:
```
{chat_history}
<Latest human input>: {user_input}
```

Output the information in JSON format:

<type_of_request>: The type of request that the client is making making based on the latest message. This will be one of the following values:
- conversational_message. For example: Hi. Thank you! How are you? Bye bye. Happy birthday to you!
- request_question. For example: What can you do? Is there a badminton court available? What about next week? I'm looking for a badminton court. Yes please. Go ahead. 

<crux_message>: A summary sentence of the human's request, based on the latest human input and the chat history.

<humans_language>: Identify the language that the human is comfortable in based on the ongoing conversation. This will be one of the following values: English, Chinese, Malay, Tamil

{format_instructions}
    """
    
    prompt = PromptTemplate.from_template(
        template=template,
    ).format_prompt(
        format_instructions=format_instructions,
        chat_history=chat_history,
        user_input=user_input_str
    )

    response = get_completion(prompt)
    output = parser.parse(response)

    return output


def extract_search_parameters(user_input):
    class SearchParameters(BaseModel):
        locations: List[str] = Field(description='a list of all locations that the client might be interested in when booking the facility. If this information is not found, output ["none"].')
        dates: List[str] = Field(description='a list of all relative date expressions that the client might be interested in when booking the facility. If this information is not found, output ["none"].')
        #timings: List[str] = Field(description='a list of all relative timing expressions that the client might be interested in when booking the facility. If this information is not found, output ["none"].')
        #facility_type: List[str] = Field(description="a list of all facility types that the client might be interested in when booking. The only values allowed here are: 'BADMINTON COURTS', 'BBQ PIT', 'TABLE TENNIS ROOM','FUTSAL COURT', 'TENNIS COURT', 'BASKETBALL COURT','SEPAK TAKRAW COURT'. If this information is not found, output ['none'].")
            
    parser = PydanticOutputParser(pydantic_object=SearchParameters)
    format_instructions = parser.get_format_instructions()
    
    
    template = """
Extract the following information from <Human request> in JSON format. If the information is not found, just output ["none"]:
<locations>: List out all locations that are mentioned in the human's request. Examples of locations include: Hougang, Queenstown, Tampines, Serangoon, Bedok, East Coast Park, Marine Parade
<dates>: List out all relative date references that are mentioned in human's request. Examples of date references include: next week, this weekend, Tuesday, Fridays, 2023-08-15, 15th Aug, Sept 23, today, tomorrow, between Aug 1 and Aug 10, 1st Aug to 10th Aug, 2023-08-01 to 2023-08-10, Friday in September, Tuesdays in March

<Human request>: ```{user_input}```

{format_instructions}
    """
    
    #<timings>: For example - 12pm, noon, 7.30, 1930, afternoon, evening
    #<facility_type>: The only values allowed here are: 'BADMINTON COURTS', 'BBQ PIT', 'TABLE TENNIS ROOM','FUTSAL COURT', 'TENNIS COURT', 'BASKETBALL COURT','SEPAK TAKRAW COURT'

    prompt = PromptTemplate.from_template(
        template=template,
    ).format_prompt(
        format_instructions=format_instructions,
        user_input=user_input
    )
    
    response = get_completion(prompt)
    output = parser.parse(response)

    return output


## Get availability details ##
def get_simulated_availability_details(location_list = [], date_list = []):
    date_query = None
    facility_query = None
    combined_query = None
    df = get_simulated_data()
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%d/%m/%Y')
    
    if len(date_list) > 0:
        date_query = ' | '.join([f"date == '{date}'" for date in date_list])
    if len(location_list) > 0:
        facility_query = ' | '.join([f"facility_id == '{facility_id}'" for facility_id in location_list])
    
    if date_query and facility_query:
        combined_query = f"({date_query}) and ({facility_query})"
    elif date_query:
        combined_query = f"({date_query})"
    elif facility_query:
        combined_query = f"({facility_query})"

    if combined_query:
        filtered_df = df.query(combined_query)
    else:
        filtered_df = df
    
    pivot_df = summarize_availability_dataframe(filtered_df)

    return pivot_df


# method to take a dataframe at a per availability slot level and pivot into a per booking url level (outlet and date level)
def summarize_availability_dataframe(df):
    def pivot_agg_func(seq):
        seen = set()
        seen_add = seen.add

        # to de-dup overlap timings of multiple courts in a single date and outlet entry
        timings = [item for item in seq if not (item in seen or seen_add(item))]
        
        merged_timings = []
        start_time = None
        end_time = None

        #merge neighbouring timings together. E.g [2pm-3pm, 3pm-4pm] becomes [2pm-4pm]
        for timing in timings:
            timing_parts = timing.split(" - ")
            start, end = timing_parts[0], timing_parts[1]
        
            if start_time is None:
                start_time = datetime.strptime(start, "%I:%M %p")
                end_time = datetime.strptime(end, "%I:%M %p")
            else:
                next_start_time = datetime.strptime(start, "%I:%M %p")
                if next_start_time == end_time:
                    end_time = datetime.strptime(end, "%I:%M %p")
                else:
                    merged_timings.append(
                        f"{start_time.strftime('%I:%M %p')} - {end_time.strftime('%I:%M %p')}"
                    )
                    start_time = datetime.strptime(start, "%I:%M %p")
                    end_time = datetime.strptime(end, "%I:%M %p")
    
        if start_time and end_time:
            merged_timings.append(
                f"{start_time.strftime('%I:%M %p')} - {end_time.strftime('%I:%M %p')}"
            )

        return merged_timings

    pivot_df = df.pivot_table(
        index=['outlet_name','date', 'booking_url','facility_type'],
        values=['timeRangeName'],
        aggfunc=pivot_agg_func
    ).reset_index(drop=False)

    return pivot_df


## Format availibility output ##
def get_availability_response_from_summarized_dataframe(df, rephrase_user_request, cleaned_cc, cleaned_dates, humans_language = 'english'):

    if df is None or len(df) == 0:
        # might also want to check set(['outlet_name', 'date', 'booking_url','timeRangeName']).issubset(df.columns) == False
        return ''

    formatted_data = {}

    for _, row in df.iterrows():
        outlet_name = row['outlet_name']
        date = row['date']
        booking_url = row['booking_url']
        timings = row['timeRangeName']

        if outlet_name not in formatted_data:
            formatted_data[outlet_name] = {}

        if date not in formatted_data[outlet_name]:
            formatted_data[outlet_name][date] = {
                'booking_url': booking_url,
                'timings': timings
            }

    availability_options = ''
    for outlet_name in formatted_data:
        availability_options += f'- {outlet_name} \n'
        
        for date in formatted_data[outlet_name]:
            availability_options += f'  - {date} \n'
            availability_options += f'    - [Book Here]({formatted_data[outlet_name][date]["booking_url"]}) \n'
            availability_options += f'    - Time slots: {", ".join(formatted_data[outlet_name][date]["timings"])} \n'

    response = f"""
{rephrase_user_request} 

I've checked the availability for {', '.join(cleaned_cc)} on {', '.join(cleaned_dates)}. Here are the available slots I found:

{availability_options}

Please note that these slots are subject to availability and can be booked on a first-come, first-served basis. So, I recommend booking your preferred slot as soon as possible. Enjoy your game!
    """

    return translate_if_not_english(response,humans_language)

## Give user response ##
def rephrase_user_request(crux_message):
    class RephrasedRequest(BaseModel):
        rephrased_msg: str = Field(description='The rephrased request to demonstrate active listening.')
            
    parser = PydanticOutputParser(pydantic_object=RephrasedRequest)
    format_instructions = parser.get_format_instructions()

    template = """
Rephrase to demonstrate active listening (For example - "I've noted that you are asking for...", "Got it, you're interested in...", "I'm here to help you..."):
{user_request}

{format_instructions}
    """
    
    prompt = PromptTemplate.from_template(
        template=template,
    ).format_prompt(
        user_request=crux_message,
        format_instructions=format_instructions,
    )
    
    response = get_completion(prompt)
    output = parser.parse(response)

    return output.rephrased_msg

def construct_chat_history(session_state_chat_history):
    #chat_history = ''
    chat_history_buffer = ConversationBufferMemory(input_key='input')

    #skip the first intro greeting line by the ai assistant
    for line in session_state_chat_history[1:]:
        if line['role'] == 'user':
            #chat_history = chat_history + f"<Human>: {line['content']}\n"
            chat_history_buffer.chat_memory.add_user_message(line['content'])
        if line['role'] == 'assistant':
            #chat_history = chat_history + f"<Assistant>: {line['content']}\n"
            chat_history_buffer.chat_memory.add_ai_message(line['content'])

    return chat_history_buffer


def translate_if_not_english(msg,humans_language='english'):

    if humans_language.lower() == 'english':
        return msg

    template = """
Translate the following to {humans_language}:

{msg}
"""
    prompt = PromptTemplate.from_template(
        template=template,
    ).format_prompt(
        humans_language=humans_language,
        msg=msg
    )
    
    response = get_completion(prompt)

    return response


def respond_to_conversational_message(chat_history, conversational_msg, humans_language='english'):
    system_template = """
The following is a friendly conversation between a human and an AI Badminton Court booking assistant chatbot. The assistant always replies in a happy and friendly tone.
The assistant's main objective is to aid clients in securing their desired Badminton Courts by addressing their inquiries. 
The assistant ignores all other requests that are not related to helping the client to booking Badminton Courts.
The assistant will try to get the human's interested location and date to help the human find a badminton court.
Respond to the ongoing exchange with a friendly tone in {humans_language} as it appears the human is engaging in conversation.

Current conversation:
{history}
Human: {input}
AI response in {humans_language}:
"""
    conversation = ConversationChain(llm = get_llm_instance(), memory=chat_history, verbose=True)
    conversation.prompt = PromptTemplate.from_template(system_template)

    response = conversation.predict(
        input = conversational_msg,
        humans_language=humans_language,
    )

    return response

def respond_to_get_more_info(search_param, cleaned_dates, cleaned_cc, is_simulation_mode=True, humans_language="english"):

    missing_info_msg = ''
    existing_info_msg = ''

    if is_simulation_mode:
        df = get_simulated_data()
        simulated_date_range = sorted(df.date.unique())

        min_date_range = min(simulated_date_range)
        max_date_range = max(simulated_date_range)

    else:
        min_date_range = datetime.now(pytz.timezone("Singapore")).date()
        max_date_range = datetime.now(pytz.timezone("Singapore")).date() + timedelta(days=15)

    if 'none' in cleaned_dates or len(cleaned_dates) == 0:
        missing_info_msg =  missing_info_msg + f"- Please specify the date you are interested in? Please note that the date should be between {min_date_range.strftime('%d %B %Y')}, and {max_date_range.strftime('%d %B %Y')}.\n"
    else:
        existing_info_msg = existing_info_msg + f"- I understand that you are interested in the following dates: {', '.join(cleaned_dates)}.\n"

    if 'none' in cleaned_cc or len(cleaned_cc) == 0:
        missing_info_msg = missing_info_msg + "- Please provide a specific location in Singapore where you'd like to book the court?\n"
    else:
        existing_info_msg = existing_info_msg + f"- I understand that you are interested in these locations: {', '.join(search_param.locations)}. I have found the following nearby facilities: {', '.join(cleaned_cc)}\n"

    formatted_message = existing_info_msg + missing_info_msg
    response = f"""
It looks like you're interested in booking a badminton court! However, I need a bit more information to help you better. 

{existing_info_msg}{missing_info_msg}

For example, you could say: "I am looking to book a badminton court in Jurong West on {max_date_range.strftime('%d %B')}". Looking forward to your response!
    """

    return translate_if_not_english(response,humans_language)

def respond_no_results_found(rephrase_user_request, cleaned_cc, cleaned_dates, humans_language='english'):

    response = f"""
{rephrase_user_request}

I've checked the availability for {', '.join(cleaned_cc)} on {', '.join(cleaned_dates)}. Unfortunately, there are no availabile timings at these locations on these dates.

Would you like to consider alternative dates or perhaps explore other locations? I'm here to help you find the best option!
    """

    return translate_if_not_english(response,humans_language)


def respond_too_many_results_found(rephrase_user_request, result_df, humans_language='English'):

    response = f'''
{rephrase_user_request}

We have {len(result_df)} options across {', '.join(result_df['outlet_name'].unique())} for the dates {', '.join(result_df['date'].unique())}.

As there are quite a few options, could you please provide more specific details like preferred dates and locations? This will help me narrow down the options for you. Looking forward to your response!
    '''

    return translate_if_not_english(response,humans_language)


def respond_to_user_input(user_input, session_state_chat_history, is_simulation_mode = True):

    st.session_state.result_df = ''
    st.chat_message("assistant").write(f'Thinking...')
    chat_history = construct_chat_history(session_state_chat_history)

    #Step 1: Understand the user's message
    parsed_user_intent = understand_user_intent(user_input, chat_history.buffer)

    print(parsed_user_intent)

    if parsed_user_intent.type_of_request == 'conversational_message':
        return respond_to_conversational_message(chat_history, user_input, parsed_user_intent.humans_language)

    #Step 2: extract search parameters
    search_param = extract_search_parameters(parsed_user_intent.crux_message)

    print(search_param)

    cleaned_dates = []
    for d in search_param.dates:
        if d == 'none':
            continue
        interested_dates = get_interested_dates(d,is_simulation_mode)
        cleaned_dates.extend(interested_dates.interested_dates)
    
    cleaned_dates = sorted(list(set(cleaned_dates)))
    
    cleaned_locations = []
    cleaned_cc = []

    for l in search_param.locations:
        if l == 'none':
            continue

        nearby_facility_ids, nearby_cc = get_nearby_facilities(l)

        cleaned_locations.extend(list(nearby_facility_ids))
        cleaned_cc.extend(list(nearby_cc))
    
    cleaned_locations = sorted(list(set(cleaned_locations)))
    cleaned_cc = sorted(list(set(cleaned_cc)))


    print(cleaned_dates, cleaned_locations)

    if 'none' in cleaned_locations or 'none' in cleaned_dates or len(cleaned_dates) == 0 or len(cleaned_locations) == 0:
        return respond_to_get_more_info(search_param, cleaned_dates, cleaned_cc, is_simulation_mode, parsed_user_intent.humans_language)

    search_param.dates = cleaned_dates
    search_param.locations = cleaned_locations

    #Step 3: perform search
    st.chat_message("assistant").write(f'Fetching results...')
    formated_result_list = ''

    if is_simulation_mode:

        result_df = get_simulated_availability_details(location_list = search_param.locations, date_list = search_param.dates)
        st.session_state.result_df = result_df

    else:
        #Cut down the search parameters so as to reduce the API limit issue...

        if len(search_param.dates) > 1:
            st.chat_message("assistant").write(f"I noticed that you are asking for several dates. Due to an API limit, I will only search for the first date, {search_param.dates[:1]}")
            search_param.dates = search_param.dates[:1]

        if len(search_param.locations) > 3:
            st.chat_message("assistant").write(f"I noticed that you are asking for several different areas. Due to an API limit, I will only search for the first 3 locations, {search_param.locations[:3]}")
            search_param.locations = search_param.locations[:3]


        result_df, searched_outlets = get_live_availability_details(location_list = search_param.locations, date_list = search_param.dates)
        
        print(searched_outlets)
        cleaned_cc = sorted(list(set(searched_outlets)))


    #Step 4: give response based on search
    if len(result_df) == 0:
        response = respond_no_results_found(
            rephrase_user_request(parsed_user_intent.crux_message),
            cleaned_cc,
            search_param.dates,
            parsed_user_intent.humans_language,
        )

        return response

    elif len(result_df) > 15:
        response = respond_too_many_results_found(
            rephrase_user_request(parsed_user_intent.crux_message),
            result_df,
            parsed_user_intent.humans_language
        )

        return response
    else:
        response = get_availability_response_from_summarized_dataframe(
            result_df,
            rephrase_user_request(parsed_user_intent.crux_message), 
            cleaned_cc, 
            search_param.dates, 
            parsed_user_intent.humans_language
        )

        return response



#### Live methods here ####
## Get availability slots
def get_live_availability_details(location_list, date_list):
    
    result = []
    outlets = []

    for fid in location_list:
        for input_date in date_list:
            outlet_name, single_search_result = get_facility_details_from_pa_website(fid, input_date)
            outlets.append(outlet_name)
            result.extend(single_search_result)

    if len(result) == 0:
        #return an empy list
        return result, outlets

    df = pd.DataFrame(result)
    df = df.sort_values(by=['booking_url','startTime'])
    result_df = summarize_availability_dataframe(df)

    return result_df, outlets


def attempt_request_get(url, max_attempts=5, wait_delay=10):
    attempts = 0
    while attempts < max_attempts:
        try:
            response = requests.get(url)
            response_data = response.json()
        except Exception as e:
            print("An error occurred:", str(e))
            attempts += 1
            print(f"Trying again... Attempt {attempts}")
            st.chat_message("assistant").write(f'Oops, got rate limited by the API, will need to wait {attempts * wait_delay} seconds...')
            time.sleep(attempts * wait_delay)  # Add a delay before retrying
        else:
            break
    else:
        print("Max attempts reached. Exiting...")
        return
    
    return response_data

def get_facility_details(outlet, facility_type):
    get_facility_details_url = 'https://www.onepa.gov.sg/pacesapi/facilitysearch/searchjson?facility={facility_type}&outlet={outlet}'
    print(get_facility_details_url.format(facility_type = facility_type, outlet = outlet))
    f_details_json = attempt_request_get(get_facility_details_url.format(facility_type = facility_type, outlet = outlet))
    #f_details_response = requests.get(get_facility_details_url.format(facility_type = facility_type, outlet = outlet))
    #f_details_json = f_details_response.json()
    
    return f_details_json

def get_facility_slots(f_id, date_str):
    get_facility_slots_url = 'https://www.onepa.gov.sg/pacesapi/facilityavailability/GetFacilitySlots?selectedFacility={facility_id}&selectedDate={date_str}'
    print(get_facility_slots_url.format(facility_id = f_id, date_str = date_str))
    f_slots_json = attempt_request_get(get_facility_slots_url.format(facility_id = f_id, date_str = date_str))
    #f_slots_response = requests.get(get_facility_slots_url.format(facility_id = f_id, date_str = date_str))
    #f_slots_json = f_slots_response.json()
    
    return f_slots_json

def get_facility_details_from_pa_website(facility_id, date_str):
    outlet_name = ''
    formatted_list = []

    try:
        date_str = date_parser.parse(date_str, dayfirst=True).strftime('%d/%m/%Y')
    except:
        return f"{date_str} is an invalid date format!"
    
    try:
        outlet = facility_id.split('_')[0]
        facility_type = facility_id.split('_')[1]
    except:
        print(f'{facility_id} is an invalid facility_id!')
        return outlet_name, formatted_list
    
    f_details_json = get_facility_details(outlet, facility_type)
    
    if not f_details_json['data']['results']:
        print('empty facility details!')
        return outlet_name, formatted_list
    
    max_price = f_details_json['data']['results'][0]['maxPrice']
    outlet_name = f_details_json['data']['results'][0]['outlet']
    facility_type = f_details_json['data']['results'][0]['title']

    f_slots_json = get_facility_slots(facility_id, date_str)

    formatted_list = []

    resp_date_str = f_slots_json['response']['date']
    if not f_slots_json['response']['resourceList']:
        print('empty JSON!')
        return outlet_name, formatted_list

    for resource in f_slots_json['response']['resourceList']:
        resource_id = resource['resourceId']
        resource_name = resource['resourceName']

        for slot in resource['slotList']:
            if slot['isAvailable']:
                startTime = slot['startTime']
                endTime = slot['endTime']
                time_range_name = slot['timeRangeName']
                availability_status = slot['availabilityStatus']
                is_available = slot['isAvailable']

                formatted_slot = {
                    'outlet_name': outlet_name,
                    'facility_id': facility_id,
                    'facility_type': facility_type,
                    'date': resp_date_str,
                    'resourceName': resource_name,
                    #'resourceId': resource_id,
                    'timeRangeName': time_range_name,
                    'startTime': startTime,
                    'endTime': endTime,
                    #'availabilityStatus': availability_status,
                    #'isAvailable': is_available,
                    'price': f'{max_price:.2f}',
                    'booking_url': generate_pa_facility_booking_link(facility_id, date_str)
                }
                formatted_list.append(formatted_slot)
    
    return outlet_name, formatted_list

def generate_pa_facility_booking_link(fid, date_str):
    return f'https://www.onepa.gov.sg/facilities/availability?facilityId={fid}&date={date_str}&time=all'
