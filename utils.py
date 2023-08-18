import streamlit as st
import requests
import pandas as pd
from pandas import json_normalize
from datetime import datetime, timedelta
from dateutil import parser as date_parser
import calendar
from math import radians, cos, sin, asin, sqrt
import time

from google.oauth2 import service_account

from langchain.chat_models import AzureChatOpenAI
from langchain.chat_models import ChatVertexAI

from langchain import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import ConversationChain
from pydantic import BaseModel, Field
from typing import List

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

def get_llm_api_config_details():
    '''
    config = {
        'api_type' : st.secrets['api_type'],
        'api_base' : st.secrets['api_base'],
        'api_version' : st.secrets['api_version'],
        'api_key' : st.secrets['api_key']
    }
    
    return config
    
    '''
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
    

@st.cache_resource 
def get_llm_instance(model='gpt-4'):   

    '''
    #model = 'gpt-4' | 'gpt-4-32k' | 'gpt-35-turbo'
    config = get_llm_api_config_details()
    llm = AzureChatOpenAI(
        deployment_name=model, 
        model_name=model, 
        openai_api_key = config['api_key'],
        temperature=0, 
        max_tokens=500,
        n=1, 
        openai_api_base = config['api_base'],
        openai_api_version = config['api_version'],
    )
    '''
    config = get_llm_api_config_details()

    llm = ChatVertexAI(
        model_name="chat-bison@001",
        max_output_tokens=1024,
        temperature=0,
        top_p=0.8,
        top_k=40,
        verbose=True,
        credentials = config,
        project=config.project_id
    )
    
    return llm

## Date formatter
def todayDate():
    today = datetime.now()
    return today.strftime('%d/%m/%Y')

# get day of week for a date (or 'today')
def dayOfWeek(date):
    today = datetime.now()
    if date == 'today':
        return calendar.day_name[today.weekday()]
    else:
        try:
            theDate = parser.parse(date)
        except:
            return 'invalid date format, please use format: dd/mm/yyyy'

        return calendar.day_name[theDate.weekday()]

def get_calendar_reference():
    # Get today's date
    today = datetime.now().date()

    # Initialize the output string
    output = ""

    # Generate and append day of the week and date for the next 28 days
    for day in range(16):
        current_date = today + timedelta(days=day)
        day_of_week = current_date.strftime('%A')
        date_formatted = current_date.strftime('%d/%m/%Y')
        if current_date == today:
            date_formatted = f"{date_formatted} [TODAY]"
        output += f'- {day_of_week} {date_formatted}\n'

    return output


def get_interested_dates(date_str):
    class InterestedDates(BaseModel):
        interested_dates: List[str] = Field(description='a list of dates in dd/mm/yyyy format that the client could be interested in when booking the facility. If this information is not found, output [-1].')

    parser = PydanticOutputParser(pydantic_object=InterestedDates)
    format_instructions = parser.get_format_instructions()

    template = """
The calendar for the next 15 days are as deliminted by triple backticks:
```
{calendar_ref}
```

Today's date is {today_day_of_week} {today_date_str}.
Identify the closest date or dates `{date_str}` could be referring to from the calendar above in dd/mm/yyyy format. 
If there are no valid dates in the calendar above or if the date doesn't exist, do not make up an answer and just output [-1].
{format_instructions}
    """
    prompt = ChatPromptTemplate.from_template(
        template=template, 
    )

    messages=prompt.format_messages(
        today_day_of_week = dayOfWeek('today'),
        today_date_str = todayDate(),
        calendar_ref = get_calendar_reference(),
        date_str = date_str,
        format_instructions = format_instructions
    )

    llm = get_llm_instance()

    response = llm(messages)
    output = parser.parse(response.content)

    return output


## Get nearby facilities
def getcoordinates(address):
    req = requests.get('https://developers.onemap.sg/commonapi/search?searchVal='+address+'&returnGeom=Y&getAddrDetails=Y&pageNum=1')
    resultsdict = eval(req.text)
    if len(resultsdict['results'])>0:
        return resultsdict['results'][0]['LATITUDE'], resultsdict['results'][0]['LONGITUDE']
    else:
        pass


@st.cache_resource
def get_facility_outlets():
    get_facility_outlets_url = 'https://www.onepa.gov.sg/pacesapi/FacilitySearch/GetFacilityOutlets'

    response = requests.get(get_facility_outlets_url)
    json_data = response.json()

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
    ], axis=1)

    return facility_id_df

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


def get_nearby_facilities(user_interested_location):
    '''
    1. geocode location
    2. filter "value" to facility of choice
    3. get top x distinct id based on closest haversine distance
    '''
    #To get rid of duplicated facility_id entries in full dataframe
    nearest_id_set = set()
    
    user_coords = getcoordinates(user_interested_location)
    
    if user_coords:
        facility_id_df = get_facility_outlets()

        facility_id_df['dist_from_user'] = facility_id_df[['lat','lng']].apply(
            lambda x: haversine(x['lat'], x['lng'], float(user_coords[0]),float(user_coords[1])),
            axis=1
        )

        sorted_df = facility_id_df[facility_id_df['value'] == 'BADMINTON COURTS'].sort_values('dist_from_user')
        for index, row in sorted_df.iterrows():
            nearest_id_set.add(row.id)
            if len(nearest_id_set) >=3:
                break
    
    print(nearest_id_set)
    return nearest_id_set


## Get availability slots
def attempt_request_get(url, max_attempts=5, wait_delay=45):
    attempts = 0
    while attempts < max_attempts:
        try:
            response = requests.get(url)
            response_data = response.json()
        except Exception as e:
            print("An error occurred:", str(e))
            st.chat_message("assistant").write(f'Oops, got rate limited by the API, will need to wait {wait_delay} seconds...')
            attempts += 1
            print(f"Trying again... Attempt {attempts}")
            time.sleep(wait_delay)  # Add a delay before retrying
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

def generate_pa_facility_booking_link(fid, date_str):
    return f'https://www.onepa.gov.sg/facilities/availability?facilityId={fid}&date={date_str}&time=all'

def format_availibility_from_list_to_string(availability_list):
    template = '''
Summarize and format the following information in bullet points. Combine repeated information into just 1 bullet point.
Be sure to preserve the booking_url, outlet_name, timeRangeName and date fields:
```
{availability_list}
```
    '''
    prompt = ChatPromptTemplate.from_template(template=template)
    messages = prompt.format_messages(
        availability_list=availability_list, 
    )
    llm = get_llm_instance()
    response = llm(messages) 
    
    return response.content

def get_facility_availability(facility_id, date_str):
    try:
        date_str = date_parser.parse(date_str, dayfirst=True).strftime('%d/%m/%Y')
    except:
        return f"{date_str} is an invalid date format!"
    
    try:
        outlet = facility_id.split('_')[0]
        facility_type = facility_id.split('_')[1]
    except:
        return f'{facility_id} is an invalid facility_id!'
    
    f_details_json = get_facility_details(outlet, facility_type)
    
    if not f_details_json['data']['results']:
        print('empty facility details!')
        return 'no facility found'
    
    max_price = f_details_json['data']['results'][0]['maxPrice']
    outlet_name = f_details_json['data']['results'][0]['outlet']

    f_slots_json = get_facility_slots(facility_id, date_str)

    formatted_list = []

    resp_date_str = f_slots_json['response']['date']
    if not f_slots_json['response']['resourceList']:
        print('empty JSON!')
        return "no available timings"

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
                    'date': resp_date_str,
                    'resourceName': resource_name,
                    #'resourceId': resource_id,
                    'timeRangeName': time_range_name,
                    #'startTime': startTime,
                    #'endTime': endTime,
                    #'availabilityStatus': availability_status,
                    #'isAvailable': is_available,
                    'price': max_price,
                    'booking_url': generate_pa_facility_booking_link(facility_id, date_str)
                }
                formatted_list.append(formatted_slot)

    if len(formatted_list) == 0:
        return "no available timings"
    else:
        return format_availibility_from_list_to_string(formatted_list)


## Parse user input
def parse_user_input(user_input, session_state_chat_history):
    
    chat_history = ''
    
    #skip the first intro greeting line by the ai assistant
    for line in session_state_chat_history[1:]:
        if line['role'] == 'user':
            chat_history = chat_history + f"<Human>: {line['content']}\n"
        if line['role'] == 'assistant':
            chat_history = chat_history + f"<Assistant>: {line['content']}\n"
    
    class UserInput(BaseModel):
        type_of_request: str = Field(description='the type of request that the client is making based on the latest message. This will be one of the following values: <request_question>, <conversational_message>')
        interested_locations: List[str] = Field(description='a list of all locations that the client might be interested in when booking the facility. If this information is not found, output [-1] as a list with just 1 element.')
        interested_dates: List[str] = Field(description='a list of all relative date expressions that the client might be interested in when booking the facility. If this information is not found, output [-1] as a list with just 1 element.')
    
    parser = PydanticOutputParser(pydantic_object=UserInput)
    format_instructions = parser.get_format_instructions()
    
    template = """
You are reviewing a conversation between a facility booking conceirge and a human. 
Your objective is to analyze the chat history and the latest human input to infer the ultimate request the human is seeking.
Give more importance to the latest human input in your analysis. If information is unavailable, use the chat history to understand the user's intent.
The end goal is to have this output be passed onto the facility booking conceirge so they will know what to process.

Previous conversation:
```
{chat_history}
```
Latest human input: ```{user_input}```

Extract the following information:

type_of_request: The type of request that the client is making making based on the latest message. This will be one of the following values: <request_question>, <conversational_message>
location: A list of all potential locations that the client is interested in. If this information is not found, output [-1] as a list with just 1 element.
requested_date: A list of all relative date expressions that the client is interested in. Copy the exact text as provided by the client without making any additional inferences. If this information is not found, output [-1] as a list with just 1 element.

{format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(
        template=template,
    )
    
    messages = prompt.format_messages(
        format_instructions=format_instructions,
        chat_history=chat_history,
        user_input=user_input
    )
    
    llm = get_llm_instance()
    
    response = llm(messages) 
    output = parser.parse(response.content)

    return output


## Respond to user's input

## NOTES: based on test, the convo history seems to be confusing the LLM (Vertex is impacted by this, openai doesn't have this problem). Maybe remove convo history? Then maybe don't need to use conversationchain and just use regular LLMchain?
def respond_to_user_input(user_input_str, session_state_chat_history):
    available_slots_str = ''
    chat_history = ConversationBufferMemory(input_key='input')

    #Might want to consider only saving the last k messages? So as not to make the prompt too long for longer conversations?
    for line in session_state_chat_history[1:]:
        if line['role'] == 'user':
            chat_history.chat_memory.add_user_message(line['content'])
        if line['role'] == 'assistant':
            chat_history.chat_memory.add_ai_message(line['content'])

    conversation = ConversationChain(llm = get_llm_instance(), memory=chat_history, verbose=True)


    user_request = parse_user_input(user_input_str, session_state_chat_history)

    print(user_request)

    if user_request.type_of_request == 'conversational_message':
        # it is a conversational message, just need the bot to respond properly.
        system_template = """
The following is a friendly conversation between a human and an AI facility booking assistant. The assistant always replies in a happy and friendly tone.
The assistant's main objective is to aid clients in securing their desired facilities by addressing their inquiries. The assistant ignores all other requests that are not related to helping the client to booking facilities.

Current conversation:
{history}
Human: {input}
AI:
"""

    elif "-1" in user_request.interested_dates:
        # it is a request, but either the date or location is missing, will need to prompt users to get that info
        system_template = """
The following is a friendly conversation between a human and an AI facility booking assistant. The assistant always replies in a happy and friendly tone.
The assistant's main objective is to aid clients in securing their desired facilities by addressing their inquiries. The assistant ignores all other requests that are not related to helping the client to booking facilities.
From the ongoing conversation, it seems the human has not shared details about the date they are interested in. The assistant will prompt them in a friendly manner to ask about their interested date.

Current conversation:
{history}
Human: {input}
AI:
"""

    elif "-1" in user_request.interested_locations:
        # it is a request, but either the date or location is missing, will need to prompt users to get that info
        system_template = """
The following is a friendly conversation between a human and an AI facility booking assistant. The assistant always replies in a happy and friendly tone.
The assistant's main objective is to aid clients in securing their desired facilities by addressing their inquiries. The assistant ignores all other requests that are not related to helping the client to booking facilities.
From the ongoing conversation, it seems the human has not shared details about the location they are interested in. The assistant will prompt them in a friendly manner to ask where in Singapore they will be interested in.

Current conversation:
{history}
Human: {input}
AI:
"""

    #elif is_simulated_mode:
    #   Code to run simulated mode... assuming we are not limited by the API
    #   remember to add this tick option on the chatbot UI. Or maybe execute a different method?
    #   esp since simulated data means "today" will be different from real today




    else:
        # regular user request parsing

        #Begin fetching results
        st.chat_message("assistant").write("Fetching results...")
        cleaned_interested_dates = []

        for input_date in user_request.interested_dates:
            cleaned_dates = get_interested_dates(input_date)
            cleaned_interested_dates.extend([date_parser.parse(i, dayfirst=True).strftime('%d/%m/%Y') for i in cleaned_dates.interested_dates if i != -1])

        user_request.interested_dates = sorted(list(set(cleaned_interested_dates)))
        print(f'after cleaning dates: {user_request}')

        if len(user_request.interested_dates) > 2:
            st.chat_message("assistant").write(f"I noticed that you are asking for more than 2 different dates. Due to an API limit, I will only search for the first 2 dates, {user_request.interested_dates[:2]}")

        if len(user_request.interested_locations) > 2:
            st.chat_message("assistant").write(f"I noticed that you are asking for more than 2 different areas. Due to an API limit, I will only search for the first 2 areas, {user_request.interested_locations[:2]}")

        nearest_id_set = set()

        for input_location in user_request.interested_locations[:2]:
            nearest_id_set.update(get_nearby_facilities(input_location))

        for fid in nearest_id_set:
            for input_date in user_request.interested_dates[:2]:
                result = get_facility_availability(fid, input_date)

                #if result == 'no facility found' or result == 'no available timings':
                #    continue

                available_slots_str += f'\n Current availability for {fid} on {input_date}: \n'
                available_slots_str += result + ' \n'


        system_template = """
The following is a friendly conversation between a human and an AI facility booking assistant. The assistant always replies in a happy and friendly tone.
The assistant will respond by referring to both the user's request and the availability details provided, ensuring that it gives accurate responses and do not make up an answer.
If there are availability options, the assistant will present all the choices listed below and give the facility and availability details for the human's information, and give the booking URL link(s) for the human to do their booking via the website. The message must be in point form for easier reading. THE URL LINK MUST ALWAYS BE INCLUDED IN YOUR RESPONSE!
If there are no availability options, the assistant will reference the human's request and state that no options meet their request in a friendly manner.

Human's request: ```{user_request}```

Here are the facility availability details for options near the requested location and date: 
```
{available_slots}
```

"""

    conversation.prompt = PromptTemplate.from_template(system_template)

    response = conversation.predict(
        available_slots = available_slots_str,
        input = user_input_str,
        user_request = user_request.json()
    )

    return response
