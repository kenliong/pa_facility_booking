import streamlit as st
import requests
import pandas as pd
from pandas import json_normalize
from datetime import datetime
import calendar
from math import radians, cos, sin, asin, sqrt
import time

from langchain.chat_models import AzureChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.agents import AgentType


def get_llm_api_config_details():
    config = {
        'api_type' : st.secrets['api_type'],
        'api_base' : st.secrets['api_base'],
        'api_version' : st.secrets['api_version'],
        'api_key' : st.secrets['api_key']
    }

    return config

def get_llm_instance(model = "gpt-35-turbo"):
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
    
    return llm

def parse_user_input(user_input_string):
    
    llm = get_llm_instance()
    
    location_schema = ResponseSchema(
        name='location',
        description="Is there a location mentioned? If this information is not found, output -1"
    )

    date_schema = ResponseSchema(
        name='requested_date',
        description="Is there a date mentioned? If this information is not found, output -1"
    )

    response_schemas = [
        location_schema,
        date_schema
    ]
    
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    instruction_template = """
For the following text, extract the following information:

location: Is there a location mentioned? If this information is not found, output -1

requested_date: Is there a date mentioned? If this information is not found, output -1

text: {text}

{format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(template=instruction_template)

    messages = prompt.format_messages(
        text=user_input_string, 
        format_instructions=format_instructions
    )
    response = llm(messages) #might want to find the langchain equivalent of invoking the prompt template.
    output_dict = output_parser.parse(response.content)
    
    return output_dict

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

def parse_date_input(date_str):

    llm = get_llm_instance()

    agent = create_python_agent(
        llm=llm,
        tool=PythonREPLTool(),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        
        stop=["\nObservation:"]
    )
    
    
    #What is the date of [{date_str}] given that today is {todayDate()}(dd/mm/yyyy)], {dayOfWeek('today')}? The date should always be either in the future of today's date or today's date.
    output = agent.run(f"""
    If today is [{dayOfWeek('today')} {todayDate()}(dd/mm/yyyy)], what will [{date_str}]'s date be? The output date should always be either in the future of today's date or today's date, and in "dd/mm/yyyy".
    """)
    




    ## TO DO: to format this in a prompt template?
    
    



    
    print(output)
    return output

def getcoordinates(address):
    req = requests.get('https://developers.onemap.sg/commonapi/search?searchVal='+address+'&returnGeom=Y&getAddrDetails=Y&pageNum=1')
    resultsdict = eval(req.text)
    if len(resultsdict['results'])>0:
        return resultsdict['results'][0]['LATITUDE'], resultsdict['results'][0]['LONGITUDE']
    else:
        pass

@st.cache_data
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

def get_nearest_label_from_user_location(user_location):
    '''
    1. geocode location
    2. filter "value" to facility of choice
    3. get top x distinct id based on closest haversine distance
    '''
    
    user_coords = getcoordinates(user_location)
    facility_id_df = get_facility_outlets()
    relative_user_facility_id_df = facility_id_df.copy()

    relative_user_facility_id_df['dist_from_user'] = relative_user_facility_id_df[['lat','lng']].apply(
        lambda x: haversine(x['lat'], x['lng'], float(user_coords[0]),float(user_coords[1])),
        axis=1
    )
    
    sorted_df = relative_user_facility_id_df[relative_user_facility_id_df['value'] == 'BADMINTON COURTS'].sort_values('dist_from_user')

    nearest_id_set = set()

    for index, row in sorted_df.iterrows():
        nearest_id_set.add(row.id)

        if len(nearest_id_set) >=3:
            break
            
    
    print(nearest_id_set)
    return nearest_id_set

def generate_pa_facility_booking_link(fid, date_str):
    return f'https://www.onepa.gov.sg/facilities/availability?facilityId={fid}&date={date_str}&time=all'

def get_facility_availability(fid, date_str):
    outlet = fid.split('_')[0]
    facility_type = fid.split('_')[1]
    
    f_details_json = get_facility_details(outlet, facility_type)
    max_price = f_details_json['data']['results'][0]['maxPrice']
    
    f_slots_json = get_facility_slots(fid, date_str)
    return get_available_slots_from_facility_slots_json(f_slots_json, fid)


def attempt_request_get(url, max_attempts=5, wait_delay=40):
    attempts = 0
    while attempts < max_attempts:
        try:
            response = requests.get(url)
            response_data = response.json()
        except Exception as e:
            print("An error occurred:", str(e))
            attempts += 1
            print(f"Trying again... Attempt {attempts}")
            time.sleep(40)  # Add a delay before retrying
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

def get_available_slots_from_facility_slots_json(f_slots_json, facility_id):
    
    # Possible to merge the prior 2 methods, so we can put in the max_price into the json object
    
    formatted_list = []

    resp_date_str = f_slots_json['response']['date']

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
                    'facility_id': facility_id,
                    'date': resp_date_str,
                    'resourceName': resource_name,
                    'resourceId': resource_id,
                    'timeRangeName': time_range_name,
                    'startTime': startTime,
                    'endTime': endTime,
                    'availabilityStatus': availability_status,
                    'isAvailable': is_available
                }
                formatted_list.append(formatted_slot)
                
    return formatted_list

def respond_to_user_input(user_input_str):
    
    output_dict = parse_user_input(user_input_str)

    if output_dict['requested_date'] != -1:
        output_dict['requested_date'] = parse_date_input(output_dict['requested_date'])

    requested_date = output_dict['requested_date']
    user_location = output_dict['location']
    
    nearest_id_set = get_nearest_label_from_user_location(user_location)
    available_slots = []

    for fid in nearest_id_set:
        available_slots.extend(get_facility_availability(fid, requested_date))

    booking_url_set = set()

    for slot in available_slots:
        booking_url_set.add(generate_pa_facility_booking_link(slot['facility_id'],slot['date']))
    
    print(booking_url_set)
    
    system_template = """
You are a helpful facility booking conceirge. Your goal is to assist clients in reserving facilities according to their requests.
You will receive a roster of open time slots for facilities near the user's requested location and the URL links for booking these facilities.
Upon receiving the client's request, you will try to recommend a booking slot and give them the URL link(s) so they can complete the booking on the website. 
If the client did not provide a timing, just assume that they are flexible with any timing and recommend whatever is available. 
If you cannot meet the client's request, refrain from fabricating a response.
    """
    system_message_prompt_template = SystemMessagePromptTemplate.from_template(system_template)

    ai_template = '''
The available courts at the moment are as follows:
```
{available_slots}
```

Court booking URL links:
```
{booking_url_links}
```
    '''

    ai_message_prompt_template = AIMessagePromptTemplate.from_template(ai_template)


    human_template = "My request: ```{user_input}```"
    human_message_prompt_template = HumanMessagePromptTemplate.from_template(human_template)


    template = ChatPromptTemplate.from_messages(
        [
            system_message_prompt_template,
            human_message_prompt_template,
            ai_message_prompt_template
        ]
    )
    
    llm = get_llm_instance('gpt-4')

    response = llm(
        template.format_messages(
            available_slots=available_slots,
            booking_url_links=booking_url_set,
            user_input=user_input_str
        )
    )
    
    return response
