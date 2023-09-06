from utils import *
import streamlit as st

st.markdown(get_custom_css_modifier(), unsafe_allow_html=True)
st.markdown("<h4 style='text-align: Left;'>💬 PA facility booking concierge</h4>", unsafe_allow_html=True)

settings_tab, chat_tab = st.tabs(['settings', 'chat'])

with settings_tab:

    caveat_text = "I acknowledge that AI language models may generate inaccurate or even hallucinated responses. I understand the importance of using these models with caution, and I take responsibility over the generated output before using them."
    agree_llm = st.checkbox(caveat_text, key='agree_llm')

    if agree_llm:
        st.markdown("**Chatbot settings:**")
        col1, col2 = st.columns(2)

        with col1: 
            simulated_mode_description = '''
            - Test the bot using a simulated dataset, not the live booking API.
            - The live API has rate limits, causing slower performance. Simulated mode demonstrates the bot's full potential without these restrictions, showcasing its true value.
            '''

            simulation_mode = st.checkbox('simulation mode', help=simulated_mode_description, value=True)

        with col2:
            llm_choice_description = '''
            - Choose between 2 LLM providers
                - **Vertex**: PaLM2 (chat-bison) on Google Cloud Vertex AI
                - **OpenAI**: GPT-4 on Azure OpenAI Service
            - **Note**: Switching LLM providers mid-conversation may cause it mimic earlier responses from the chat history. Recommend restarting the conversation to get a feel for the difference.
            '''

            llm_choice = st.radio(
                "Choice of LLM:",
                ["Vertex", "OpenAI"],
                #key = 'llm_choice',
                horizontal = True,
                help=llm_choice_description
            )

            if 'llm_choice' not in st.session_state or st.session_state.llm_choice != llm_choice:
                #st.cache_resource.clear()
                st.session_state.llm_choice = llm_choice
        
        if 'simulated_df' not in st.session_state:
            st.session_state.simulated_df = get_simulated_data()

        df = st.session_state.simulated_df

        if simulation_mode:
            simulated_date_range = sorted(df.date.unique())
            simulated_mode_helper_message = f"We will be using simulated data. You can ask for Badminton Court availability between **{min(simulated_date_range)} (today)** and **{max(simulated_date_range)}**."
            st.info(simulated_mode_helper_message)

            with st.expander("View simulated dataset"):
                st.dataframe(df, hide_index=True)

with chat_tab:

    if agree_llm: 

        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
    else:
        st.write('Please accept the acceptable use policy in the settings tab before proceeding.')

if prompt := st.chat_input(disabled = not agree_llm):

    st.chat_message("user").write(prompt)
    
    response = respond_to_user_input(prompt, st.session_state.messages, simulation_mode)
    #response = get_translated_response(prompt,response)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})

    st.chat_message("assistant").write(response)
    st.experimental_rerun()

if 'result_df' in st.session_state and len(st.session_state.result_df) > 0:
    with st.expander("[Simulation mode] View result query dataset"):
        st.dataframe(st.session_state.result_df, hide_index=True)