import streamlit as st
from src.inference import get_prediction

# Initialise session state variable
if 'input_features' not in st.session_state:
    st.session_state['input_features'] = {}

def app_sidebar():
    st.sidebar.header('Applicant')
    dep = st.sidebar.text_input("No. of Dependents")
    ln_amt = st.sidebar.text_input("Loan Amount '000s", placeholder="in '000s")
    ln_tm = st.sidebar.text_input("Loan Term")
    cbl = st.sidebar.text_input("CIBIL Score (300-900)")
    rav = st.sidebar.text_input("Residential Assets Value '000s", placeholder="in '000s")
    def get_input_features():
        input_features = {'dep': int(dep),
                          'ln_amt': int(ln_amt),
                          'ln_tm': int(ln_tm),
                          'cbl': int(cbl),
                          'rav': int(rav)*1000
                         }
        return input_features
    sdb_col1, sdb_col2 = st.sidebar.columns(2)
    with sdb_col1:
        predict_button = st.sidebar.button("Assess", key="predict")
    with sdb_col2:
        reset_button = st.sidebar.button("Reset", key="clear")
    if predict_button:
        st.session_state['input_features'] = get_input_features()
    if reset_button:
        st.session_state['input_features'] = {}
    return None

def app_body():
    title = '<p style="font-family:arial, sans-serif; color:Black; font-size: 40px;"><b> Welcome to DSSI Loan Assessment</b></p>'
    st.markdown(title, unsafe_allow_html=True)
    default_msg = '**System assessment says:** {}'
    if st.session_state['input_features']:
        assessment = get_prediction(no_of_dependents=st.session_state['input_features']['dep'],
                                    loan_amount=st.session_state['input_features']['ln_amt'],
                                    loan_term=st.session_state['input_features']['ln_tm'],
                                    cibil_score=st.session_state['input_features']['cbl'],
                                    residential_assets_value=st.session_state['input_features']['rav'])
        if assessment == 1:
            st.success(default_msg.format('Approved'))
        else:
            st.warning(default_msg.format('Rejected'))
    return None

def main():
    app_sidebar()
    app_body()
    return None

if __name__ == "__main__":
    main()