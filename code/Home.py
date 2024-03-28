"""
Streamlit app for visualizing behavior data
https://foraging-behavior-browser.streamlit.app/

Note the url is now queryable, e.g. https://foraging-behavior-browser.streamlit.app/?subject_id=41392

Example queries:
 /?subject_id=699982   # only show one subject
 /?session=10&session=20  # show sessions between 10 and 20
 /?tab_id=tab_1  # Show specific tab
 /?if_aggr_all=false

"""

#%%
import streamlit as st

def app():
    st.markdown('## Dear foraging browser user, \n'
                '#### For better stability and accessibility, this app has been migrated to Amazon ECS (thank you Yoseh and Jon 🙌).\n'
                '##         👉  [Click me](https://aindephysforagingapplb-23793000.us-west-2.elb.amazonaws.com/) ')
    st.divider()
    st.markdown('If you see this, click "Advanced"')
    st.image('https://github.com/hanhou/foraging-behavior-browser/assets/24734299/4431c438-438c-4591-a62e-15526615d5b0')
    st.markdown('and then click "Continue to..."')
    st.image('https://github.com/hanhou/foraging-behavior-browser/assets/24734299/0799dc32-8bc5-4c53-a85a-58c50f57f33e')

app()

