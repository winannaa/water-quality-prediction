import pandas as pd

URL_PANEL_A = "https://drive.google.com/uc?export=download&id=1fZCYvQ8JDaJmuy2E1IcdU7ijfsGp4mbz"
URL_PANEL_B = "https://drive.google.com/uc?export=download&id=1CdtMOSwjHd8OLfrbJy1AxVRajB51OcvQ"

@st.cache_data
def load_panelA():
    return pd.read_csv(URL_PANEL_A)

@st.cache_data
def load_panelB():
    return pd.read_csv(URL_PANEL_B)
