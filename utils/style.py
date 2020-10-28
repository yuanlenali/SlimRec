import streamlit as st
import itertools

NUM_MOVIE_RECOMMENDED = 5


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def remote_css(url='https://fonts.googleapis.com/icon?family=Material+Icons'):
    st.markdown(
        f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True
    )


def icon(icon_name):
    st.markdown(
        f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True
    )


def paginator(label, items, items_per_page=10, on_sidebar=True):
    # Display a pagination selectbox in the specified location.
    items = list(items)
    return itertools.islice(enumerate(items), 0, NUM_MOVIE_RECOMMENDED)


def empty_lines(n, side_bar=False):
    for _ in range(n):
        if side_bar:
            st.sidebar.text("")
        else:
            st.text("")