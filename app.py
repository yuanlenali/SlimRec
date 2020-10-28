import streamlit as st
import csv
import os.path
import torch
from PIL import Image

from model.two_tower import TwoTower
import metrics.recall_k as metrics
import utils.crawl_image as download
import utils.style as style

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_MOVIE_RECOMMENDED = 5


def get_meta_info(meta_data):
    meta_infos = [0] * 3
    with open(meta_data, newline='') as file:
        rows = csv.reader(file, delimiter=':', quotechar='|')
        for idx, row in enumerate(rows):
            meta_infos[idx] = int(row[1])
    return meta_infos


@st.cache()
def topkRate_have_watched(user_id, k=NUM_MOVIE_RECOMMENDED):
    return metrics.topkRate_have_watched(torch.LongTensor([int(user_id)]), k)


def sidebar_model():
    style.empty_lines(2, side_bar=True)
    option = st.sidebar.selectbox(
                'Recommendation Engine', ['baseline model: TTSN', 'SlimRec']
                )
    ph_ratio = 1
    quantization = False
    if option == 'SlimRec':
        style.empty_lines(3, side_bar=True)
        ph_ratio = st.sidebar.slider(
                    'Popularity Hashing Compression Ratio', 0.0, 0.99, 0.7
                    )
        quantization = st.sidebar.checkbox('Quantization')
    return option, ph_ratio, quantization


def sidebar_model_size(quantization):
    n_user, n_movie, n_category = get_meta_info('data/train_meta_data.dat')
    baseline_size = int((n_user + n_movie) * 64 * 4 * 2 * 1.03 / 1024 / 1024)
    if option == 'baseline model: TTSN':
        style.empty_lines(17, side_bar=True)
        st.sidebar.info("TTSN model size is {}MB".format(baseline_size))
    elif option == 'SlimRec':
        style.empty_lines(4, side_bar=True)
        model_size = round((n_user + n_movie) * 64 * 4 * 2 * 1.03 / 1024 / 1024 * ph_ratio, 2)
        compression_ratio = ph_ratio
        if quantization:
            model_size = round(model_size * 0.25, 2)
            compression_ratio = round(compression_ratio * 0.25, 2)
        st.sidebar.info("Baseline: TTSN model size is {}MB".format(baseline_size))
        st.sidebar.info("SlimRec model size is {}MB".format(model_size))
        st.sidebar.info("Compression Ratio is {}".format(compression_ratio))
    return n_user, n_movie, n_category


def main_panel():
    style.local_css("utils/style.css")
    style.remote_css()
    user_id = st.text_input('Enter User ID from 1 to 60,000', '1')
    style.empty_lines(3)
    st.markdown(
        "<div>Welcome back,<span class='bold'>{}</span> !</div>".format(user_id),
        unsafe_allow_html=True
    )
    style.empty_lines(4)
    return user_id


def main_panel_watching_history(user_id):
    watched_movie = topkRate_have_watched(user_id, NUM_MOVIE_RECOMMENDED)
    watched_movie_image = ["tt" + str(movie_watched[1]) + ".jpeg" for 
                            (movie_watched, rating) in watched_movie]
    watched_movie_names = [movie_watched[0] for (movie_watched, rating) in watched_movie]
    st.markdown("<div><span class='sec_title'>Watch again your favorite movies:</span></div>", 
                unsafe_allow_html=True)
    for m in watched_movie_image:
        if not os.path.exists("data/images/"+m):
            imdb = m[:-5]
            download.tmdb_posters(imdb, "data/images")
    images = [Image.open("data/images/"+m) for m in watched_movie_image]
    image_iterator = style.paginator("", images)
    _, images_on_page = map(list, zip(*image_iterator))
    st.image(images_on_page, width=126, caption=watched_movie_names)
    style.empty_lines(4)


def main_panel_recomendations(option, ph_ratio, quantization, n_user, n_movie, n_category):
    with torch.no_grad():
        if option == 'baseline model: TTSN':
            model = TwoTower(num_user=n_user, num_movie=n_movie, num_category=n_category, 
                                popularity_hashing=False).to(DEVICE)
            file = "checkpoints/ep_15_lr_0.012_baseline"
        elif option == 'SlimRec':
            ph_ratio = int(ph_ratio * 10) / 10
            model = TwoTower(num_user=n_user, num_movie=n_movie, 
                                        num_category=n_category, popularity_hashing=True, 
                                        uid_ph=[ph_ratio, 0.9], mid_ph=[ph_ratio, 0.9]).to(DEVICE)
            file = "checkpoints/ep_15_lr_0.012_pHash_{}".format(ph_ratio)
        model.load_state_dict(torch.load(file))
        recommended_movie = metrics.topK_movie_name_and_id(model, torch.LongTensor([int(user_id)]), -1)

    recommended_movie = [(movie_recommended[0], "tt" + movie_recommended[1] + ".jpeg") for 
                            (movie_recommended, rating) in recommended_movie]
    st.markdown("<div><span class='sec_title'>We think you might also like:</span></div>", 
                    unsafe_allow_html=True)
    images = []
    movie_name = []
    cnt = 0
    for m in recommended_movie:
        if cnt >= NUM_MOVIE_RECOMMENDED:
            break
        if not os.path.exists("data/images/"+m[1]):
            imdb = m[1][:-5]
            download.tmdb_posters(imdb, "data/images")
        if os.path.exists("data/images/"+m[1]):
            images.append(Image.open("data/images/"+m[1]))
            movie_name.append(m[0])
            cnt += 1
    image_iterator = style.paginator("", images)
    _, images_on_page = map(list, zip(*image_iterator))
    st.image(images_on_page, width=126, caption=movie_name)


if __name__ == '__main__':
    # SideBar: select the model
    option, ph_ratio, quantization = sidebar_model()

    # SideBar: Model Size Display
    n_user, n_movie, n_category = sidebar_model_size(quantization)

    # Main panel
    user_id = main_panel()

    # Main panel watching history
    main_panel_watching_history(user_id)

    # Main panel recommendations
    main_panel_recomendations(option, ph_ratio, quantization, 
                                n_user, n_movie, n_category)

