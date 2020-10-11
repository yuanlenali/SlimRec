import os
import csv
import requests

CONFIG_PATTERN = 'http://api.themoviedb.org/3/configuration?api_key={key}'
IMG_PATTERN = 'http://api.themoviedb.org/3/movie/{imdbid}/images?api_key={key}' 
KEY = 'a95bdd0303029f8ebee1bfa57d4eabd1'
            
def _get_json(url):
    r = requests.get(url)
    return r.json()
    
def _download_images(urls, imdbid, path='.'):
    """download all images in list 'urls' to 'path' """

    for nr, url in enumerate(urls):
        # only down the first poster
        if nr >= 1:
            break
        r = requests.get(url)
        filetype = r.headers['content-type'].split('/')[-1]
        filename = '{0}.{1}'.format(imdbid, filetype)
        filepath = os.path.join(path, filename)
        with open(filepath,'wb') as w:
            w.write(r.content)

def get_poster_urls(imdbid):
    """ return image urls of posters for IMDB id
        returns all poster images from 'themoviedb.org'. Uses the
        maximum available size. 
        Args:
            imdbid (str): IMDB id of the movie
        Returns:
            list: list of urls to the images
    """
    config = _get_json(CONFIG_PATTERN.format(key=KEY))
    base_url = config['images']['base_url']
    sizes = config['images']['poster_sizes']

    """
        'sizes' should be sorted in ascending order, so
            max_size = sizes[-1]
        should get the largest size as well.        
    """
    def size_str_to_int(x):
        return float("inf") if x == 'original' else int(x[1:])
    max_size = max(sizes, key=size_str_to_int)

    json_info = _get_json(IMG_PATTERN.format(key=KEY,imdbid=imdbid))
    if 'posters' not in json_info:
        return []
    posters = json_info['posters']
    poster_urls = []
    for poster in posters:
        rel_path = poster['file_path']
        url = "{0}{1}{2}".format(base_url, max_size, rel_path)
        poster_urls.append(url) 

    return poster_urls

def tmdb_posters(imdbid, outpath='.', count=None):    
    urls = get_poster_urls(imdbid)
    if count is not None:
        urls = urls[:count]
    _download_images(urls, imdbid, outpath)

if __name__=="__main__":
    with open('../data/movie_id_mapping.dat', 'r') as file:
        rows = csv.reader(file, delimiter=':')
        for row in rows:
            movie_id = row[2]
            imdb = 'tt' + movie_id
            tmdb_posters(imdb, "../data/images")

