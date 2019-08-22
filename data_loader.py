"""
This script is based on official recomentations regarding 
retrieving data from VBGI digital herbarium:

https://nbviewer.jupyter.org/github/VBGI/herbs/blob/master/herbs/docs/tutorial/Python/ru/Python.ipynb
"""



import json, os
import pandas as pd
from tqdm import tqdm
import requests
from urllib.parse import quote
from urllib.request import urlopen
from concurrent.futures import ThreadPoolExecutor

# ------------------- Paramete definition --------------------------

NUM_THREADS = 10
HERBARIUM_SEARCH_URL = 'http://botsad.ru/hitem/json/'
TRAIN_IMG_PATH = './train_data/'
DESIRED_GENUS = 'Rhododendron'

# --------------------------------------------------------------------

    
# --------- Modify search parameters to access all data -------------    

if  DESIRED_GENUS:
    search_parameters = (('genus', DESIRED_GENUS),) 
else:
    search_parameters = (('colend', '2019-20-20'))

# --------------------------------------------------------------------

# ------------------------ Getting data ------------------------------
search_request_url = HERBARIUM_SEARCH_URL + '?' + '&'.join(map(lambda x: x[0] + '=' + quote(x[1].strip()), search_parameters))

server_response = urlopen(search_request_url)
data = json.loads(server_response.read().decode('utf-8'))
server_response.close()

# --------------------------------------------------------------------


# -------------- Converting data to pandas dataframe -----------------
herbarium_data = pd.DataFrame(data['data'])
all_image_urls = list()
for ind, row in herbarium_data.iterrows():
    if row['images']:
        all_image_urls.append([im for im in row['images'] if '/ms/' in im][0].strip())
        



# -------------- Create dirs/ image data dowlonading -----------------
os.makedirs(TRAIN_IMG_PATH, exist_ok=True)

def downloader(url):
    response = requests.get(url)
    filename = url.split('/')[-1]
    with open(os.path.join(TRAIN_IMG_PATH, filename), "wb") as handle:
        print("Downloading file: ", filename)
        handle.write(response.content)

with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
    pool.map(downloader, all_image_urls)
# --------------------------------------------------------------------







