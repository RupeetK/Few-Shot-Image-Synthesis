

import os
import requests
from io import BytesIO
from PIL import Image
from multiprocessing.pool import ThreadPool
from tqdm.notebook import tqdm
get_ipython().system('pip install imagehash')
import imagehash
import time
import random
from threading import Lock


# API key for Pexels
pexels_key='PEXELS_KEY'

categories={
    'pizza': 'pizza', # Search query for pizza images
    'sushi': 'sushi platter',
    'pasta':'pasta dish'
}

target=100 # Images per category
img_size=1024
save_root='Food Dataset' # Root folder
threads=10 # Thread count
lock=Lock() # Shared lock for hash set

os.makedirs(save_root,exist_ok=True)

# Fetch image URLs from Pexels
def fetch_img_urls(query):
    headers={'Authorization': pexels_key}
    page = random.randint(1, 50) # Randomize page for variation
    url=f'https://api.pexels.com/v1/search?query={query}&per_page=50&page={page}'
    try:
        data=requests.get(url,headers=headers,timeout=10).json()
        # Extract high‑res URLs if available
        if 'photos' in data and len(data['photos'])>0:
            return [photo['src']['large2x'] for photo in data['photos']]
    except Exception as e:
        print('Error fetching image:', e)
        return []
    return []


# Download image + validate (size + dedupe)
def download_validate(args):
    url,save_path,existing_hashes=args

    try:
        response=requests.get(url,timeout=10)
        img=Image.open(BytesIO(response.content))

        # Skip small images
        if img.width<img_size or img.height<img_size:
            return False,None

        # Compute perceptual hash for deduplication
        img_hash=imagehash.phash(img)
        
        # Ensure uniqueness across threads
        with lock:
            if img_hash in existing_hashes:
                return False,None
            existing_hashes.add(img_hash)

        # Resize + save
        img=img.resize((img_size,img_size), Image.LANCZOS)
        img.save(save_path)
        return True, save_path
        
    except:
        return False,None


# Build dataset for a single category
def build_category(cat_name,query):
    save_dir=os.path.join(save_root,cat_name)
    os.makedirs(save_dir,exist_ok=True)

    saved=0
    pool=ThreadPool(threads) # Threaded downloader pool
    existing_hashes=set() # Track unique images

    print(f'Downloading {cat_name} dataset...')

    while saved<target:
        urls = []

        # Fetch multiple batches concurrently
        for _ in range(threads):
            urls.extend(fetch_img_urls(query))  
        urls = [u for u in urls if u]
        jobs=[]

        if not urls:
            time.sleep(1)
            continue

        temp_counter = saved

        # Assign filenames and prepare jobs
        for url in urls:
            if url:
                filename=os.path.join(save_dir,f'{cat_name}_{temp_counter:03d}.jpg')
                jobs.append((url,filename,existing_hashes))
                temp_counter+=1
               
        if not jobs:  
            time.sleep(1)
            continue

        # Run downloads in parallel
        results=list(tqdm(pool.imap_unordered(download_validate,jobs),total=len(jobs)))

        # Extract valid saved images
        valid=[filepath for success,filepath in results if success]

        # Rename sequentially
        for filepath in valid:
            if saved<target:
                new_filename=os.path.join(save_dir,f'{cat_name}_{saved:03d}.jpg')
                if new_filename!=filepath: os.rename(filepath,new_filename)
                saved+=1

       
        print(f'{cat_name}: saved {saved}/{target}')
        time.sleep(1)
    
    pool.close()
    pool.join()

        
    

# Build all categories
for folder,query in categories.items():
    build_category(folder,query)

print('Dataset is ready at: ', save_root)





