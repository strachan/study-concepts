import concurrent.futures
import requests
import threading


thread_local = threading.local()


def get_session():
    if not getattr(thread_local, 'session', None):
        thread_local.session = requests.Session()
    return thread_local.session


def download_site(url):
    session = get_session()
    with session.get(url) as response:
        print(f'Read {len(response.content)} from {url}')


def download_all_sites(sites):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(download_site, sites)
