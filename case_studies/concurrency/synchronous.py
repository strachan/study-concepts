import requests


def download_site(url, session):
    with session.get(url) as response:
        print(f'Read {len(response.content)} from {url}')


def download_all_sites(sites):
    with requests.Session() as session:
        for i, url in enumerate(sites):
            download_site(url, session)
