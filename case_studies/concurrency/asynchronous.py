import asyncio
import aiohttp


async def download_site(session, url):
    async with session.get(url) as response:
        print(f'Read {response.content_length} from {url}')


async def download_all_sites(sites):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in sites:
            task = asyncio.ensure_future(download_site(session, url))
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)


def run_async(sites):
    asyncio.get_event_loop().run_until_complete(download_all_sites(sites))
