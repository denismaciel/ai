from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from dataclasses import dataclass
from multiprocessing import cpu_count
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import httpx
import instructor
from ai.log import configure_logging
from ai.log import get_logger
from bs4 import BeautifulSoup
from openai import OpenAI
from pydantic import BaseModel
from pyppeteer import launch
from pyppeteer.browser import Browser
from tqdm.asyncio import tqdm

from ai.car.models import CarInfo
from ai.car.models import EBAY_KLEINANZEIGE_URL
from ai.car.models import CarInfo

from ai.car.models import DATA_DIR
from ai.car.models import SEARCH_PAGES_HTML_RAW
from ai.car.utils import load_cars

from ai.car.parse_car_details import extract_detailed_information_for_each_car_parallel

configure_logging()
log = get_logger()

httpxc = httpx.Client(follow_redirects=True)


class CarMark(BaseModel):
    mark: str
    count: int
    href: str


class CarMarkContainer(BaseModel):
    items: list[CarMark]


class CarModel(BaseModel):
    mark: str
    model: str
    count: int
    href: str

    @property
    def url(self) -> str:
        return EBAY_KLEINANZEIGE_URL + self.href


class CarModelContainer(BaseModel):
    car_models: list[CarModel]
    """
    All models should be from the same mark
    """
    mark: str


class CategorySearchPage(BaseModel):
    base_url: str
    before_page: str
    after_page: str
    page: int

    @property
    def url(self) -> str:
        if self.page < 1:
            raise ValueError('Page cannot be less than 1')

        if self.page == 1:
            return self.base_url + self.before_page + '/' + self.after_page

        return (
            self.base_url
            + self.before_page
            + '/'
            + f'seite:{self.page}'
            + '/'
            + self.after_page
        )

    def file_prefix(self) -> str:
        return (
            f'{self.before_page.replace("/", "__")}__{self.after_page}__{self.page:03d}'
        )

    @classmethod
    def from_car_model(cls, car_model: CarModel) -> CategorySearchPage:
        split = car_model.href.split('/')

        return CategorySearchPage(
            base_url=EBAY_KLEINANZEIGE_URL,
            before_page='/'.join(split[:-1]),
            after_page=split[-1],
            page=1,
        )


SearchPage = CategorySearchPage


def fetch_car_marks_from_menu():
    client = instructor.from_openai(OpenAI())
    resp = httpxc.get('https://www.kleinanzeigen.de/s-autos/c216+autos.typ_s:bus')
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    for section in soup.select('section.browsebox-attribute'):
        h2 = section.select_one('h2')
        assert h2
        if 'Marke' in h2.text:
            break
    else:
        raise ValueError('Marke subsection not found')

    prompt = f"""
    Extract all the car marks from the html:

    {section.contents}
    """

    marks = client.chat.completions.create(
        model='gpt-3.5-turbo',
        response_model=CarMarkContainer,
        messages=[{'role': 'user', 'content': prompt}],
    )

    with open('car_marks.json', 'w') as f:
        f.write(marks.model_dump_json())


def fetch_search_pages_for_car_models():
    client = instructor.from_openai(OpenAI())
    with open('car_marks.json') as f:
        marks = CarMarkContainer.model_validate_json(f.read())

    for mark in marks.items:
        log.info('fetching car models', mark=mark.mark)
        resp = httpxc.get(EBAY_KLEINANZEIGE_URL + mark.href)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'html.parser')
        for section in soup.select('section.browsebox-attribute'):
            h2 = section.select_one('h2')
            assert h2
            if 'Marke' in h2.text:
                break
        else:
            raise ValueError('Marke subsection not found')

        prompt = f"""
        Extract all the car models from the html:

        {section.contents}
        """

        models = client.chat.completions.create(
            model='gpt-3.5-turbo',
            response_model=CarModelContainer,
            messages=[{'role': 'user', 'content': prompt}],
        )

        log.info('fetched car models', mark=models.mark, models=models.car_models)

        with open(
            f'car_models_for_{models.mark.lower().replace(" ", "-")}.json', 'w'
        ) as f:
            f.write(models.model_dump_json())


def load_car_models_search_pages() -> list[CarModel]:
    models = []
    for f in (DATA_DIR / 'car_models').glob('*.json'):
        cms = CarModelContainer.model_validate_json(f.read_text())
        models.extend(cms.car_models)
    return models


def fetch_search_pages_httpx(pages: Iterable[SearchPage]) -> None:
    for pq in pages:
        html_file_name = pq.file_prefix() + '.html'
        log.info('visiting', url=pq.url, file_name=html_file_name)
        resp = httpxc.get(pq.url)
        resp.raise_for_status()
        page_content = resp.text

        with open(SEARCH_PAGES_HTML_RAW / html_file_name, 'w') as f:
            f.write(page_content)

        pagination_state = determine_pagination_state(page_content)

        log.info(
            'fetched page',
            url=pq.url,
            page_from_iterator=pq.page,
            page_from_html=pagination_state.current_page,
            viewable_max_page_from_html=pagination_state.viewable_max_page,
        )

        if pq.page >= pagination_state.viewable_max_page:
            log.info('reached last page')
            break


_BROWSER: Browser | None = None


async def get_browser() -> Browser:
    global _BROWSER
    if _BROWSER is None:
        _BROWSER = await launch(
            executablePath='/etc/profiles/per-user/denis/bin/google-chrome-stable',
            headless=False,
        )
    return _BROWSER


async def fetch_search_pages_pyppeteer(pages: Iterable[SearchPage]) -> None:
    browser = await get_browser()
    page = await browser.newPage()
    await page.setViewport(viewport={'width': 1500, 'height': 1240})
    await asyncio.sleep(1)

    for pq in pages:
        while True:
            html_file_name = pq.file_prefix() + '.html'
            log.info('visiting', url=pq.url, file_name=html_file_name)

            await page.goto(pq.url)
            content = await page.content()
            with open(SEARCH_PAGES_HTML_RAW / html_file_name, 'w') as f:
                f.write(content)

            pagination_state = determine_pagination_state(content)

            log.info(
                'fetched page',
                url=pq.url,
                page_from_iterator=pq.page,
                page_from_html=pagination_state.current_page,
                viewable_max_page_from_html=pagination_state.viewable_max_page,
            )

            if pq.page >= pagination_state.viewable_max_page:
                log.info('reached last page')
                break

            pq.page += 1

            await asyncio.sleep(1)

    input('Press enter to close the browser')
    await browser.close()


@dataclass
class PaginationState:
    # The highest page number that can be viewed in the bottom pagination menu
    viewable_max_page: int

    # Current page that can be extracted with CSS selectors
    current_page: int


def determine_pagination_state(html: str) -> PaginationState:
    soup = BeautifulSoup(html, 'html.parser')
    # All the pages except the current one.
    # The current has .pagination-current as a CSS class
    non_current_pages = [int(int(div.text)) for div in soup.select('.pagination-page')]

    if len(non_current_pages) > 0:
        viewable_max_page = max(non_current_pages)
    else:
        viewable_max_page = 1
    (current_page,) = soup.select('.pagination-current')
    return PaginationState(
        viewable_max_page=viewable_max_page, current_page=int(current_page.text)
    )


async def fetch_search_pages():
    car_models = load_car_models_search_pages()
    search_pages = []
    for car_model in car_models:
        sp = CategorySearchPage.from_car_model(car_model)
        search_pages.append(sp)

    await fetch_search_pages_pyppeteer(search_pages)


def extract_car_info_article(soup: Any) -> CarInfo:
    title = soup.find('a', class_='ellipsis').text.strip()
    location = soup.find('div', class_='aditem-main--top--left').text.strip()
    description = soup.find('p', class_='aditem-main--middle--description').text.strip()
    price = soup.find(
        'p', class_='aditem-main--middle--price-shipping--price'
    ).text.strip()

    year = None
    mileage = soup.find('span', class_='simpletag')
    if mileage is not None:
        mileage = mileage.text.strip()
        # year = soup.find_all('span', class_='simpletag')[1].text.strip()
    details_link = soup.get('data-href')

    car_info = CarInfo(
        title=title,
        location=location,
        description=description,
        price=price,
        mileage=mileage,
        year=year,  # TODO: remove it from here
        details_link=details_link,
    )
    return car_info


def extract_car_infos_from_html(htmls: Iterable[Path]) -> None:
    htmls = list(htmls)
    for p in tqdm(htmls):
        with open(p) as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            soup.prettify()

        # remove javascript tags
        script_tags = soup.find_all('script')
        for script_tag in script_tags:
            script_tag.decompose()

        cars = [extract_car_info_article(car) for car in soup.find_all('article')]

        for car in cars:
            car.save()


def split_cars(cars: Iterable[CarInfo]) -> tuple[list[CarInfo], list[CarInfo]]:
    fetched = []
    not_fetched = []

    for car in cars:
        if car.car_details_html_file_raw.exists():
            fetched.append(car)
        else:
            not_fetched.append(car)

    return fetched, not_fetched


@dataclass
class VisitableUrl:
    url: str
    file: Path


async def fetch_detailed_information_for_each_car() -> None:
    cars = list(load_cars())
    fetched, not_fetched = split_cars(cars)
    log.info(
        'loaded cars',
        total=len(cars),
        fetche=len(fetched),
        not_fetched=len(not_fetched),
    )

    urls = [
        VisitableUrl(
            url=car.details_url,
            file=car.car_details_html_file_raw,
        )
        for car in not_fetched
    ]

    await visit_and_save_html_httpx(urls, max_requests_per_second=30)


def visit_and_save_html_httpx_sync(urls: list[VisitableUrl], sleep: float = 0) -> None:
    for url in tqdm(urls):
        resp = httpxc.get(url.url)
        resp.raise_for_status()
        with open(url.file, 'w') as f:
            f.write(resp.text)
        time.sleep(sleep)


def clean_detailed_information(car: CarInfo) -> None:
    with open(car.car_details_html_file_raw) as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    script_tags = soup.find_all('script')
    for script_tag in script_tags:
        script_tag.decompose()

    gdpr_banner = soup.find_all('div', id='gdpr-banner-container')
    for banner in gdpr_banner:
        banner.decompose()

    with open(car.car_details_html_file_clean, 'w') as f:
        f.write(soup.prettify())


def clean_detailed_information_car(car):
    if car.car_details_html_file_raw.exists():
        clean_detailed_information(car)


def clean_detailed_information_for_each_car():
    cars = list(load_cars())

    # Input are available
    cars = [c for c in cars if c.car_details_html_file_raw.exists()]

    # Output hasn't yet been processed
    cars = [c for c in cars if not c.car_details_html_file_clean.exists()]

    # Suggest using Pool's context manager for better resource management
    with Pool(processes=cpu_count()) as pool:
        # Use tqdm.starmap to display progress bar for parallel processing
        list(tqdm(pool.imap(clean_detailed_information_car, cars), total=len(cars)))


# def clean_detailed_information_for_each_car():
#     cars = list(load_cars())
#     cars = [c for c in cars if c.car_details_html_file_raw.exists()]
#
#     for car in tqdm(cars):
#         clean_detailed_information(car)


async def fetch_save(url: VisitableUrl, semaphore: asyncio.Semaphore):
    # Acquire the semaphore before making the HTTP request
    async with semaphore:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            try:
                resp = await client.get(url.url)
                resp.raise_for_status()
                with open(url.file, 'w') as f:
                    f.write(resp.text)
            except Exception as e:
                log.warning('an exception ocurred', e=str(e))


async def visit_and_save_html_httpx(
    urls: list[VisitableUrl], max_requests_per_second: int
) -> None:
    # Create a semaphore with a limit on concurrent requests
    semaphore = asyncio.Semaphore(max_requests_per_second)

    # Setting a pause interval to avoid exceeding the rate limit
    interval = 1 / max_requests_per_second

    async def throttled_fetch_save(
        url: VisitableUrl, semaphore: asyncio.Semaphore, interval: float
    ):
        await fetch_save(url, semaphore)
        await asyncio.sleep(interval)

    tasks = [throttled_fetch_save(url, semaphore, interval) for url in urls]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await f


def main():
    asyncio.get_event_loop()

    # Get page
    # 1. Crawl search pages
    # event_loop.run_until_complete(fetch_search_pages())

    # Extract cars info from search page HTML
    # extract_car_infos_from_html(SEARCH_PAGES_HTML_RAW.glob("*"))

    # 3. Extract information from each car
    # event_loop.run_until_complete(fetch_detailed_information_for_each_car())

    # 4. Clean detailed information
    # clean_detailed_information_for_each_car()
    # 5. Extract
    extract_detailed_information_for_each_car_parallel()


if __name__ == '__main__':
    main()
