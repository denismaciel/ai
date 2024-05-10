from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
import asyncio

from bs4 import BeautifulSoup
from typing import Any
from pyppeteer import launch
from pyppeteer.browser import Browser
from pathlib import Path
from pydantic import BaseModel

from ai.log import configure_logging, get_logger

EBAY_KLEINANZEIGE_URL = "https://www.kleinanzeigen.de"
CAR_NAME = 'sharan 7 sitzer'
DATA_DIR = Path("data")
CAR_INFO_DIR = DATA_DIR / 'car_info'
N_PAGES_TO_SEARCH = 10

configure_logging()

log = get_logger()


@dataclass
class SearchQuery:
    query: str
    code: str


@dataclass
class PaginationQuery:
    search_query: SearchQuery
    page: int

    @classmethod
    def from_search_query(cls, s: SearchQuery, page: int) -> PaginationQuery:
        return PaginationQuery(
            search_query=s,
            page=page,
        )

    def file_prefix(self) -> str:
        return f"{self.search_query.query}__{self.page:03d}"


def parse_search_query(url: str) -> SearchQuery:
    split = [part for part in url.split("/") if part]
    return SearchQuery(query=split[-2].removeprefix("s-"), code=split[-1])


def render_url_with_pagination_query(q: PaginationQuery) -> str:
    """
    https://www.kleinanzeigen.de/s-seite:3/sharan-7-sitzer/k0
    """
    return f"{EBAY_KLEINANZEIGE_URL}/s-seite:{q.page}/{q.search_query.query}/{q.search_query.code}"


async def fetch_search_pages(query: str, pages: int) -> None:
    browser = await launch(
        executablePath="/etc/profiles/per-user/denis/bin/google-chrome-stable",
        headless=False,
    )
    page = await browser.newPage()
    await page.setViewport(viewport={'width': 1500, 'height': 1240})
    await page.goto(EBAY_KLEINANZEIGE_URL)

    await page.type('#site-search-query', query)
    # await page.type('#site-search-area', '21033')
    await page.keyboard.press('Enter')
    await page.waitForNavigation({'timeout': 10000})
    log.info("parsing page url after search submission", url=page.url)
    sq = parse_search_query(page.url)

    for page_nr in range(2, 2 + pages):
        pq = PaginationQuery.from_search_query(sq, page_nr)
        next_url = render_url_with_pagination_query(pq)
        html_file_name = pq.file_prefix() + '.html'
        log.info('visiting', url=next_url, file_name=html_file_name)

        await page.goto(next_url)
        content = await page.content()
        with open(DATA_DIR / html_file_name, 'w') as f:
            f.write(content)

    input("Press enter to close the browser")
    await browser.close()


@dataclass
class VisitableUrl:
    url: str
    file: Path


async def visit_and_save_html(browser: Browser, urls: Iterable[VisitableUrl]) -> None:
    page = await browser.newPage()
    for url in urls:
        await page.goto(url.url)
        content = await page.content()
        with open(url.file, 'w') as f:
            f.write(content)


def extract_car_infos_from_html(htmls: Iterable[Path]) -> None:
    for p in htmls:
        with open(p) as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            soup.prettify()

        # remove javascript tags
        script_tags = soup.find_all('script')
        for script_tag in script_tags:
            script_tag.decompose()

        with open(Path("./data/html/clean/") / p.name, 'w') as f:
            f.write(soup.prettify())

        cars = []
        for car in soup.find_all('article'):
            cars.append(extract_car_info_article(car))

        for car in cars:
            car.save()


class CarInfo(BaseModel):
    title: str
    location: str
    description: str
    price: str
    mileage: str | None
    year: str | None
    details_link: str

    @property
    def uid(self) -> str:
        hash_object = hashlib.sha256(self.details_link.encode())
        return hash_object.hexdigest()

    def save(self) -> None:
        file_name = self.uid + '.json'
        with open(CAR_INFO_DIR / file_name, 'w') as f:
            f.write(self.model_dump_json())

    @classmethod
    def load(cls, json_str: str) -> CarInfo:
        return cls.model_validate_json(json_str)

    @property
    def details_url(self) -> str:
        return EBAY_KLEINANZEIGE_URL + f'/{self.details_link}'


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
        year = soup.find_all('span', class_='simpletag')[1].text.strip()
    details_link = soup.get('data-href')

    car_info = CarInfo(
        title=title,
        location=location,
        description=description,
        price=price,
        mileage=mileage,
        year=year,
        details_link=details_link,
    )
    return car_info


def load_cars() -> Iterable[CarInfo]:
    for file in CAR_INFO_DIR.glob("*"):
        if file.is_dir():
            continue
        yield CarInfo.load(file.read_text())


async def extract_detailed_information_for_each_car():
    browser = await launch(
        executablePath="/etc/profiles/per-user/denis/bin/google-chrome-stable",
        headless=False,
    )
    cars = load_cars()
    urls = []

    for car in cars:
        vu = VisitableUrl(
            url=car.details_url,
            file=DATA_DIR / 'car_detail' / (car.uid + '.html'),
        )
        urls.append(vu)

    await visit_and_save_html(browser, urls)


if __name__ == "__main__":
    # 1. Fetch data fro
    # asyncio.get_event_loop().run_until_complete( fetch_search_pages(CAR_NAME, N_PAGES_TO_SEARCH))

    # 2.From each page extract the summary of the car
    # extract_car_infos_from_html(Path("./data/html/raw/").glob("*"))

    # 3. Extract information from each car
    asyncio.get_event_loop().run_until_complete(
        extract_detailed_information_for_each_car()
    )
