from __future__ import annotations

import asyncio
import csv
import hashlib
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
from tqdm import tqdm
# from tqdm.asyncio import tqdm

EBAY_KLEINANZEIGE_URL = 'https://www.kleinanzeigen.de/'
DATA_DIR = Path('data')
CAR_INFO_DIR = DATA_DIR / 'car_info'
SEARCH_PAGES_HTML_RAW = DATA_DIR / 'search_pages' / 'raw'

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
            f'{self.before_page.replace('/', '__')}__{self.after_page}__{self.page:03d}'
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
    def load(cls, file: Path) -> CarInfo:
        return cls.model_validate_json(file.read_text())

    @property
    def details_url(self) -> str:
        return EBAY_KLEINANZEIGE_URL + f'/{self.details_link}'

    @property
    def car_details_html_file_raw(self) -> Path:
        return DATA_DIR / 'car_detail' / 'html' / 'raw' / (self.uid + '.html')

    @property
    def car_details_html_file_clean(self) -> Path:
        return DATA_DIR / 'car_detail' / 'html' / 'clean' / (self.uid + '.html')


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


def load_cars() -> Iterable[CarInfo]:
    for file in CAR_INFO_DIR.glob('*'):
        if file.is_dir():
            continue
        yield CarInfo.load(file)


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


class AdDetailsList(BaseModel):
    marke: str = ''
    modell: str = ''
    kilometerstand: str = ''
    fahrzeugzustand: str = ''
    erstzulassung: str = ''
    kraftstoffart: str = ''
    leistung: str = ''
    getriebe: str = ''
    fahrzeugtyp: str = ''
    anzahl_tueren: str = ''
    hu_bis: str = ''
    umweltplakette: str = ''
    schadstoffklasse: str = ''
    aussenfarbe: str = ''
    material_innenausstattung: str = ''
    art: str = ''


@dataclass
class CarParsedInfo:
    km: int
    horsepower: int
    year: int
    title_parsed: str
    plz: int
    # price_parsed: float


class AdMainInfo(BaseModel):
    price: str
    title: str
    locality: str


class CarDetailPage(BaseModel):
    ad_details_list: AdDetailsList
    ad_main_info: AdMainInfo


class CarDetailPageWithParsedInfo(BaseModel):
    url: str
    ad_details_list: AdDetailsList
    ad_main_info: AdMainInfo
    parsed_info: CarParsedInfo


def remove_line_breaks(s: str) -> str:
    return ' '.join(s.split())


def extract_detailed_information(car: CarInfo) -> CarDetailPage:
    keys_mapping = {
        'Marke': 'marke',
        'Modell': 'modell',
        'Kilometerstand': 'kilometerstand',
        'Fahrzeugzustand': 'fahrzeugzustand',
        'Erstzulassung': 'erstzulassung',
        'Kraftstoffart': 'kraftstoffart',
        'Leistung': 'leistung',
        'Getriebe': 'getriebe',
        'Fahrzeugtyp': 'fahrzeugtyp',
        'Anzahl Türen': 'anzahl_tueren',
        'HU bis': 'hu_bis',
        'Umweltplakette': 'umweltplakette',
        'Schadstoffklasse': 'schadstoffklasse',
        'Außenfarbe': 'aussenfarbe',
        'Material Innenausstattung': 'material_innenausstattung',
        'Art': 'art',
    }

    with open(car.car_details_html_file_clean) as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    title = remove_line_breaks(soup.select_one('#viewad-title').text.strip())
    price = soup.select_one('#viewad-price').text.strip()
    locality = soup.select_one('#viewad-locality').text.strip()

    ad_main_info = AdMainInfo(
        title=title,
        price=price,
        locality=locality,
    )

    (details,) = soup.find_all('div', id='viewad-details')
    details_dict = {}
    for detail in details.select('.addetailslist--detail'):
        key = detail.contents[0].text.strip()
        value = detail.select_one('.addetailslist--detail--value').text.strip()
        if key in keys_mapping:
            details_dict[keys_mapping[key]] = value
        else:
            log.warning('unhandled car detail key', key=key, value=value)

    ad_details_list = AdDetailsList.model_validate(details_dict)
    return CarDetailPage(
        ad_main_info=ad_main_info,
        ad_details_list=ad_details_list,
    )


def parse_car_details(car: CarInfo):
    try:
        detail = extract_detailed_information(car)
    except Exception as e:
        log.warning(
            'extract_detailed_information',
            exception=str(e),
            file=car.car_details_html_file_clean,
        )
        return

    if detail.ad_details_list.art.strip() != '':
        return
    if detail.ad_details_list.kilometerstand == '':
        return

    try:
        return CarDetailPageWithParsedInfo(
            url=car.details_url,
            ad_details_list=detail.ad_details_list,
            ad_main_info=detail.ad_main_info,
            parsed_info=parse_car_info(detail.ad_main_info, detail.ad_details_list),
        )
    except Exception as e:
        log.warning('parsing car details failed', exception=str(e))


def extract_detailed_information_for_each_car():
    cars = list(load_cars())

    cars = [c for c in cars if c.car_details_html_file_clean.exists()]

    parsed_details: list[CarDetailPageWithParsedInfo] = []
    for car in tqdm(cars):
        parsed = parse_car_details(car)
        if parsed:
            parsed_details.append(parsed)

    save_to_csv(
        [flatten_car_detail_page(d) for d in parsed_details], DATA_DIR / 'output.csv'
    )


def remove_non_numbers(s: str) -> str:
    return ''.join(char for char in s if char.isdigit())


def parse_car_info(
    ad_main_info: AdMainInfo, ad_details_list: AdDetailsList
) -> CarParsedInfo:
    """
    kilometerstand -> int: remove letters, remove dot, coerce int
    leistung -> int: remove letters, coerce int
    erstzulassung -> year -> extract numbers from string, coerce int
    title -> remove "Reserviert • Gelöscht • "
    plz -> first five numbers from locality
    city -> ...
    """
    km = int(remove_non_numbers(ad_details_list.kilometerstand))

    if ad_details_list.leistung == '':
        horsepower = 0
    else:
        horsepower = int(remove_non_numbers(ad_details_list.leistung))

    year = int(remove_non_numbers(ad_details_list.erstzulassung))
    title = ad_main_info.title.replace('Reserviert • Gelöscht • ', '')
    plz = int(ad_main_info.locality.strip()[:5])

    return CarParsedInfo(
        km=km,
        horsepower=horsepower,
        year=year,
        title_parsed=title,
        plz=plz,
    )


def flatten_car_detail_page(parsed: CarDetailPageWithParsedInfo) -> dict[str, Any]:
    return {
        'marke': parsed.ad_details_list.marke,
        'modell': parsed.ad_details_list.modell,
        'km': parsed.parsed_info.km,
        'horsepower': parsed.parsed_info.horsepower,
        'year': parsed.parsed_info.year,
        'title_parsed': parsed.parsed_info.title_parsed,
        'plz': parsed.parsed_info.plz,
        'url': parsed.url,
        'kilometerstand': parsed.ad_details_list.kilometerstand,
        'fahrzeugzustand': parsed.ad_details_list.fahrzeugzustand,
        'erstzulassung': parsed.ad_details_list.erstzulassung,
        'kraftstoffart': parsed.ad_details_list.kraftstoffart,
        'leistung': parsed.ad_details_list.leistung,
        'getriebe': parsed.ad_details_list.getriebe,
        'fahrzeugtyp': parsed.ad_details_list.fahrzeugtyp,
        'anzahl_tueren': parsed.ad_details_list.anzahl_tueren,
        'hu_bis': parsed.ad_details_list.hu_bis,
        'umweltplakette': parsed.ad_details_list.umweltplakette,
        'schadstoffklasse': parsed.ad_details_list.schadstoffklasse,
        'aussenfarbe': parsed.ad_details_list.aussenfarbe,
        'material_innenausstattung': parsed.ad_details_list.material_innenausstattung,
        'art': parsed.ad_details_list.art,
        'price': parsed.ad_main_info.price,
        'title': parsed.ad_main_info.title,
        'locality': parsed.ad_main_info.locality,
    }


def save_to_csv(l: list[dict[str, Any]], file: Path) -> None:
    if len(l) == 0:
        log.warning('list is empty')

    columns = l[0].keys()
    with open(file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        writer.writerows(l)


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
    extract_detailed_information_for_each_car()


if __name__ == '__main__':
    main()
