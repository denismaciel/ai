from __future__ import annotations

from tqdm import tqdm
import csv
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
import asyncio

from bs4 import BeautifulSoup
from typing import Any, Literal
from pyppeteer import launch
from pyppeteer.browser import Browser
from pathlib import Path
from pydantic import BaseModel

from ai.log import configure_logging, get_logger

EBAY_KLEINANZEIGE_URL = "https://www.kleinanzeigen.de/"
DATA_DIR = Path("data")
CAR_INFO_DIR = DATA_DIR / 'car_info'

CAR_NAMES = [
    'Volkswagen Caddy',
    'Mercedes-Benz V',
    'Ford Transit Custom',
    'Toyota Proace Verso',
    'Peugeot Traveller',
    'Renault Trafic Passenger',
    'Chrysler Pacifica',
    'Nissan NV',
]


configure_logging()

log = get_logger()


@dataclass
class SimpleSearchPage:
    base_url: str
    query: str
    code: Literal['k0']
    page: int

    @property
    def url(self) -> str:
        """
        https://www.kleinanzeigen.de/s-seite:3/sharan-7-sitzer/k0
        """
        return f"{self.base_url}/s-autos/s-seite:{self.page}/{self.query}/{self.code}"

    def file_prefix(self) -> str:
        return f"{self.query}__{self.page:03d}"


def iterate_search_page(base_url: str, query: str) -> Iterable[SimpleSearchPage]:
    page = 1
    while True:
        yield SimpleSearchPage(base_url, query, 'k0', page)
        page += 1


@dataclass
class CategorySearchPage:
    base_url: str
    before_page: str
    after_page: str
    page: int

    @property
    def url(self) -> str:
        if self.page < 1:
            raise ValueError("Page cannot be less than 1")

        if self.page == 1:
            return self.base_url + self.before_page + '/' + self.after_page

        return (
            self.base_url
            + self.before_page
            + '/'
            + f"seite:{self.page}"
            + '/'
            + self.after_page
        )

    def file_prefix(self) -> str:
        return f"{self.before_page}__{self.after_page}__{self.page:03d}"


SearchPage = SimpleSearchPage | CategorySearchPage


def iterate_category_search_page(
    base_url: str, before_page: str, after_page: str
) -> Iterable[CategorySearchPage]:
    page = 1
    while True:
        yield CategorySearchPage(
            base_url,
            before_page,
            after_page,
            page,
        )
        page += 1


async def fetch_search_pages(pages: Iterable[SearchPage]) -> None:
    browser = await launch(
        executablePath="/etc/profiles/per-user/denis/bin/google-chrome-stable",
        headless=False,
    )
    page = await browser.newPage()
    await page.setViewport(viewport={'width': 1500, 'height': 1240})
    await asyncio.sleep(1)

    for pq in pages:
        html_file_name = pq.file_prefix() + '.html'
        log.info('visiting', url=pq.url, file_name=html_file_name)

        await page.goto(pq.url)
        content = await page.content()
        with open(DATA_DIR / 'html' / 'raw' / html_file_name, 'w') as f:
            f.write(content)

        pagination_state = determine_pagination_state(content)

        log.info(
            'fetched page',
            url=pq.url,
            page_from_iterator=pq.page,
            page_from_html=pagination_state.current_page,
            viewable_max_page_from_html=pagination_state.viewable_max_page,
        )

        # We reached the max
        if pq.page >= pagination_state.viewable_max_page:
            log.info("reached last page")
            break

        await asyncio.sleep(1)

    input("Press enter to close the browser")
    await browser.close()


@dataclass
class VisitableUrl:
    url: str
    file: Path


async def visit_and_save_html(
    browser: Browser, urls: list[VisitableUrl], sleep: float = 0
) -> None:
    page = await browser.newPage()
    for url in tqdm(urls):
        await page.goto(url.url)
        content = await page.content()
        with open(url.file, 'w') as f:
            f.write(content)

        await asyncio.sleep(sleep)


def extract_car_infos_from_html(htmls: Iterable[Path]) -> None:
    for p in htmls:
        with open(p) as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            soup.prettify()

        # remove javascript tags
        script_tags = soup.find_all('script')
        for script_tag in script_tags:
            script_tag.decompose()

        # with open(Path("./data/html/clean/") / p.name, 'w') as f:
        #     f.write(soup.prettify())

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


def load_cars() -> Iterable[CarInfo]:
    for file in CAR_INFO_DIR.glob("*"):
        if file.is_dir():
            continue
        yield CarInfo.load(file)


def split_cars(cars: Iterable[CarInfo]) -> tuple[list[CarInfo], list[CarInfo]]:
    fetched = []
    not_fetched = []

    for car in cars:
        if car.car_details_html_file_raw.exists():
            fetched.append(car)
        else:
            not_fetched.append(car)

    return fetched, not_fetched


async def fetch_detailed_information_for_each_car():
    browser = await launch(
        executablePath="/etc/profiles/per-user/denis/bin/google-chrome-stable",
        headless=False,
    )
    cars = list(load_cars())
    fetched, not_fetched = split_cars(cars)
    log.info(
        "loaded cars",
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

    await visit_and_save_html(browser, urls, sleep=0.2)


def clean_detailed_information(car: CarInfo) -> None:
    with open(car.car_details_html_file_raw) as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    script_tags = soup.find_all('script')
    for script_tag in script_tags:
        script_tag.decompose()

    gdpr_banner = soup.find_all("div", id="gdpr-banner-container")
    for banner in gdpr_banner:
        banner.decompose()

    with open(car.car_details_html_file_clean, 'w') as f:
        f.write(soup.prettify())


def clean_detailed_information_for_each_car():
    cars = list(load_cars())

    for car in tqdm(cars):
        clean_detailed_information(car)


class AdDetailsList(BaseModel):
    marke: str = ""
    modell: str = ""
    kilometerstand: str = ""
    fahrzeugzustand: str = ""
    erstzulassung: str = ""
    kraftstoffart: str = ""
    leistung: str = ""
    getriebe: str = ""
    fahrzeugtyp: str = ""
    anzahl_tueren: str = ""
    hu_bis: str = ""
    umweltplakette: str = ""
    schadstoffklasse: str = ""
    aussenfarbe: str = ""
    material_innenausstattung: str = ""
    art: str = ""


class AdMainInfo(BaseModel):
    price: str
    title: str
    locality: str


class CarDetailPage(BaseModel):
    ad_details_list: AdDetailsList
    ad_main_info: AdMainInfo


def extract_detailed_information(car: CarInfo) -> CarDetailPage | None:
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

    title = remove_line_breaks(soup.select_one("#viewad-title").text.strip())
    price = soup.select_one("#viewad-price").text.strip()
    locality = soup.select_one("#viewad-locality").text.strip()

    ad_main_info = AdMainInfo(
        title=title,
        price=price,
        locality=locality,
    )

    (details,) = soup.find_all('div', id="viewad-details")
    details_dict = {}
    for detail in details.select('.addetailslist--detail'):
        key = detail.contents[0].text.strip()
        value = detail.select_one('.addetailslist--detail--value').text.strip()
        if key in keys_mapping:
            details_dict[keys_mapping[key]] = value
        else:
            log.warning("unhandled car detail key", key=key, value=value)

    ad_details_list = AdDetailsList.model_validate(details_dict)
    return CarDetailPage(
        ad_main_info=ad_main_info,
        ad_details_list=ad_details_list,
    )


def extract_detailed_information_for_each_car():
    cars = list(load_cars())

    details = []
    for car in tqdm(cars):
        try:
            detail = extract_detailed_information(car)
            details.append(detail)
        except Exception as e:
            log.exception(
                "extract_detailed_information",
                exception=e,
                file=car.car_details_html_file_clean,
            )
    save_to_csv([flatten_car_detail_page(d) for d in details], DATA_DIR / 'output.csv')


def flatten_car_detail_page(car_detail_page: CarDetailPage) -> dict[str, Any]:
    return {
        'marke': car_detail_page.ad_details_list.marke,
        'modell': car_detail_page.ad_details_list.modell,
        'kilometerstand': car_detail_page.ad_details_list.kilometerstand,
        'fahrzeugzustand': car_detail_page.ad_details_list.fahrzeugzustand,
        'erstzulassung': car_detail_page.ad_details_list.erstzulassung,
        'kraftstoffart': car_detail_page.ad_details_list.kraftstoffart,
        'leistung': car_detail_page.ad_details_list.leistung,
        'getriebe': car_detail_page.ad_details_list.getriebe,
        'fahrzeugtyp': car_detail_page.ad_details_list.fahrzeugtyp,
        'anzahl_tueren': car_detail_page.ad_details_list.anzahl_tueren,
        'hu_bis': car_detail_page.ad_details_list.hu_bis,
        'umweltplakette': car_detail_page.ad_details_list.umweltplakette,
        'schadstoffklasse': car_detail_page.ad_details_list.schadstoffklasse,
        'aussenfarbe': car_detail_page.ad_details_list.aussenfarbe,
        'material_innenausstattung': car_detail_page.ad_details_list.material_innenausstattung,
        'art': car_detail_page.ad_details_list.art,
        'price': car_detail_page.ad_main_info.price,
        'title': car_detail_page.ad_main_info.title,
        'locality': car_detail_page.ad_main_info.locality,
    }


def save_to_csv(l: list[dict[str, Any]], file: Path) -> None:
    if len(l) == 0:
        log.warning('list is empty')

    columns = l[0].keys()
    with open(file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        writer.writerows(l)


def remove_line_breaks(s: str) -> str:
    return ' '.join(s.split())


@dataclass
class PaginationState:
    # The highest page number that can be viewed in the bottom pagination menu
    viewable_max_page: int

    # Current page that can be extracted with CSS selectors
    current_page: int


def determine_pagination_state(html: str) -> PaginationState:
    soup = BeautifulSoup(html, 'html.parser')
    viewable_max_page = max(
        int(int(div.text)) for div in soup.select(".pagination-page")
    )
    (current_page,) = soup.select(".pagination-current")
    return PaginationState(
        viewable_max_page=viewable_max_page, current_page=int(current_page.text)
    )


if __name__ == "__main__":
    # 1. Fetch data fro
    # it = iterate_category_search_page(
    #     base_url=EBAY_KLEINANZEIGE_URL,
    #     before_page='s-auto',
    #     after_page='c216+autos.typ_s:bus',
    # )
    # asyncio.get_event_loop().run_until_complete(fetch_search_pages(it))

    # 2.From each page extract the summary of the car
    # extract_car_infos_from_html(Path("./data/html/raw/").glob("*"))

    # 3. Extract information from each car
    # asyncio.get_event_loop().run_until_complete(
    #     fetch_detailed_information_for_each_car()
    # )

    # 4. Clean html for car details
    clean_detailed_information_for_each_car()

    # 5. Extract deatils
    extract_detailed_information_for_each_car()
