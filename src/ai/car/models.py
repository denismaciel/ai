from __future__ import annotations
import hashlib
from pydantic import BaseModel
from dataclasses import dataclass

from pathlib import Path


EBAY_KLEINANZEIGE_URL = 'https://www.kleinanzeigen.de/'
DATA_DIR = Path('data')
CAR_INFO_DIR = DATA_DIR / 'car_info'
SEARCH_PAGES_HTML_RAW = DATA_DIR / 'search_pages' / 'raw'


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
