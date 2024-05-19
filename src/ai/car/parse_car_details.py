from __future__ import annotations
from ai.car.utils import load_cars
from pathlib import Path
import csv
from ai.car.models import (
    CarInfo,
    CarDetailPage,
    AdMainInfo,
    AdDetailsList,
    CarDetailPageWithParsedInfo,
    CarParsedInfo,
    DATA_DIR,
)
from tqdm import tqdm
from typing import Any
from bs4 import BeautifulSoup


from multiprocessing import Queue, Process, cpu_count
from dataclasses import dataclass
import queue
from ai.log import get_logger

log = get_logger()


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


def remove_non_numbers(s: str) -> str:
    return ''.join(char for char in s if char.isdigit())


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


# =============================
# ------ Parallel stuff -------
# =============================


@dataclass
class TaskPayload:
    car_info: CarInfo


@dataclass
class TaskResult:
    parsed_car: CarDetailPageWithParsedInfo | None


def extract_detailed_information_for_each_car_parallel():
    N_WORKERS = cpu_count() - 2
    cars = list(load_cars())
    cars = [c for c in cars if c.car_details_html_file_clean.exists()]
    num_tasks = len(cars)

    tasks: Queue[TaskPayload] = Queue()
    results: Queue[TaskResult] = Queue()

    for car in cars:
        tasks.put(TaskPayload(car))

    workers: list[Process] = []
    for _ in range(N_WORKERS):
        p = Process(target=do_work, args=(tasks, results))
        p.start()
        workers.append(p)

    parsed_cars: list[CarDetailPageWithParsedInfo | None] = []
    # Monitor progress
    completed_tasks = 0
    progress_bar = tqdm(total=num_tasks)
    while completed_tasks < num_tasks:
        result = results.get()
        completed_tasks += 1
        progress_bar.update(1)
        parsed_cars.append(result.parsed_car)
    progress_bar.close()


    parsed_cars_no_nones = [car for car in parsed_cars if car]

    save_to_csv(
        [flatten_car_detail_page(d) for d in parsed_cars_no_nones], DATA_DIR / 'output.csv'
    )


def do_work(tasks: Queue[TaskPayload], results: Queue[TaskResult]) -> None:
    while True:
        try:
            task = tasks.get(block=True, timeout=0.2)
        except queue.Empty:
            break

        parsed = parse_car_details(task.car_info)
        results.put(TaskResult(parsed))
