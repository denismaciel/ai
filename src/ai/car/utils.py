from ai.car.models import CarInfo
from ai.car.models import CAR_INFO_DIR
from typing import Iterable

def load_cars() -> Iterable[CarInfo]:
    for file in CAR_INFO_DIR.glob('*'):
        if file.is_dir():
            continue
        yield CarInfo.load(file)
