import pytest
from ai.car.main import parse_price
from ai.car.main import Price


@pytest.mark.parametrize(
    'price_string, expected_price',
    [
        ('5.900 € VB', Price(currency='EUR', is_negotiable=True, value=5900.0)),
        ('11.80€', Price(currency='EUR', is_negotiable=False, value=11.8)),
        ('7.20€', Price(currency='EUR', is_negotiable=False, value=7.2)),
        ('18.80€', Price(currency='EUR', is_negotiable=False, value=18.8)),
        ('18.50€', Price(currency='EUR', is_negotiable=False, value=18.5)),
        ('13.000 € VB', Price(currency='EUR', is_negotiable=True, value=13000.0)),
        ('14.750 € VB', Price(currency='EUR', is_negotiable=True, value=14750.0)),
        ('7.500 € VB', Price(currency='EUR', is_negotiable=True, value=7500.0)),
        ('2.999 € VB', Price(currency='EUR', is_negotiable=True, value=2999.0)),
        ('2.200 € VB', Price(currency='EUR', is_negotiable=True, value=2200.0)),
        ('4.00€', Price(currency='EUR', is_negotiable=False, value=4.0)),
        ('9.00€', Price(currency='EUR', is_negotiable=False, value=9.0)),
        ('23.98€', Price(currency='EUR', is_negotiable=False, value=23.98)),
        ('2.80€', Price(currency='EUR', is_negotiable=False, value=2.8)),
        ('8.49€', Price(currency='EUR', is_negotiable=False, value=8.49)),
        ('5.30€', Price(currency='EUR', is_negotiable=False, value=5.3)),
        ('7.999 € VB', Price(currency='EUR', is_negotiable=True, value=7999.0)),
        ('3.80€', Price(currency='EUR', is_negotiable=False, value=3.8)),
        ('49.40€', Price(currency='EUR', is_negotiable=False, value=49.4)),
        ('12.00€', Price(currency='EUR', is_negotiable=False, value=12.0)),
        ('14.900 € VB', Price(currency='EUR', is_negotiable=True, value=14900.0)),
        ('29.500 € VB', Price(currency='EUR', is_negotiable=True, value=29500.0)),
        ('4.700 € VB', Price(currency='EUR', is_negotiable=True, value=4700.0)),
        ('8.000 € VB', Price(currency='EUR', is_negotiable=True, value=8000.0)),
        ('2.800 € VB', Price(currency='EUR', is_negotiable=True, value=2800.0)),
        ('39.90€', Price(currency='EUR', is_negotiable=False, value=39.9)),
        ('37.600 € VB', Price(currency='EUR', is_negotiable=True, value=37600.0)),
        ('16.000 € VB', Price(currency='EUR', is_negotiable=True, value=16000.0)),
        ('29.999 € VB', Price(currency='EUR', is_negotiable=True, value=29999.0)),
        ('17.29€', Price(currency='EUR', is_negotiable=False, value=17.29)),
        ('11.50€', Price(currency='EUR', is_negotiable=False, value=11.5)),
        ('10.500 € VB', Price(currency='EUR', is_negotiable=True, value=10500.0)),
        ('3.00€', Price(currency='EUR', is_negotiable=False, value=3.0)),
        ('7.750 € VB', Price(currency='EUR', is_negotiable=True, value=7750.0)),
        ('8.50€', Price(currency='EUR', is_negotiable=False, value=8.5)),
        ('1.099 € VB', Price(currency='EUR', is_negotiable=True, value=1099.0)),
        ('8.950 € VB', Price(currency='EUR', is_negotiable=True, value=8950.0)),
        ('6.50€', Price(currency='EUR', is_negotiable=False, value=6.5)),
        ('12.00€', Price(currency='EUR', is_negotiable=False, value=12.0)),
        ('42.00€', Price(currency='EUR', is_negotiable=False, value=42.0)),
        ('48.990 € VB', Price(currency='EUR', is_negotiable=True, value=48990.0)),
    ],
)
def test_parse_prices(price_string, expected_price):
    parsed_price = parse_price(price_string)
    assert parsed_price == expected_price
