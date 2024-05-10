import pytest
from ai.car.main import transform_url


@pytest.mark.parametrize(
    "url, expected_url, expected_query",
    [
        (
            'https://www.kleinanzeigen.de/s-sharan-7-sitzer/k0',
            'https://www.kleinanzeigen.de/s-seite:3/sharan-7-sitzer/k0',
            'sharan-7-sitzer',
        ),
        (
            'https://www.kleinanzeigen.de/s-audi-a4/k0',
            'https://www.kleinanzeigen.de/s-seite:3/audi-a4/k0',
            'audi-a4',
        ),
        (
            'https://www.kleinanzeigen.de/s-bmw-3er/k0',
            'https://www.kleinanzeigen.de/s-seite:3/bmw-3er/k0',
            'bmw-3er',
        ),
        (
            'https://www.kleinanzeigen.de/s-invalid-url',
            'https://www.kleinanzeigen.de/s-seite:3/invalid-url',
            '',
        ),
    ],
)
def test_transform_url(url, expected_url, expected_query):
    new_url, query = transform_url(url)
    assert new_url == expected_url
    assert query == expected_query
