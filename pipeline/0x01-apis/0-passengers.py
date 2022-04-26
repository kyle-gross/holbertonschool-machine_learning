#!/usr/bin/env python3
"""Contains the function availableShips()"""

import requests
import json


def availableShips(passengerCount):
    """Returns the list of ships that can hold a given number of passengers
    """
    url = 'https://swapi-api.hbtn.io/api/starships/?format=json'
    ships = []

    while url:
        r = requests.get(url).json()
        for result in r['results']:
            try:
                if int(result['passengers'].replace(',', ''))\
                   >= passengerCount:
                    ships.append(result['name'])
            except Exception as e:
                continue
        url = r.get('next')

    return ships


if __name__ == '__main__':
    ships = availableShips(4)
    for ship in ships:
        print(ship)
