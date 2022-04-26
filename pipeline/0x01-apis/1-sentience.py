#!/usr/bin/env python3
"""Contains the function sentientPlanets()"""

import requests


def sentientPlanets():
    """Returns the list of names of the home planets of all sentient species
    """
    url = 'https://swapi-api.hbtn.io/api/species/?format=json'
    planets = []

    while url:
        r = requests.get(url).json()
        for species in r['results']:
            if (species['designation'] == 'sentient' and
                species['homeworld'] is not None):
                planets.append(requests.get(
                    species['homeworld']
                ).json()['name'])
        url = r.get('next')

    return planets

if __name__ == '__main__':
    planets = sentientPlanets()
    for planet in planets:
        print(planet)
