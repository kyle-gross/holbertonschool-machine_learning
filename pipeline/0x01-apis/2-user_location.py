#!/usr/bin/env python3
"""Script that finds a GitHub user location from their user name."""

import requests
import sys
import time


if __name__ == '__main__':
    if len(sys.argv) > 1:
        user = sys.argv[1]
        r = requests.get(user)

        if r.status_code == 404:
            print('Not found')
        if r.status_code == 403:
            reset_time = int(r.headers['X-Ratelimit-Reset']) - time.time()
            minutes = round(reset_time / 60)
            print(f'Reset in {minutes} min')
        if r.status_code == 200:
            location = r.json()['location']
            if location is not None:
                print(f'{location}')
            else:
                print('Not found')
    else:
        print('Invalid url. Run the script again with a user\'s url.')
