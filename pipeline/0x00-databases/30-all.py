#!/usr/bin/env python3
"""Contains the method list_all()"""

from pymongo import MongoClient


def list_all(mongo_collection):
    """Lists docs from mongo collection"""
    return mongo_collection.find()


if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    school_collection = client.my_db.school
    schools = list_all(school_collection)
    for school in schools:
        print("[{}] {}".format(school.get('_id'), school.get('name')))
