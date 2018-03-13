import json


def createJSON():
    person={}
    person['Ankesh']={
        'Name' : 'Ankesh'
    }
    person['Akku']={
        'Name' : 'Akku'
    }
    return  json.dumps(person)


