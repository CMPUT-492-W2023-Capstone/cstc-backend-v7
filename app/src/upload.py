import datetime
import requests


DB = 'https://cmput492-cstc-2023-default-rtdb.firebaseio.com'


async def main(traffic_datas):
    json_data = {
        'detected': len(traffic_datas)
    }

    for vehicle in traffic_datas.values():
        if vehicle.vehicle_type in json_data:
            json_data[vehicle.vehicle_type] += 1
        else:
            json_data[vehicle.vehicle_type] = 1

    t = datetime.datetime.now().strftime("%y-%m-%d-%H:%M:%S")

    response = requests.put(f'{DB}/dev/{t}.json', json=json_data)
