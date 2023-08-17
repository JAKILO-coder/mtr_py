import io
import paho.mqtt.client as mqtt
import time
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from flask import (
    Blueprint,
    render_template,
    abort,
    current_app,
    make_response
    )
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import datetime
import matplotlib.colors as mcolors
import ast
import base64
import csv
import datetime
import json
import time
import zlib

from paho.mqtt import client as mqtt_client
import threading

client = Blueprint('client', __name__, template_folder='templates', static_url_path='/static')
# broker = 'mtr-uat-internal.smartsensing.biz'
# port = 4000
broker = '143.89.49.63'
port = 1883
topic = "topic/sub"
# client_id = f'python-mqtt-1'
pos_data = None
w_data = None

@client.route('/')
def home():
    title = current_app.config['TITLE']
    # poly_location = np.random.random((1, 3, 3))
    if pos_data == None or w_data == None:
        p_location = np.array([[114.19890707301564, 22.32988199829699], [114.19904346836372, 22.3299019348249]])
        p_location += np.random.random((2, 2))*0.0001
        weights = np.random.random(2)
        plot = plot_map_practicle(polygon_location=polygon_location, practicle_location=p_location, weights=weights,
                                  is_save=None, source_location=source_location)
    else:
        plot = plot_map_practicle(polygon_location=polygon_location, practicle_location=pos_data, weights=w_data,
                                  is_save=None, source_location=source_location)

    return render_template('index.html', title=title, plot=plot)

#####################################################################################################################
def plot_map_practicle(polygon_location, practicle_location, weights, is_save, source_location=None):
    # 创建一个绘图对象和一个子图
    fig, ax = plt.subplots(figsize=(10, 20))
    #     fig, ax = plt.subplots(figsize=(10, 5))

#     绘制每一个多边形
    for polygon in polygon_location:
        polygon = np.array(polygon)
        x = polygon[:, 0]
        y = polygon[:, 1]
        z = polygon[:, 2]
        # 绘制多边形顶点
        ax.plot(x, y, '-', color='blue', linewidth=0.01)

        # 连接多边形的边
        ax.plot(np.append(x, x[0]), np.append(y, y[0]), color='blue')

        # 在多边形中间用浅蓝色填充
        ax.fill(x, y, color='lightblue')

    # plot practicle
    # 计算权重的最小值和最大值
    min_weight = np.min(weights)
    max_weight = np.max(weights)

    # 创建一个颜色映射
    norm = mcolors.Normalize(vmin=min_weight, vmax=max_weight)
    cmap = plt.cm.RdYlGn  # 使用RdYlGn颜色映射，权重越小越绿，权重越大越红
    scalar_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #     colors = plt.cm.RdYlGn(weights)  # 使用RdYlGn颜色映射，权重越大颜色越红，权重越小颜色越绿
    colors = scalar_map.to_rgba(weights)
    p_location = np.array(practicle_location)
    ax.scatter(p_location[:, 0], p_location[:, 1], c=colors, s=5, zorder=10)  # 设置点的大小为50

    #     plot source
    s_location = np.array(source_location)
    # print(s_location.shape)
    ax.scatter(s_location[:, 0], s_location[:, 1], zorder=12, s=50)

    # 设置绘图范围和标签
    ax.set_aspect('equal', adjustable='datalim')  # 保持纵横比相等
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_title('Polygon Vertices and Edges')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    polygon_np = np.array(polygon_location)
    ax.set_xlim(polygon_np[:, :, 0].min() - 0.00001, polygon_np[:, :, 0].max() + 0.00001)  # 设置 x 坐标范围
    ax.set_ylim(polygon_np[:, :, 1].min() - 0.00001, polygon_np[:, :, 1].max() + 0.00001)  # 设置 y 坐标范围
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

#     print(np.array(polygon_location)[0, 0, :])
    # 显示绘图
    # plt.show()

    if is_save:
        # 获取当前时间
        current_time = datetime.datetime.now()

        # 提取年、月、日、时、分、秒
        year = current_time.year
        month = current_time.month
        day = current_time.day
        hour = current_time.hour
        minute = current_time.minute
        second = current_time.second
        plt.save(f'{year}{month}{day}{hour}{minute}{second}')

    # plt.show()
    img = io.StringIO()
    fig.savefig(img, format='svg')
    # clip off the xml headers from the image
    svg_img = '<svg' + img.getvalue().split('<svg')[1]

    return svg_img
##################################################################################################################
# get map
import re

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1#
# use the map geo location here
# file_path = 'D:\\留学\\港科大\\research assistant job\\mtr_cms\\planb_python\\KAT-polygon_source-geojson-1690900701.txt'  # 替换成你的文件路径
file_path = '/home/mtrec/Desktop/mtr-py/mtr_py/flaskplotlib/KAT-polygon_source-geojson-1690900701.txt'

with open(file_path, 'r') as file:
    data = file.read()

# 提取数据
maps_start = data.find('MAPS = {')
maps_end = data.find('}', maps_start)
maps_data = data[maps_start + len('MAPS = {'):maps_end].strip()

# 解析多边形数据
polygon_name = []
polygon_location = []

for line in maps_data.split('\n'):
    parts = line.strip().split(':')
    if len(parts) == 2:
        key = parts[0].strip()
        value = parts[1].strip()
        if value.endswith(','):
            value = value[:-1]
        points = value.split('],')
        points_list = []
        for point in points:
            coords = point.strip('[]').split(',')
            x = float(coords[0])
            y = float(coords[1])
            z = int(coords[2])
            points_list.append([x, y, z])
        polygon_name.append(int(key))
        polygon_location.append(points_list)

# print("MAPS:", polygon_name, polygon_location)

# Extract data using regular expressions
source_name = []
source_location = []
source_info_match = re.findall(r"'(\d+)': SOURCE_INFO\(source_identifier='(\d+)', x=([\d.]+), y=([\d.]+), z=([\d.]+)",
                               data)

if source_info_match:
    source_info_dict = {}
    for identifier, _, x, y, z in source_info_match:
        source_name.append(identifier)
        source_location.append([float(x), float(y), float(z)])

########################################################################################################################

def connect_mqtt():
    global pos_data, w_data

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
            client.on_message = on_message
            client.subscribe('tester01')
        else:
            print("Failed to connect, return code %d\n", rc)

    def on_message(client, userdata, msg):
        get_data = msg.payload
        # 将字节数据转换为字符串
        data_str = get_data.decode('utf-8')
        # while(1):
        #     pos_data = np.array([[114.19900707301564, 22.32988599829699], [114.19907346836372, 22.3299519348249]])
        #     pos_data += np.random.random((2, 2)) * 0.0001
        #     w_data = np.random.random(2)

        # 解析JSON数据
        data = json.loads(data_str)
        pos_data = np.array(data['pos'].replace('[', '').replace(']', '').split(', '), dtype=float).reshape(-1, 2)

        # 提取w字段的数据并转换为numpy.array
        w_data = np.array(data['w'].replace('[', '').replace(']', '').split(', '), dtype=float)


    #         res = json.loads(zlib.decompress(msg.payload))
    #         print(res)
    # res = json.loads(zlib.decompress(msg.payload))
    # for str_idx, encrypted_str in res.items():
    #     x = base64.b64decode((bytes(encrypted_str, encoding='utf-8')))
    #     x = json.loads(x)
    #     x = ast.literal_eval(x)
    #     if isinstance(x, dict):
    #         print(ts2date(x['_timestamp']), ' | ', ts2date(time.time()), x)
    #     elif isinstance(x, list):
    #         print(ts2date(x[7]), ' | ', ts2date(time.time()), x)
    #         f_csv.writerow(x)
    #         f.flush()

    # Set Connecting Client ID
    client = mqtt_client.Client()
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def publish(client):
    global msg_count
    while True:
        time.sleep(1)
        msg = f"messages: {msg_count}"
        result = client.publish(topic=topic, payload=msg, qos=2)
        # result: [0, 1]
        status = result[0]
        msg_count += 1
        print(client.is_connected())
        if status == 0:
            print(f"Send `{msg}` to topic `{topic}` {int(time.time())} {status}")
        else:
            print(f"Failed to send message to topic {topic} {status}")
            client.disconnect()
            client.reinitialise()
            break


print('create clinet mqtt')
client1 = connect_mqtt()
print('start mqtt loop')
infinite_thread = threading.Thread(target=client1.loop_forever)
infinite_thread.daemon = True
infinite_thread.start()
client1.is_connected()
