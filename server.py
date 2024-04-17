import json
from aiohttp import ClientSession
import asyncio
import re
import gradio as gr

can_use_servers = []

def load_server():
    with open("server.json", "r", encoding="utf8") as f:
        return json.load(f)["servers"]

def init_load_server():
    global can_use_servers
    can_use_servers = load_server()

async def request_for_free(timeout_count = 1):
    global can_use_servers
    servers = load_server()
    i = 0
    while i < timeout_count:
        for server in servers:
            # print(server)
            # 如果服务器没被使用
            if server in can_use_servers:
                ip = "http://" + server["ip"]
                port = server["port"]
                url = ip + ":" + str(port) + "/sdapi/v1/progress"

                # 移除，占用当前服务器
                can_use_servers.remove(server)
                try:
                    # timeout = ClientTimeout(total=3)
                    async with ClientSession() as session:
                        async with session.get(url) as response:
                            res_state = await response.json()
                            if res_state["state"]["job"] == "":
                                return True, ip, port

                            can_use_servers.append(server)
                except Exception as e:
                    print("request for free error: ", e)
                    gr.Warning(f"报错：{e}")
                    can_use_servers.append(server)
        i += 1
        # time.sleep(delay)
    gr.Warning("正在排队，当前等待用户：1名")
    await asyncio.sleep(2)
    return False, "", 0

def pop_back_server(url):
    global can_use_servers
    url = url.split(":")
    ip = url[0] + ":" + url[1]
    port = url[2]

    ip_address = ip
    if "http" in ip:
        pattern = r'\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}'
        ip_address = re.search(pattern, ip).group()
    server = {
        "ip": ip_address,
        "port": int(port)
    }
    can_use_servers.append(server)
