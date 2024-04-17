from webui import webui
from server import init_load_server
global_port = 7860

if __name__ == "__main__":
    init_load_server()
    webui(global_port=global_port)
