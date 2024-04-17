import base64
import json
import gradio as gr
from PIL import Image
from io import BytesIO
import asyncio
from aiohttp import ClientSession
from server import *
import time
import logging
from logging import handlers

max_limit = 1536

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)

log = Logger('all.log',level='debug')

def load_config():
    with open("config.json", "r", encoding="utf8") as f:
        return json.load(f)

def get_types():
    config = load_config()
    types = config["type"]
    return [x["name"] for x in types]

def get_modes(type_name):
    config = load_config()
    types = config["type"]
    for _type in types:
        if _type["name"] == type_name:
            modes = _type["mode"]
            return [x["name"] for x in modes]

def get_styles(type_name):
    config = load_config()
    types = config["type"]
    for _type in types:
        if _type["name"] == type_name:
            styles = _type["style"]
            return [x["name"] for x in styles]

def get_mode(type_name, mode_name):
    config = load_config()
    types = config["type"]
    for _type in types:
        if _type["name"] == type_name:
            modes = _type["mode"]
            for mode in modes:
                if mode["name"] == mode_name:
                    return mode

def get_style(type_name, style_name):
    config = load_config()
    types = config["type"]
    for _type in types:
        if _type["name"] == type_name:
            styles = _type["style"]
            for style in styles:
                if style["name"] == style_name:
                    return style

def get_reference(type_name):
    config = load_config()
    types = config["type"]
    for _type in types:
        if _type["name"] == type_name:
            reference = _type["reference"]
            return reference

def base64_encode(np_array):
    img = Image.fromarray(np_array, mode='RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

async def post_request(url, data):
    async with ClientSession() as session:
        async with session.post(url, json=data) as response:
            res = await response.json()
            return res

async def fetch(session, url):
    # await asyncio.sleep(count * 0.1)
    while True:
        await asyncio.sleep(1)
        async with session.get(url) as resp:
            if resp.status != 200:
                resp.raise_for_status()
            data = await resp.json()

            if data["state"]["job_count"] <= 0 and data["progress"] == 0:
                break
            yield data

async def progress_request(state, url):
    if not state:
        yield gr.Slider(visible=False)
    async with ClientSession() as session:
        combine_url = url + "/sdapi/v1/progress"
        async for i in fetch(session, combine_url):
            progress = i["progress"]
            progress = round(progress * 100, 2)
            eta = i["eta_relative"]
            eta = round(eta, 0)
            yield gr.Slider(label=f"{progress}% 预计剩余时间: {eta}s", visible=True, value=progress)

def _upload_image(image):
    if image is not None:
        # 检查分辨率
        width = image.shape[1]
        height = image.shape[0]
        if 512 <= width <= max_limit and 512 <= height <= max_limit:
            return width, height
        else:
            gr.Warning(f"图片尺寸应在 512x512 至 {max_limit}x{max_limit} 之间！")
            return width, height
    return gr.Number(), gr.Number()

def _scene_mode(scene_mode):
    modes = get_modes(scene_mode)
    styles = get_styles(scene_mode)
    return gr.Dropdown(value=modes[0], choices=modes), gr.Dropdown(value=styles[0], choices=styles)

def check_image_same(image, mask, tolerance=5):
    width_diff = abs(image.shape[1] - mask.shape[1])
    height_diff = abs(image.shape[0] - mask.shape[0])
    if width_diff <= tolerance and height_diff <= tolerance:
        return True
    return False

async def gen_images(
        is_busy_state,
        url_text,
        upload_image,
        upload_mask_image,
        image_width,
        image_height,
        scene_mode,
        gen_mode,
        gen_style,
        ref_image,
        draw_influence,
        controlnet_influence,
        num_of_images):
    if not is_busy_state:
        return gr.Gallery(), gr.Image(), gr.Image(), gr.Slider(), gr.Button(), gr.Checkbox(), gr.Button(), gr.Button()

    mode = get_mode(scene_mode, gen_mode)
    style = get_style(scene_mode, gen_style)

    if mode["need_image"]:
        input_image = base64_encode(upload_image)

    # controlnet的参数
    control_nets = mode["control_net"]
    control_nets_args = []
    for control_net in control_nets:
        params = {
            "input_image": input_image,
            "module": control_net["module"],
            "model": control_net["model"],
            "weight": control_net["weight"] * controlnet_influence / 100,
            "pixel_perfect": True
        }
        control_nets_args.append(params)

    # 增加参考图
    if ref_image is not None:
        reference = get_reference(scene_mode)
        params = {
            "input_image": base64_encode(ref_image),
            "module": reference["module"],
            "model": reference["model"],
            "weight": draw_influence / 125,
            "pixel_perfect": True
        }
        control_nets_args.append(params)

    # lora的参数
    loras = style["lora"]
    lora_params = [False, False]
    if len(loras) != 0:
        lora_params = [True, False]
    for i in range(5):
        lora_params.append("LoRA")
        if i < len(loras):
            lora = loras[i]
            lora_params.append(lora["model"])
            lora_params.append(lora["weight"])
            lora_params.append(lora["weight"])
        else:
            lora_params.append("None")
            lora_params.append(1)
            lora_params.append(1)

    # 提示词
    prompt = ""
    neg_prompt = ""
    if ref_image is not None:
        prompt = mode["reference_prompt"]
        neg_prompt = mode["reference_neg_prompt"]
    else:
        mode_styles = mode["style"]
        for mode_style in mode_styles:
            if mode_style["name"] == "None":
                prompt = mode_style["prompt"]
                neg_prompt = mode_style["neg_prompt"]
                break
            if mode_style["name"] == style["name"]:
                prompt = mode_style["prompt"]
                neg_prompt = mode_style["neg_prompt"]
                break

    # 是否使用vae
    override_settings = {
        "sd_model_checkpoint": style["checkpoint"]
    }
    if style["vae"] != "":
        override_settings["sd_vae"] = style["vae"]

    if mode["type"] == "txt2img":
        json_data = {
            "enable_hr": False,
            "firstphase_width": 0,
            "firstphase_height": 0,
            "hr_scale": 2,
            "hr_upscaler": "string",
            "hr_second_pass_steps": 0,
            "hr_resize_x": 0,
            "hr_resize_y": 0,
            "prompt": prompt,
            "seed": -1,
            "subseed": -1,
            "subseed_strength": 0,
            "seed_resize_from_h": -1,
            "seed_resize_from_w": -1,
            "batch_size": 1,
            "n_iter": num_of_images,
            "steps": 20,
            "cfg_scale": 7,
            "width": image_width,
            "height": image_height,
            "restore_faces": False,
            "tiling": False,
            "negative_prompt": neg_prompt,
            "eta": 0,
            "s_churn": 0,
            "s_tmax": 0,
            "s_tmin": 0,
            "s_noise": 1,
            "override_settings": override_settings,
            "override_settings_restore_afterwards": True,
            "script_args": [],
            "sampler_index": "Euler a",
            "alwayson_scripts": {}
        }
    elif mode["type"] == "img2img":
        json_data = {
            "denoising_strength": 0.75,
            "prompt": style["prompt"],
            "seed": -1,
            "batch_size": 1,
            "n_iter": num_of_images,
            "steps": 20,
            "cfg_scale": 7,
            "width": image_width,
            "height": image_height,
            "tiling": False,
            "negative_prompt": style["neg_prompt"],
            "eta": 0,
            "s_churn": 0,
            "s_tmax": 0,
            "s_tmin": 0,
            "s_noise": 0,
            "override_settings": {
                "sd_model_checkpoint": style["checkpoint"]
            },
            "override_settings_restore_afterwards": True,
            "script_args": [],
            "sampler_index": "Euler a",
            "inpainting_fill": 1,
            "inpaint_full_res_padding": 0,
            "init_images": [input_image],
            "alwayson_scripts": {}
        }
        if mode["mask"]:
            json_data["mask"] = base64_encode(upload_mask_image)
            json_data["mask_blur"] = 4

    if mode["refiner"]:
        json_data["refiner_checkpoint"] = "sd_xl_refiner_1.0.safetensors"
        json_data["refiner_switch_at"] = 0.8

    json_data["alwayson_scripts"]["controlnet"] = {
        "args": control_nets_args
    }

    if ref_image is None:
        json_data["alwayson_scripts"]["Additional networks for generating"] = {
            "args": lora_params
        }

    gr.Info("正在生成中，较大分辨率的图片生成更加耗费时间")
    start_time = time.time()

    res = None
    try:
        if mode["type"] == "txt2img":
            url = url_text + "/sdapi/v1/txt2img"
            res = await post_request(url, json_data)

            res_images = []
            for b64img in res["images"]:
                image_data = base64.b64decode(b64img)
                res_images.append(Image.open(BytesIO(image_data)))

            info = json.loads(res["info"])
            infotexts = info["infotexts"]

            spend_time = time.time() - start_time
            log.logger.debug(f"\nurl:{url_text}\n花费时间:{spend_time}\n生成信息：{infotexts}\n")

            pop_back_server(url_text)
            return gr.Gallery(res_images, visible=True), gr.Image(visible=False), gr.Image(visible=False), gr.Slider(visible=False),  gr.Button(interactive=True), False, gr.Button(interactive=True), gr.Button(visible=True)
    except Exception as e:
        if res is not None:
            Logger('error.log', level='error').logger.error(f"\n错误：{e}\nurl:{url_text}\n服务器返回：{res}\n")
            gr.Warning(f"错误: {e} 服务器返回：{res}\n")
        else:
            Logger('error.log', level='error').logger.error(f"\n错误：{e}\nurl:{url_text}\n")
            gr.Warning(f"错误: {e}\n")

        pop_back_server(url_text)
        await asyncio.sleep(5)
        return gr.Gallery(), gr.Image(), gr.Image(), gr.Slider(visible=False), gr.Button(interactive=True), False, gr.Button(interactive=True), gr.Button(visible=True)

async def detect(scene_mode, gen_mode, upload_image, upload_mask_image, ref_image):
    mode = get_mode(scene_mode, gen_mode)
    need_image = mode["need_image"]
    mask = mode["mask"]
    if need_image and mask:
        if upload_image is None and upload_mask_image is None:
            gr.Warning("请上传图片！")
            return gr.Button(interactive=True), "", False, gr.Slider(visible=False), gr.Button(interactive=True), gr.Button(visible=True)
        if not check_image_same(upload_image, upload_mask_image, tolerance=5):
            gr.Warning("图片和蒙版尺寸不一致！")
            return gr.Button(interactive=True), "", False, gr.Slider(visible=False), gr.Button(interactive=True), gr.Button(visible=True)

    if need_image:
        if upload_image is None:
            gr.Warning("请上传图片！")
            return gr.Button(interactive=True), "", False, gr.Slider(visible=False), gr.Button(interactive=True), gr.Button(visible=True)

        # 检查分辨率
        width = upload_image.shape[1]
        height = upload_image.shape[0]
        if not 512 <= width <= max_limit and not 512 <= height <= max_limit:
            gr.Warning(f"请重新上传图片，尺寸应在 512x512 至 {max_limit}x{max_limit} 之间！")
            return gr.Button(interactive=True), "", False, gr.Slider(visible=False), gr.Button(interactive=True), gr.Button(visible=True)

    # 检查参考图尺寸
    if ref_image is not None:
        if need_image:
            if not check_image_same(upload_image, ref_image, tolerance=5):
                gr.Warning("参考图与图片尺寸不一致！")
                return gr.Button(interactive=True), "", False, gr.Slider(visible=False), gr.Button(interactive=True), gr.Button(visible=True)


    is_response, ip, port = await request_for_free()
    if is_response:
        url = ip + ":" + str(port)
        return gr.Button(interactive=False), url, True, gr.Slider(visible=True), gr.Button(interactive=False), gr.Button(visible=False)
    else:
        return gr.Button(interactive=True), "", False, gr.Slider(visible=False), gr.Button(interactive=True), gr.Button(visible=True)

def _new_btn(scene_mode, gen_mode):
    mode = get_mode(scene_mode, gen_mode)
    need_image = mode["need_image"]
    mask = mode["mask"]
    if need_image and mask:
        return gr.Image(value=None, visible=True), gr.Image(value=None, visible=True), gr.Gallery(value=None, visible=False)
    else:
        return gr.Image(value=None, visible=True), gr.Image(value=None, visible=False), gr.Gallery(value=None, visible=False)

def _change_mode(scene_mode, gen_mode):
    mode = get_mode(scene_mode, gen_mode)
    need_image = mode["need_image"]
    mask = mode["mask"]
    if not need_image:
        return gr.Image(interactive=False), gr.Image(visible=False), gr.Slider(visible=False)
    if need_image and mask:
        return gr.Image(interactive=True), gr.Image(visible=True), gr.Slider(visible=True)
    if need_image:
        return gr.Image(interactive=True), gr.Image(visible=False), gr.Slider(visible=True)

def upload_ref_image(ref_image):
    if ref_image is not None:
        # 检查分辨率
        width = ref_image.shape[1]
        height = ref_image.shape[0]
        if 512 <= width <= max_limit and 512 <= height <= max_limit:
            return gr.Dropdown(visible=False)
        else:
            gr.Warning(f"图片尺寸应在 512x512 至 {max_limit}x{max_limit} 之间！")
            return gr.Dropdown(visible=True)
    else:
        return gr.Dropdown(visible=True)

def webui(global_port):
    with (gr.Blocks(title="小鹏造型中心AI", analytics_enabled=False, css="style.css", theme='WeixuanYuan/Base_dark') as demo):
        with gr.Row():
            with gr.Column(scale=4):
                head = """# ![](https://xps01.xiaopeng.com/www/public/img/white-logo.570fd7b8.svg)   Design Center AI
                        ### 小鹏造型中心    v1.240415"""
                gr.Markdown(head)
                # gr.Markdown("![](file/logo.png)")
                # gr.Markdown("![](https://xps01.xiaopeng.com/www/public/img/white-logo.570fd7b8.svg)")
                new_btn = gr.Button(value="新建", variant="primary", scale=0, size="lg")
                with gr.Row():
                    upload_image = gr.Image(label="上传图片", height=1000, elem_id="upload-image", container=False)
                    upload_mask_image = gr.Image(label="上传蒙版图片", height=1000, visible=False)

                results_gallery = gr.Gallery(label="生成结果", height=1000, visible=False, preview=True, show_download_button=False, columns=4, container=False)
                progress_bar = gr.Slider(label="正在生成中", value=0, visible=False)
                pro_btn = gr.Button(value="尝试专业版", variant="primary", scale=0, size="lg")

            with gr.Column(scale=1):
                gr.Markdown("![](https://raw.githubusercontent.com/Circle1688/SimpleStableDiffusionUI/main/image/logo.png)")
                with gr.Row():
                    image_width = gr.Slider(value=1024, minimum=512, maximum=max_limit, label="宽度", step=1)
                    image_height = gr.Slider(value=1024, minimum=512, maximum=max_limit, label="高度", step=1)
                gr.Markdown(f"""### 图片尺寸限制在 512x512 至 {max_limit}x{max_limit} 之间""")
                # modes = ["草图上色", "草渲上色", "无中生有", "局部重绘"]
                # styles = ["Int"]
                types = get_types()
                modes = get_modes(types[0])
                styles = get_styles(types[0])

                scene_mode = gr.Radio(label="内外饰", value=types[0], choices=types)
                gen_mode = gr.Dropdown(label="模式", value=modes[0], choices=modes)

                controlnet_influence = gr.Slider(label="上传图影响", value=100, minimum=0, maximum=100, step=1)

                gen_style = gr.Dropdown(label="风格", value=styles[0], choices=styles)


                with gr.Accordion(label="参考图", open=False):
                    ref_image = gr.Image(label="上传参考图", height=250)
                    draw_influence = gr.Slider(label="参考图影响", value=80, minimum=60, maximum=100, step=1)

                num_of_images = gr.Radio(label="生成数量", value="1", choices=["1", "4"])

                gen_btn = gr.Button(value="生成", variant="primary", size="lg")


        js = """
                function addParamFunction() {
                    function addParam() {
                        gradioURL = 'http://10.192.119.100:7861/?__theme=dark';
                        window.open(gradioURL + '?__theme=dark', '_self');
                    }
                    if (document.readyState === "loading") {
                      // Loading hasn't finished yet
                      document.addEventListener("DOMContentLoaded", addParam);
                    } else {
                      // `DOMContentLoaded` has already fired
                      addParam();
                    }
                }
                """
        def go_to():
            pass
        pro_btn.click(go_to, js=js)

        scene_mode.change(_scene_mode, scene_mode, [gen_mode, gen_style])
        upload_image.change(_upload_image, upload_image, [image_width, image_height])
        new_btn.click(_new_btn, inputs=[scene_mode, gen_mode], outputs=[upload_image, upload_mask_image, results_gallery])
        gen_mode.change(_change_mode, inputs=[scene_mode, gen_mode], outputs=[upload_image, upload_mask_image, controlnet_influence])
        ref_image.change(upload_ref_image, ref_image, gen_style)

        is_busy_state = gr.Checkbox(value=False, visible=False)
        url_text = gr.Textbox("", visible=False)

        input_comps = [is_busy_state, url_text, upload_image, upload_mask_image, image_width, image_height, scene_mode, gen_mode, gen_style, ref_image, draw_influence, controlnet_influence, num_of_images]
        output_comps = [results_gallery, upload_image, upload_mask_image, progress_bar, gen_btn, is_busy_state, new_btn, pro_btn]

        gen_btn.click(fn=detect, inputs=[scene_mode, gen_mode, upload_image, upload_mask_image, ref_image], outputs=[gen_btn, url_text, is_busy_state, progress_bar, new_btn, pro_btn], show_progress="hidden")
        is_busy_state.change(gen_images, input_comps, output_comps)
        is_busy_state.change(progress_request, [is_busy_state, url_text], progress_bar, show_progress="hidden")

    demo.queue(default_concurrency_limit=400).launch(server_name="0.0.0.0", server_port=global_port, allowed_paths=["/"])
