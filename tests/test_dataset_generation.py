import io
import PIL
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import warnings

plt.switch_backend('agg')

import matplotlib.patheffects as path_effects

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['SimHei']
font_manager.fontManager.addfont('./SimHei.ttf')

import os

os.environ["http_proxy"] = "http://127.0.0.1:8889"
os.environ["https_proxy"] = "http://127.0.0.1:8889"

from datasets import load_dataset


def test_load_dataset():
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
    print(dataset['train']['text'][0:2])

def draw_text_pil(text: str):
    font = ImageFont.truetype('SimHei.ttf', 16)
    img = Image.new('RGB', (500, 500), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, font=font, fill=(0, 0, 0), wrap=True)
    return img


def draw_text(text: str):
    color_a_all = ['red', 'green', 'blue', 'white', 'black', 'yellow', 'purple', 'orange', 'pink']
    color_b_all = ['red', 'green', 'blue', 'white', 'black', 'yellow', 'purple', 'orange', 'pink']
    color_a = np.random.randint(0, len(color_a_all))
    color_b = np.random.randint(0, len(color_b_all))
    text_size = (550000 / len(text)) ** 0.5
    while color_b == color_a:
        color_b = np.random.randint(0, 2)
    fig = plt.figure(figsize=(10, 10), dpi=100, facecolor=color_a_all[color_a])
    text = fig.text(0.5, 0.5, text,
                    ha='center', va='center', size=text_size, wrap=True, 
                    color=color_b_all[color_b])
    text.set_path_effects([path_effects.Normal()])
    buffer = io.BytesIO()
    canvas = fig.canvas
    canvas.print_figure(buffer, format='png', dpi=100)
    img = Image.open(buffer)
    plt.close('all')
    return img

def preprocess_text(text: str):
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = ' '.join(text)
    text = ' '.join(text.split())
    return text

def test_draw_text():
    img = draw_text_pil(
        text=preprocess_text(
            '''
施氏食狮史
赵元任
                    
石室诗士施氏，嗜狮，誓食十狮。施氏时时适市视狮。
十时，适十狮适市。是时，适施氏适市。
施氏视是十狮，恃矢势，欲食之。
十狮惧，施氏退，适市视之。
是时，适施氏适市。
施氏视是十狮，恃矢势，欲食之。
十狮惧，施氏退，适市视之。
和关于风流反馈与法律健康辅导苦于施氏。施氏食狮史，赵元任，石室诗士，嗜狮，誓食十狮。施氏时时适市视狮。十时，适十狮适市。是时，适施氏适市。施氏视是十狮，恃矢势，欲食之。十狮惧，施氏退，适市视之。是时，适施氏适市。施氏视是十狮，恃矢势，欲食之。十狮惧，施氏退，适市视之。和关于风流反馈与法律健康辅导苦于施氏。施氏食狮史，赵元任，石室诗士，嗜狮，誓食
和vjhfgjh和关于风流反馈与法律健康辅导苦于施氏。施氏食狮史，赵元任，石室诗士，嗜狮，誓食十狮。施氏时时适市视狮。十时，适十狮适市。是时，适施氏适市。施氏视是十狮，恃矢势，欲食之。十狮惧，施氏退，适市视之。是时，适施氏适市。施氏视是十狮，恃矢势，欲食之。十狮惧，施氏退，适市视之。和关于风流反馈与法律健康辅导苦于施氏。施氏食狮史，赵
            ''' * 2
        )
    )
    img = np.asanyarray(img)
    plt.figimage(img)
    plt.show()

flag = True


def setup_dataset():
    from tqdm import tqdm
    from datasets import Dataset
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
    def aa():
        global flag
        if flag:
            warnings.filterwarnings("error")
            flag = False
        for field in dataset.keys():
            for index, data in enumerate(tqdm(dataset[field]['text'])):
                if (index + 1) % 40 != 0:
                    continue
                try:
                    data = data.strip()
                    if len(data) < 2:
                        continue
                    text = preprocess_text(data)
                    img = draw_text(text)
                    yield {'image': img, 'text': text}
                except:
                    continue
    dataset = Dataset.from_generator(aa, num_proc=4)
    print(dataset)
    dataset.save_to_disk('data')


if __name__ == '__main__':
    # test_draw_text()
    # test_load_dataset()
    setup_dataset()

