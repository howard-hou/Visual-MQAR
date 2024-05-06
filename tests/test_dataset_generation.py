import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

import matplotlib.patheffects as path_effects

if __name__ == '__main__':
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    font_manager.fontManager.addfont('./SimHei.ttf')

    fig = plt.figure(figsize=(10, 10), dpi=100, facecolor='g')
    text = fig.text(0.5, 0.5, '''
施氏食狮史
赵元任
                    
石室诗士施氏，嗜狮，誓食十狮。施氏时时适市视狮。
十时，适十狮适市。是时，适施氏适市。
施氏视是十狮，恃矢势，使是十狮逝世。
氏拾是十狮尸，适石室。石室湿，氏使侍拭石室。
石室拭，氏始试食是十狮尸。
食时，始识是十狮尸，实十石狮尸。试释是事。 
                    '''.strip(),
                    ha='center', va='center', size=60, wrap=True, color='yellow')
    text.set_path_effects([path_effects.Normal()])
    plt.show()
