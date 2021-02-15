from datetime import date
from nsepy import get_history
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastai.vision import *
from fastai.metrics import error_rate
from fastai.callbacks.hooks import *


def show_heatmap(hm, xb_im):
    fig, ax = plt.subplots()
    xb_im.show(ax)
    ax.imshow(hm, alpha=0.6, extent=(0, 352, 352, 0),
              interpolation='bilinear', cmap='magma');
    return fig


def hooked_backward(m, xb, cat):
    with hook_output(m[0]) as hook_a:
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            preds[0, int(cat)].backward()
    return hook_a, hook_g


def obtain_data(ticker, start, end):
    # Enter the start and end dates using the method date(yyyy,m,dd)
    stock = get_history(symbol=ticker, start=start, end=end)
    df = stock.copy()
    df = df.reset_index()
    df = df.drop(['Series', 'Prev Close', 'Last', 'Turnover', '%Deliverble', 'Trades'], axis=1)
    df = df.rename({'Open': 'open_price', 'Close': 'close_price', 'High': 'high', 'Low': 'low', 'Volume': 'volume'},
                   axis='columns')
    df.index = df.Date
    return df


"""This cell defined the plot_candles function"""


def plot_candles(pricing, title=None, volume_bars=False, color_function=None, technicals=None):
    """ Plots a candlestick chart using quantopian pricing data.

    Author: Daniel Treiman

    Args:
      pricing: A pandas dataframe with columns ['open_price', 'close_price', 'high', 'low', 'volume']
      title: An optional title for the chart
      volume_bars: If True, plots volume bars
      color_function: A function which, given a row index and price series, returns a candle color.
      technicals: A list of additional data series to add to the chart.  Must be the same length as pricing.
    """

    def default_color(index, open_price, close_price, low, high):
        return 'r' if open_price[index] > close_price[index] else 'g'

    color_function = color_function or default_color
    technicals = technicals or []
    open_price = pricing['open_price']
    close_price = pricing['close_price']
    low = pricing['low']
    high = pricing['high']
    oc_min = pd.concat([open_price, close_price], axis=1).min(axis=1)
    oc_max = pd.concat([open_price, close_price], axis=1).max(axis=1)

    if volume_bars:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(7, 7))
    else:
        fig, ax1 = plt.subplots(1, 1)
    if title:
        ax1.set_title(title)
    fig.tight_layout()
    x = np.arange(len(pricing))
    candle_colors = [color_function(i, open_price, close_price, low, high) for i in x]
    candles = ax1.bar(x, oc_max - oc_min, bottom=oc_min, color=candle_colors, linewidth=0)
    lines = ax1.vlines(x, low, high, color=candle_colors, linewidth=1)
    ax1.xaxis.grid(True)
    ax1.yaxis.grid(True)
    ax1.xaxis.set_tick_params(which='major', length=3.0, direction='in', top='off')
    ax1.set_yticklabels([])
    # Assume minute frequency if first two bars are in the same day.
    frequency = 'minute' if (pricing.index[1] - pricing.index[0]).days == 0 else 'day'
    time_format = '%d-%m-%Y'
    if frequency == 'minute':
        time_format = '%H:%M'
    # Set X axis tick labels.
    # plt.xticks(x, [date.strftime(time_format) for date in pricing.index], rotation='vertical')
    for indicator in technicals:
        ax1.plot(x, indicator)

    if volume_bars:
        volume = pricing['volume']
        volume_scale = None
        scaled_volume = volume
        if volume.max() > 1000000:
            volume_scale = 'M'
            scaled_volume = volume / 1000000
        elif volume.max() > 1000:
            volume_scale = 'K'
            scaled_volume = volume / 1000
        ax2.bar(x, scaled_volume, color=candle_colors)
        volume_title = 'Volume'
        if volume_scale:
            volume_title = 'Volume (%s)' % volume_scale
        # ax2.set_title(volume_title)
        ax2.xaxis.grid(True)
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
    return fig


def data_gathering():
    # Data_Gathering
    equities = ['BAJFINANCE', 'RELIANCE', 'INFY', 'HDFC', 'HDFCBANK', 'HDFCLIFE', 'ZEEL', 'DELTACORP', 'ITC',
                'ASIANPAINT']
    for equity in equities:
        df_pricing = obtain_data(equity, date(2020, 1, 1), date(2020, 9, 1))
        df = df_pricing.copy()
        df = df.reset_index(drop=True)
        n_days = 5
        fraction_movement = 0.037
        df['Trend'] = None
        for i in range(len(df)):
            try:
                for n in range(n_days):
                    if df.loc[i, 'close_price'] - df.loc[i + n, 'close_price'] >= fraction_movement * df.loc[
                        i, 'close_price']:
                        df.loc[i, 'Trend'] = 'Down'
                        if i >= 20:
                            fig = plot_candles(df_pricing[i - 20:i], volume_bars=True)
                            fig.savefig(
                                'C:/Users/hishah/PycharmProjects/Candlestick-Patterns/Candle_Data/Down/{0}{1}.png'.format(
                                    df_pricing['Symbol'][i], i), dpi=70)

                        print('Down', i, n)
                        break
                    elif df.loc[i + n, 'close_price'] - df.loc[i, 'close_price'] >= fraction_movement * df.loc[
                        i, 'close_price']:
                        df.loc[i, 'Trend'] = 'Up'
                        if i > 20:
                            fig = plot_candles(df_pricing[i - 20:i], volume_bars=True)
                            fig.savefig(
                                'C:/Users/hishah/PycharmProjects/Candlestick-Patterns/Candle_Data/Up/{0}{1}.png'.format(
                                    df_pricing['Symbol'][i], i), dpi=70)
                        print('Up', i, n)
                        break
                    else:
                        df.loc[i, 'Trend'] = 'No Trend'
                fig.clf()  # normally I use these lines to release the memory
                plt.close()
                gc.collect()
            except:
                print(i)
                pass


# Training Model
def training_the_model():
    bs = 8
    # bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart
    print("Start training...!")
    path = Path('C:/Users/hishah/PycharmProjects/Candlestick-Patterns/Candle_Data')
    path_save = Path('C:/Users/hishah/PycharmProjects/Candlestick-Patterns/Processed')
    print(path.ls())
    np.random.seed(42)
    trans = get_transforms(max_lighting=0.1, max_zoom=1.05, max_warp=0., max_rotate=3)
    transformation = ([RandTransform(tfm=TfmCrop(crop_pad),
                                     kwargs={'row_pct': (0, 1), 'col_pct': (0, 1), 'padding_mode': 'reflection'}, p=1.0,
                                     resolved={}, do_run=True, is_random=True, use_on_y=True),
                       RandTransform(tfm=TfmPixel(flip_lr), kwargs={}, p=0.5, resolved={}, do_run=True, is_random=True,
                                     use_on_y=True),
                       RandTransform(tfm=TfmLighting(brightness), kwargs={'change': (0.45, 0.55)}, p=0.75, resolved={},
                                     do_run=True, is_random=True, use_on_y=True),
                       RandTransform(tfm=TfmLighting(contrast), kwargs={'scale': (0.9, 1.1111111111111112)}, p=0.75,
                                     resolved={}, do_run=True, is_random=True, use_on_y=True)],
                      [RandTransform(tfm=TfmCrop(crop_pad), kwargs={}, p=1.0, resolved={}, do_run=True, is_random=True,
                                     use_on_y=True)])

    data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
                                      ds_tfms=transformation, size=224, num_workers=4).normalize(imagenet_stats)
    print(data.classes)

    learn = cnn_learner(data, models.resnet34, metrics=error_rate)
    learn.fit_one_cycle(4)
    learn.unfreeze()
    learn.lr_find()
    learn.recorder.plot()
    learn.fit_one_cycle(10, max_lr=slice(1e-6, 1e-4))
    learn.lr_find()
    learn.recorder.plot()
    learn.fit_one_cycle(5, max_lr=slice(1e-6, 1e-4))
    learn.lr_find()
    learn.recorder.plot()
    learn.fit_one_cycle(5, max_lr=slice(1e-5, 1e-4))
    learn.save('First_Model')
    data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
                                      ds_tfms=transformation, size=352, num_workers=4).normalize(imagenet_stats)
    print("data is : \n")
    print(data)
    learn.data = data
    gc.collect()
    torch.cuda.empty_cache()
    '''    
    learn.lr_find()
    learn.recorder.plot()
    learn.fit_one_cycle(5, max_lr=slice(1e-5, 1e-4))
    learn.lr_find()
    learn.recorder.plot()
    learn.fit_one_cycle(5, max_lr=slice(1e-6, 1e-5))
    learn.save('Model_2')
    '''
    # use create model
    correcti = 0
    list_down = []
    list_up = []
    for i in range(len(data.valid_ds)):
        prediction = learn.predict(data.valid_ds[i][0])
        if str(data.valid_ds[i][1]) == "Down" and bool(prediction[1] == 0):
            correcti = correcti + 1
            if prediction[2][0] >= 0.75:
                list_down = list_down + [i]
                # print(prediction)
                # print()
        if str(data.valid_ds[i][1]) == "Up" and bool(prediction[1] == 1):
            correcti = correcti + 1
            if prediction[2][1] >= 0.75:
                list_up = list_up + [i]
                # print(prediction)
                # print()

    idx = 233
    x, y = data.valid_ds[idx]
    x.show()
    print(data.valid_ds.y[idx])
    xb, _ = data.one_item(x)
    xb_im = Image(data.denorm(xb)[0])
    xb = xb.cuda()
    m = learn.model.eval()
    hook_a, hook_g = hooked_backward(m, xb , y)
    acts = hook_a.stored[0].cpu()
    acts.shape
    avg_acts = acts.mean(0)
    avg_acts.shape
    answer = show_heatmap(avg_acts, xb_im)
    answer.savefig(path / 'trial.png')


if __name__ == '__main__':
    # data_gathering()
    training_the_model()
