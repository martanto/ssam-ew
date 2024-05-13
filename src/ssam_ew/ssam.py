from .magma import Plot, colors
from zipfile import ZipFile, ZipInfo
from datetime import datetime
from typing import Tuple
from pathlib import Path
from pandas import DatetimeIndex

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

month_translator = {
    'Mei': 'May',
    'Agu': 'Aug',
    'Okt': 'Oct',
    'Des': 'Dec',
}


class SsamEW:
    def __init__(self, zip_file_location: str, station: str, channel: str, network: str = None, location: str = None,
                 wildcard: str = '.dat', delimiter=' ', combine_data: bool = False, current_dir: str = None, input_dir: str = None) -> None:

        if current_dir is None:
            current_dir = os.getcwd()
        self.current_dir = current_dir

        if input_dir is None:
            input_dir = os.path.join(current_dir, 'input')
            os.makedirs(input_dir, exist_ok=True)

        self.network = 'VG' if network is None else network
        self.station = station
        self.location = '00' if location is None else location
        self.channel = channel

        self.nslc = f'{self.network}.{self.station}.{self.location}.{self.channel}'

        self.wildcard = wildcard
        self.delimiter = delimiter

        self.output_dir, self.figures_dir, self.ssam_dir = self.check_directory(os.getcwd())
        self.extract_dir = self.extract_dir()
        self.filename: str = Path(zip_file_location).stem

        zip_file_location = os.path.join(current_dir, zip_file_location)
        self.files: list = self.get_files(zip_file_location, wildcard, delimiter)

        if combine_data is True:
            self.combine_csvs(self.files)

    def check_directory(self, current_dir: str = None) -> Tuple[str, str, str]:

        if current_dir is None:
            current_dir = self.current_dir

        output_dir = os.path.join(current_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)

        figures_dir = os.path.join(current_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)

        ssam_dir = os.path.join(output_dir, 'ssam', self.nslc)
        os.makedirs(ssam_dir, exist_ok=True)

        return output_dir, figures_dir, ssam_dir

    def extract_dir(self, subdir: str = None, output_dir: str = None) -> str:
        if subdir is None:
            subdir = self.nslc

        if output_dir is None:
            output_dir = self.output_dir

        extract_dir = os.path.join(output_dir, 'extracted', subdir)
        os.makedirs(extract_dir, exist_ok=True)

        return extract_dir

    def combine_csvs(self, csv_files: list[str]) -> str:
        df_list: list = []

        csv_files.sort()

        first_date = Path(csv_files[0]).stem
        end_date_date = Path(csv_files[-1]).stem

        for csv in csv_files:
            df = pd.read_csv(csv)
            if not df.empty:
                df_list.append(df)

        df = pd.concat(df_list, ignore_index=True)
        df = df.sort_values(by=['datetime'])
        df.index = df['datetime']

        filename = f'combined_{first_date}_{end_date_date}'

        extract_dir = self.extract_dir
        save_path = os.path.join(extract_dir, f'{filename}.csv')

        df.to_csv(save_path, index=False)

        return save_path

    def fix_date(self, date: str) -> datetime:
        date = date.title()
        month = date.split('-')[1]
        if month in month_translator.keys():
            date = date.replace(month, month_translator[month])

        date = datetime.strptime(date, '%d-%b-%Y %H:%M')

        return date

    def save_daily_csv(self, df: pd.DataFrame, extract_dir: str = None) -> list[str]:
        if extract_dir is None:
            extract_dir = self.extract_dir

        daily_csvs = []

        for groups in df.groupby(df.index.date):
            date, df = groups
            save_path = os.path.join(extract_dir, f'{date}.csv')
            df.to_csv(save_path)
            daily_csvs.append(save_path)

        return daily_csvs

    def extract_files(self, zip_file: ZipFile, text_file: ZipInfo, delimiter: str = None) -> list[str]:
        if delimiter is None:
            delimiter = self.delimiter

        df = pd.read_csv(zip_file.open(text_file.filename), header=None, delimiter=delimiter)

        df = df.dropna()
        df['datetime'] = df[0] + ' ' + df[1]
        df['datetime'] = df['datetime'].apply(self.fix_date)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by=['datetime'])
        df.index = df['datetime']
        df = df.drop(columns=[0, 1, 'datetime'])
        df = df.drop_duplicates(keep='last')

        daily_csvs = self.save_daily_csv(df)

        return daily_csvs

    def get_files(self, zip_file_location: str = None, wildcard: str = None, delimiter: str = None) -> list[str]:
        if wildcard is None:
            wildcard = self.wildcard

        files = []

        zip_file = ZipFile(zip_file_location, 'r')

        for text_file in zip_file.infolist():
            if text_file.filename.endswith(wildcard):
                files.extend(self.extract_files(zip_file, text_file, delimiter))

        return files

    def get_df(self, dates: DatetimeIndex, resample: str = '1min', extract_dir: str = None) -> pd.DataFrame:
        if extract_dir is None:
            extract_dir = self.extract_dir

        df_list = []

        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            daily_csv = os.path.join(extract_dir, f'{date_str}.csv')
            try:
                df = pd.read_csv(daily_csv, index_col='datetime', parse_dates=True)
                if resample != '1min':
                    df = df.resample(resample).mean()
                df_list.append(df)
            except FileNotFoundError:
                print(f'⚠️ Skip. File not found: {daily_csv}')

        df = pd.concat(df_list)

        start_date = dates[0].strftime('%Y-%m-%d')
        end_date = dates[-1].strftime('%Y-%m-%d')

        save_path = os.path.join(self.ssam_dir, f'ssam_{start_date}_{end_date}_{resample}.csv')
        df.to_csv(save_path)

        print(f'✅ SSAM file saved at {save_path}')
        return df

    def plot_ax(self, ax: plt.Axes, df: pd.DataFrame = None, interval: int = 1, color_map: str = 'jet_r',
                value_min: float = 0.0, value_max: float = 50.0, frequencies: list[float] = None,) -> plt.Axes:

        if frequencies is None:
            frequencies = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
                                    4.5, 5.0, 5.5, 6.0, 8.0, 10.0, 15.0, 20])

        ax.contourf(df.index, frequencies, df.values.T,
                    levels=1000, cmap=color_map, vmin=value_min, vmax=value_max)

        ax.set_ylabel('Frequency', fontsize=12)
        # ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
        ax.set_ylim([0, 20])

        ax.set_xlabel('Datetime', fontsize=12)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.set_xlim(df.first_valid_index(), df.last_valid_index())

        return ax

    def plot(self, start_date: str, end_date: str, resample: str = None, title: str = None, width: int = 12, height: int = 6,
             interval: int = 1, color_map: str = 'jet_r', value_min: float = 0.0, value_max: float = 50.0,
             frequencies: list[float] = None, save: bool = True) -> plt.Figure:

        if resample is None:
            resample = '1min'

        dates: DatetimeIndex = pd.date_range(start_date, end_date, freq="D")

        df = self.get_df(dates, resample)

        fig, ax = plt.subplots(figsize=(width, height), layout='constrained')
        fig.colorbar(
            plt.cm.ScalarMappable(norm=plt.Normalize(vmin=value_min, vmax=value_max), cmap=color_map),
            ax=ax, pad=0.02)

        ax = self.plot_ax(ax, df=df, interval=interval, color_map=color_map, value_min=0.0,
                          value_max=value_max, frequencies=frequencies)

        if title is None:
            title = f'SSAM {self.nslc}'

        ax.set_title('{} \n Periode {} - {}'.format(title, start_date, end_date), fontsize=14)

        plt.tick_params(axis='both', which='major', labelsize=10, )
        plt.xticks(rotation=45)

        if save:
            save_path = os.path.join(self.figures_dir, f'ssam_{start_date}_{end_date}_{resample}.png')
            fig.savefig(save_path, dpi=300)
            print(f'📈 Graphics saved to {save_path}')

        return fig

    def plot_with_magma(self, token: str, volcano_code: str,  start_date: str, end_date: str,
                        resample: str = None, interval: int = 1, color_map: str = 'jet_r',
                        value_min: float = 0.0, value_max: float = 50.0, frequencies: list[float] = None,
                        earthquake_events: str | list[str] = None, width: int = 14, height: int = 12, height_ratios=None):

        if height_ratios is None:
            height_ratios = [2, 1]

        magma_plot = Plot(
            token=token,
            volcano_code=volcano_code,
            start_date=start_date,
            end_date=end_date,
            earthquake_events=earthquake_events,
        )

        df = magma_plot.df
        height = df.columns.size + 1

        fig = plt.figure(figsize=(12, 2 * height), dpi=100)
        (fig_magma, fig_ssam) = fig.subfigures(nrows=2, ncols=1, height_ratios=height_ratios)

        fig_magma.subplots_adjust(hspace=0.0)
        fig_magma.supylabel('Jumlah')
        axs_magma = fig_magma.subplots(nrows=len(df.columns), ncols=1, sharex=True)
        for gempa, column_name in enumerate(df.columns):
            axs_magma[gempa].bar(df.index, df[column_name], width=0.5, label=column_name,
                                 color=colors[column_name], linewidth=0)
            axs_magma[gempa].set_ylim([0, df[column_name].max() * 1.2])

            axs_magma[gempa].legend(loc=2)
            axs_magma[gempa].tick_params(labelbottom=False)
            axs_magma[gempa].yaxis.set_major_locator(mticker.MultipleLocator(1))
            axs_magma[gempa].yaxis.get_major_ticks()[0].label1.set_visible(False)

        if resample is None:
            resample = '1min'

        dates: DatetimeIndex = pd.date_range(start_date, end_date, freq="D")

        df = self.get_df(dates, resample)

        ax_ssam = fig_ssam.subplots(nrows=1, ncols=1)
        self.plot_ax(ax_ssam, df=df, interval=interval, color_map=color_map, value_min=value_min,
                          value_max=value_max, frequencies=frequencies)

        plt.tight_layout()
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.xticks(rotation=60)

        save_path = os.path.join(self.figures_dir, f'ssam_magma_{start_date}_{end_date}_{resample}.png')
        fig.savefig(save_path, dpi=300)
        print(f'📈 Graphics saved to {save_path}')

        plt.show()
