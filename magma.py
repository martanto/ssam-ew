from typing import Tuple
from datetime import datetime
from pandas.errors import EmptyDataError

import os
import json
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


colors: dict[str, str] = {
    'Letusan/Erupsi': '#F44336',
    'Awan Panas Letusan': '#e91e63',
    'Guguran': '#1976d2',
    'Awan Panas Guguran': '#673ab7',
    'Hembusan': '#3f51b5',
    'Tremor Non-Harmonik': '#0d47a1',
    'Tornillo': '#03a9f4',
    'Low Frequency': '#006064',
    'Hybrid/Fase Banyak': '#009688',
    'Vulkanik Dangkal': '#8BC34A',
    'Vulkanik Dalam': '#33691E',
    'Very Long Period': '#827717',
    'Tektonik Lokal': '#F57F17',
    'Terasa': '#FFCA28',
    'Tektonik Jauh': '#FFA726',
    'Double Event': '#ff5722',
    'Getaran Banjir': '#795548',
    'Harmonik': '#607d8b',
    'Tremor Menerus': '#9E9E9E',
}

columns: dict[str, str] = {
    'gempa.letusan_erupsi': 'Letusan/Erupsi',
    'gempa.awan_panas_letusan': 'Awan Panas Letusan',
    'gempa.awan_panas_guguran': 'Awan Panas Guguran',
    'gempa.guguran': 'Guguran',
    'gempa.hembusan': 'Hembusan',
    'gempa.harmonik': 'Harmonik',
    'gempa.tremor_non-_harmonik': 'Tremor Non-Harmonik',
    'gempa.tornillo': 'Tornillo',
    'gempa.low_frequency': 'Low Frequency',
    'gempa.hybrid_fase_banyak': 'Hybrid/Fase Banyak',
    'gempa.vulkanik_dangkal': 'Vulkanik Dangkal',
    'gempa.vulkanik_dalam': 'Vulkanik Dalam',
    'gempa.very_long_period': 'Very Long Period',
    'gempa.tektonik_lokal': 'Tektonik Lokal',
    'gempa.terasa': 'Terasa',
    'gempa.tektonik_jauh': 'Tektonik Jauh',
    'gempa.double_event': 'Double Event',
    'gempa.getaran_banjir': 'Getaran Banjir',
    'gempa.deep_tremor': 'Deep Tremor',
    'gempa.tremor_menerus': 'Tremor Menerus'
}

class Magma:
    url = 'https://magma.esdm.go.id/api/v1/python/magma-var/evaluasi'

    def __init__(self, token: str, volcano_code: str, start_date: str, end_date: str,
                 earthquake_events: str | list[str] = None) -> None:

        self.volcano_code: str = volcano_code
        self.start_date: str = start_date
        self.end_date: str = end_date
        self.output_dir, self.figures_dir = self.check_directory(os.getcwd())

        self.earthquake_events: list[str] = self.validate_earthquake_events(earthquake_events)

        self.json: dict = self.get_json_response(token)
        self.df: pd.DataFrame = self.get_df()
        self.events_not_recorded = self.df.columns[self.df.sum()==0]
        self.filename = 'magma_{}_{}_{}'.format(volcano_code, start_date, end_date)

        if not self.df.empty:
            csv = os.path.join(self.output_dir, "{}.csv".format(self.filename))
            self.df.to_csv(csv)
            print(f'💾 Saved to {csv}')
        else:
            print('⚠️ There is no event(s) between {} and {}. Please change your parameters.'.format(start_date, end_date))
            raise EmptyDataError

    def get_df(self, json: dict = None) -> pd.DataFrame:
        if json is None:
            json = self.json

        df = pd.json_normalize(json['data'])
        df.drop(columns=[
            'availability',
            'visual.visibility',
            'visual.cuaca',
            'visual.asap.teramati',
            'visual.asap.warna',
            'visual.asap.intensitas',
            'visual.asap.tekanan',
            'visual.asap.tinggi_min',
            'visual.asap.tinggi_max',
            'visual.letusan.teramati',
            'visual.letusan.tinggi_min',
            'visual.letusan.tinggi_max',
            'visual.letusan.warna',
            'visual.awan_panas_guguran.teramati',
            'visual.awan_panas_guguran.jarak_min',
            'visual.awan_panas_guguran.jarak_max'
        ], inplace=True)

        df.drop(columns=df.columns[df.sum() == 0], inplace=True)
        df.set_index(keys='date', inplace=True)
        df.index = pd.to_datetime(df.index)

        df.rename(columns=columns, inplace=True)

        if 'Tremor Menerus' in df.columns:
            df.drop(columns=['Tremor Menerus'], inplace=True)

        return df

    def get_json_response(self, token: str) -> dict:
        url = self.url

        payload = json.dumps({
            "start_date": self.start_date,
            "end_date": self.end_date,
            "code_ga": self.volcano_code,
            "gempa": self.earthquake_events
        })

        headers = {
            'Authorization': 'Bearer ' + token,
            'Content-Type': 'application/json'
        }

        start_date_object = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date_object = datetime.strptime(self.end_date, "%Y-%m-%d")

        if start_date_object > end_date_object:
            raise ValueError('End date ({}) must be greater than start date ({})'.format(self.end_date, self.start_date))

        if (end_date_object > datetime.now()) or (start_date_object > datetime.now()):
            raise ValueError('End date or start date must be greter than today ({})'.format(datetime.today().date()))

        try:
            response = requests.request("GET", url, headers=headers, data=payload).json()
        except Exception as e:
            raise ValueError(f'Please check your token or parameters {payload}. Error: {e}')

        if 'code' in response:
            if response['code'] == 401:
                raise ValueError(f'Please update your token at https://magma.esdm.go.id/chambers/token')

        return response

    def validate_earthquake_events(self, earthquake_events: str | list[str] = None) -> list[str]:
        if earthquake_events is None:
            earthquake_events = ['*']

        if isinstance(earthquake_events, str):
            earthquake_events = [earthquake_events]

        for earthquake_event in earthquake_events:
            if earthquake_event not in ['*', 'lts', 'apl', 'apg', 'gug', 'hbs', 'hrm', 'tre',
                                         'tor', 'lof', 'hyb', 'vtb', 'vta', 'vlp', 'tel', 'trs',
                                         'tej', 'dev', 'gtb', 'dpt', 'mtr']:
                raise ValueError("Earthquake_events must be one of '*', 'lts', 'apl', 'apg', 'gug', 'hbs', 'hrm', "
                                 "'tre', 'tor', 'lof', 'hyb', 'vtb', 'vta','vlp', 'tel', 'trs', 'tej', 'dev', 'gtb', "
                                 "'dpt', 'mtr'")

        return earthquake_events

    def check_directory(self, current_dir: str) -> Tuple[str, str]:
        output_dir = os.path.join(current_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)

        figures_dir = os.path.join(current_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)

        return output_dir, figures_dir

    @staticmethod
    def plot_ax(ax: plt.Axes, df: pd.DataFrame, column_name: str, width: float = 0.5, interval = 1) -> plt.Axes:

        ax.bar(df.index, df[column_name], width=width, label=column_name,
               color=colors[column_name], linewidth=0)

        ax.legend(loc=2, fontsize=8)

        ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        ax.yaxis.get_major_ticks()[0].label1.set_visible(False)
        ax.set_xlim(df.first_valid_index(), df.last_valid_index())

        ax.set_ylim([0, df[column_name].max() * 1.2])

        for label in ax.get_xticklabels(which='major'):
            label.set(rotation=30, horizontalalignment='right')

        # for key, continuous in enumerate(continuous_eruptions):
        #     # continuous[0] = start date of eruption
        #     # continuous[1] = end date of eruption
        #     axs[gempa].axvspan(continuous[0], continuous[1], alpha=0.4,
        #                        color='orange', label="_" * key + 'Continuous Eruption')
        #
        # for key, date in enumerate(single_eruptions):
        #     axs[gempa].axvline(datetime.strptime(date, '%Y-%m-%d'),
        #                        color='red', label="_" * key + 'Single Eruption')
        return ax

    def plot(self, width: float = 0.5, interval: int = 1, save_plot: bool = True, dpi: int = 300) -> plt.Figure:
        df = self.df
        fig, axs = plt.subplots(nrows=len(df.columns), ncols=1, figsize=(12, 1 * len(df.columns)), sharex=True)

        plt.subplots_adjust(hspace=0.0)

        for gempa, column_name in enumerate(df.columns):
            Magma.plot_ax(
                ax=axs[gempa],
                df=df,
                column_name=column_name,
                width=width,
                interval=interval
            )

        fig.supylabel('Jumlah', x=0.07)
        fig.suptitle('Kegempaan', fontsize=12, y=0.9)

        if save_plot:
            figure_name = os.path.join(self.figures_dir, f'{self.filename}.png')
            fig.savefig(figure_name, dpi=dpi)

        return fig

    @staticmethod
    def plot_from_csv(csv: str, width: float = 0.5, interval: int = 1, save_plot: bool = True, dpi: int = 300) -> plt.Figure:
        filename = Path(csv).stem

        df = pd.read_csv(csv, index_col='date', parse_dates=True)

        fig, axs = plt.subplots(nrows=len(df.columns), ncols=1, figsize=(12, 1 * len(df.columns)), sharex=True)

        plt.subplots_adjust(hspace=0.0)

        for gempa, column_name in enumerate(df.columns):
            Magma.plot_ax(
                ax = axs[gempa],
                df = df,
                column_name = column_name,
                width = width,
                interval = interval
            )

        fig.supylabel('Jumlah', x=0.07)
        fig.suptitle('Kegempaan', fontsize=12, y=0.9)

        if save_plot:
            figures_dir = os.path.join(os.getcwd(), 'figures')
            os.makedirs(figures_dir, exist_ok=True)

            figure_name = os.path.join(figures_dir, f'{filename}.png')
            fig.savefig(figure_name, dpi=dpi)

        return fig







