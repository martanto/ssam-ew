from zipfile import ZipFile
from datetime import datetime

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

class SsamEW:
    def __init__(self, zip_filename: str, title: str, start_date: str = None, end_date: str = None,
                 wildcard: str = '.dat', vmin: float = 0.0, vmax: float = 50.0, frequencies: list[float] = None) -> None:

        input_dir, self.output_dir, self.figures_dir = self.check_directory(os.getcwd())

        zip_path = os.path.join(input_dir, zip_filename)
        zip = ZipFile(zip_path, 'r')

        self.zip_filename = zip_filename.split('.')[0]
        self.csv_files = [pd.read_csv(zip.open(text_file.filename), header=None, delimiter=' ')
                     for text_file in zip.infolist() if text_file.filename.endswith(wildcard)]

        self.df: pd.DataFrame = self.get_df()

        if start_date is None:
            start_date = self.df.index[0].strftime('%Y-%m-%d')

        if end_date is None:
            end_date = self.df.index[-1].strftime('%Y-%m-%d')

        if frequencies is None:
            frequencies = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
                                    4.5, 5.0, 5.5, 6.0, 8.0, 10.0, 15.0, 20])

        self.title = title
        self.start_date = start_date
        self.end_date = end_date
        self.vmin = vmin
        self.vmax = vmax
        self.frequencies = frequencies

        csv = os.path.join(self.output_dir, f'{self.zip_filename}_{start_date}_{end_date}.csv')
        self.df.to_csv(csv)

        print('💾 Merged SSAM file(s) saved to : {}'.format(csv))

    def check_directory(self, current_dir: str) -> (str, str, str):
        input_dir = os.path.join(current_dir, 'input')
        os.makedirs(input_dir, exist_ok=True)

        output_dir = os.path.join(current_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)

        figures_dir = os.path.join(current_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)

        return input_dir, output_dir, figures_dir

    def get_df(self) -> pd.DataFrame:
        df = pd.concat(self.csv_files, ignore_index=True)

        df['datetime'] = pd.to_datetime(df[0] + ' ' + df[1], format='%d-%b-%Y %H:%M')
        df.index = pd.to_datetime(df['datetime'])
        df = df.drop(columns=[0, 1, 'datetime'])
        df.columns = range(df.columns.size)
        df = df.dropna()
        df = df.sort_values(by=['datetime'])
        df = df.drop_duplicates(keep='last')

        return df

    def plot(self, save: bool = True, enable_title: bool = True,
             width: int = 16, height: int = 9, interval: int = 1) -> None:
        start_date =  datetime.strptime(self.start_date, '%Y-%m-%d').strftime('%d %b %Y')
        end_date = datetime.strptime(self.end_date, '%Y-%m-%d').strftime('%d %b %Y')

        df = self.df.interpolate('time')

        fig, ax = plt.subplots(figsize=(width, height), layout='constrained')

        if enable_title:
            ax.set_title('{} \n Periode {} - {}'.format(self.title, start_date, end_date), fontsize=14)

        ax.contourf(df.index, self.frequencies, df.values.T,
                    levels=1000, cmap='jet_r', vmin=self.vmin, vmax=self.vmax)

        ax.set_ylabel('Frequency', fontsize=12)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
        ax.set_ylim([0, 20])

        ax.set_xlabel('Datetime', fontsize=12)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.set_xlim(df.first_valid_index(), df.last_valid_index())

        fig.colorbar(
            plt.cm.ScalarMappable(norm=plt.Normalize(vmin=self.vmin, vmax=self.vmax), cmap='jet_r'),
            ax=ax, pad=0.02)

        plt.tick_params(axis='both', which='major', labelsize=10, )
        plt.xticks(rotation=45)

        if save:
            save_path = os.path.join(self.figures_dir, f'ssam_{self.start_date}_{self.end_date}.png')
            fig.savefig(save_path, dpi=300)
            print(f'📈 Graphics saved to {save_path}')
