from bokeh.plotting import figure, curdoc
from bokeh.driving import linear

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show
from bokeh.transform import dodge
from bokeh.models import HoverTool
from bokeh.models import Panel, Tabs
from bokeh.models import NumeralTickFormatter
from bokeh.models import DatetimeTickFormatter

import pandas as pd
from bokeh.layouts import column, row
import math
import json
from collections import Counter
from datetime import datetime


class MonitorService:

    def __init__(self, reference_path, source_path):
        self.realtime = source_path
        self.reference = reference_path
        self.realtime_df = pd.DataFrame({'image_url': [], 'metrics': []})
        self.requests_df = pd.DataFrame({'date': [], 'count': []})

        self.__prepare_reference_data()
        self.__create_dashboard()


    def __prepare_reference_data(self):
        self.ref_df = pd.read_csv(self.reference, index_col=0)
        class_stats_group = self.ref_df.groupby('class')['ids'].count()
        self.ref_stats_df = pd.DataFrame(class_stats_group.to_dict().items())
        self.ref_stats_df.columns = ['class', 'ref_count']

        # normalize
        self.ref_stats_df['norm_ref_count'] =  self.ref_stats_df.ref_count / sum(self.ref_stats_df.ref_count)
        self.ref_stats_df['norm_realtime_count'] = [0] * len(self.ref_stats_df)
        self.ref_stats_df['realtime_count'] = [0] * len(self.ref_stats_df)

    def __create_datadrift_tab(self):
        
        self.drift_source = ColumnDataSource(data=self.ref_stats_df)

        drift_plot = figure(x_range=self.ref_stats_df['class'], width=1240, height=800,
            title='Distribution drift', toolbar_location=None, tools="zoom_in")
        
        drift_plot.add_tools(HoverTool(tooltips=[("Class", "@class"),
            ("RefPercent", "@norm_ref_count{%0.2f}"), ("RealtimePercent", "@norm_realtime_count{%0.2f}"),
            ("RefCount", "@ref_count"), ("RealtimeCount", "@realtime_count"),
            ]))

        drift_plot.vbar(x=dodge('class', -0.4, range=drift_plot.x_range), top='norm_ref_count',
            source=self.drift_source, width=0.4, color='#718dbf', legend_label='Train distribution')
        
        drift_plot.vbar(x=dodge('class', 0, range=drift_plot.x_range), top='norm_realtime_count',
            source=self.drift_source, width=0.4, color='#e84d60', legend_label='Realtime distribution')

        drift_plot.xaxis.major_label_orientation = math.pi/4

        drift_plot.x_range.range_padding = 0.1
        drift_plot.xgrid.grid_line_color = None
        drift_plot.legend.location = "top_left"
        drift_plot.legend.orientation = "horizontal"

        drift_plot.yaxis.formatter = NumeralTickFormatter(format='0%')

        return drift_plot
    

    def __update_distribution_drift(self):
    
        class_names = self.realtime_df['metrics_series'].apply(lambda s: s['top1'])
        class_counter = Counter(class_names)
        realtime_samples_count = len(class_names)
        ref_samples_count = len(self.ref_df)

        for name, count in class_counter.items():
            self.ref_stats_df.loc[(self.ref_stats_df['class'] == name), 'realtime_count'] = (count)

        self.ref_stats_df['norm_realtime_count'] = \
             self.ref_stats_df.realtime_count / sum(self.ref_stats_df.realtime_count)

        self.drift_source.data = self.ref_stats_df

        self.drift_plot.title.text = f'Distribution drift, Reference samples: {ref_samples_count}, Realtime samples: {realtime_samples_count}'

    def __create_requests_plot(self):

        self.requests_source = ColumnDataSource(data=self.requests_df)

        requests_plot = figure(x_axis_type="datetime",
            title='Daily requests', toolbar_location=None, tools="zoom_in") 

        requests_plot.vbar(x='date', top='count', width=10**6 * 4, source=self.requests_source)

        requests_plot.xaxis.major_label_orientation = math.pi/4

        requests_plot.add_tools(HoverTool(tooltips=[("Requests", "@count")]))

        requests_plot.xaxis.formatter = DatetimeTickFormatter(
            hours=["%d %B %Y"],
            days=["%d %B %Y"],
            months=["%d %B %Y"],
            years=["%d %B %Y"],
        )

        return requests_plot

    def __update_requests_plot(self):

        time_groups = self.realtime_df['metrics_series'].apply(lambda s: s['timestamp'])
        time_groups = time_groups.apply(datetime.fromisoformat)
        time_groups = time_groups.apply(lambda x: x.strftime("%d-%m-%Y"))

        date_counter = Counter(time_groups)
        self.requests_df = pd.DataFrame(date_counter.items())
        self.requests_df.columns = ['date', 'count']
        self.requests_df.date = pd.to_datetime(self.requests_df.date)

        self.requests_source.data = self.requests_df

    def __create_dashboard(self):
        
        self.drift_plot = self.__create_datadrift_tab()
        drift_tab = Panel(child=self.drift_plot, title="Distribution drift")

        
        self.requests_plot = self.__create_requests_plot()
        requests_tab = Panel(child=self.requests_plot, title="Requests")

        tabs_group = Tabs(tabs=[drift_tab, requests_tab])

        self.root = column(tabs_group)

        curdoc().add_root(self.root)
        curdoc().add_periodic_callback(self.__update_dashboard, 500)

    def __update_dashboard(self):

        try:
            self.realtime_df = pd.read_csv(self.realtime, delimiter=';')
            self.realtime_df['metrics_series'] = self.realtime_df['metrics'].apply(lambda x: x.replace("'", '"'))
            self.realtime_df['metrics_series'] = self.realtime_df['metrics_series'].apply(json.loads)

        except Exception as e:
            print('Waiting for data...')
            return

        self.__update_distribution_drift()
        self.__update_requests_plot()

# TODO add datepicker

m = MonitorService('./data/birdsy.csv', './data/birdsy_records.csv')
