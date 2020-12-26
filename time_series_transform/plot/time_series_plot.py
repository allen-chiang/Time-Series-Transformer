import numpy as np
from time_series_transform.plot.base import plot_base
import plotly.graph_objects as go
from copy import copy

class TimeSeriesPlot(plot_base):
    def __init__(self, time_series_data):
        super().__init__(time_series_data)

    def create_plot(self, dataCols,title = "", lineType='scatter'):
        for col in dataCols:
            self.add_line(col, lineType)
        
        if self.is_collection:
            buttonList = list()
            visible_array = np.zeros(len(self.category))
            buttonList.append(dict(label = 'All',
                                        method = 'update',
                                        args = [{'visible': visible_array==0},
                                                {'title': 'All',
                                                'showlegend':True}]))
            for indx in range(len(self.category)):
                col = self.category[indx]
                va = copy(visible_array)
                va[indx] = 1
                buttonList.append(dict(label = col,
                                        method = 'update',
                                        args = [{'visible': va==1},
                                                {'title': col,
                                                'showlegend':True}]))


            self.update_layout(
                updatemenus=[go.layout.Updatemenu(
                            active=0,
                            buttons=buttonList
                            )
                        ]
            )
        else:
            self.update_layout(
                title = title,
                xaxis_title = self.time_series.time_seriesIx,
                yaxis_title = "value",
                legend_title = "Legend"
            )

        return self
    
    def add_marker(self,x,y,color,legendName,showlegend=True,marker='circle'):
        self.fig.add_trace(
            go.Scatter(
                mode='markers',
                x=x,
                y=y,
                name=legendName,
                marker=dict(
                    color = color,
                    symbol=marker
                ),
                showlegend=showlegend
            )
        )
        return self

    def __repr__(self):
        self.fig.show()
        return ""
    
def create_plot(time_series_data, dataCols,title = "", type='scatter'):
    tsp = TimeSeriesPlot(time_series_data)
    tsp.create_plot(dataCols,lineType= type)
    return tsp

    