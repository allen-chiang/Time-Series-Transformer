import numpy as np
from time_series_transform.plot.base import plot_base
import plotly.graph_objects as go
from copy import copy

class TimeSeriesPlot(plot_base):
    def __init__(self, time_series_data):
        super().__init__(time_series_data)


    def create_plot(self, dataCols,title = "", lineType='scatter',**kwargs):
        """create plot based on the given data columns

        Parameters
        ----------
        dataCols : array
            array of the columns
        title : str, optional
            title of the plot, by default ""
        lineType : str, optional
            type of the line in the plot, by default 'scatter'

        Returns
        -------
        self
            the current TimeSeriesPlot object
        """
        for col in dataCols:
            self.add_line(col = col, lineType = lineType,**kwargs)
        
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
    

    def __call__(self, cols, title="", type="scatter"):
        self.create_plot(cols, title=title, lineType=type)
        return self



    def line(self, cols, title=""):
        """create the line chart of the given columns and title

        Parameters
        ----------
        cols : array
            array of the columns
        title : str, optional
            title of the plot, by default ""

        Returns
        -------
        self
            the current TimeSeriesPlot object
        """
        self.create_plot(cols, title=title, lineType='scatter')
        return self
    
    def bar(self,cols, title=""):
        """create the bar chart of the given columns and title

        Parameters
        ----------
        cols : array
            array of the columns
        title : str, optional
            title of the plot, by default ""

        Returns
        -------
        self
            the current TimeSeriesPlot object
        """
        self.create_plot(cols, title = title, lineType='bar')
        return self


    
def create_plot(time_series_data, dataCols,title = "", type='scatter', **kwargs):
    """create plot based on the given data columns

    Parameters
    ----------
    time_series_data : Time_Series_Data
        data of the plot
    dataCols : array
        array of the columns
    title : str, optional
        title of the plot, by default ""
    lineType : str, optional
        type of the line in the plot, by default 'scatter'

    Returns
    -------
    TimeSeriesPlot
        the TimeSeriesPlot object
    """
    tsp = TimeSeriesPlot(time_series_data)
    tsp.create_plot(dataCols,title,lineType= type,**kwargs)
    return tsp

    