import matplotlib.pyplot as plt
import plotly.express as px
import nbformat

def create_animated_scatter(comparison_data,variable):
    fig = px.scatter(comparison_data, x='reanalysis_lon', y='reanalysis_lat', animation_frame=comparison_data.index, 
                     size=variable, color=variable, 
                     hover_name=comparison_data.index, title='Weather Variables Over Time',
                     labels={'reanalysis_lon': 'Longitude', 'reanalysis_lat': 'Latitude'},
                     range_x=[comparison_data['reanalysis_lon'].min(), comparison_data['reanalysis_lon'].max()],
                     range_y=[comparison_data['reanalysis_lat'].min(), comparison_data['reanalysis_lat'].max()],
                     color_continuous_scale='bluered',
                     size_max=10)  
    fig.update_layout(showlegend=True)
    
    return fig

def plot_comparison_at_time(comparison_data, upper_variables=None, surface_variables=None, levels=None):
    # right now it doesnt take into account the timestamp (index), adjust it !!!!!
    """
    Plots comparison data for reanalysis and forecast.
    
    """
    if surface_variables:
        for surface_variable in surface_variables:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            reanalysis_surface = f'reanalysis_{surface_variable}'
            forecast_surface = f'forecast_{surface_variable}'
            surface_title = f'{surface_variable}'

            ax.scatter(comparison_data[f'reanalysis_lon'], comparison_data[f'reanalysis_lat'], c=comparison_data[reanalysis_surface], cmap='coolwarm', label=f'Reanalysis')
            ax.scatter(comparison_data[f'forecast_lon'], comparison_data[f'forecast_lat'], c=comparison_data[forecast_surface], cmap='coolwarm', marker='x', label=f'Pangu-Weather')

            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(f'Surface Variable: {surface_variable} comparison')
            ax.legend()
            cb = plt.colorbar(ax.get_children()[0], ax=ax, orientation='vertical', label=surface_title)

    if upper_variables and levels:
        for upper_variable in upper_variables:
            for level in levels:
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))

                reanalysis_upper = f'reanalysis_{upper_variable}_{level}' if level else f'reanalysis_{upper_variable}'
                forecast_upper = f'forecast_{upper_variable}_{level}' if level else f'forecast_{upper_variable}'
                upper_title = f'{upper_variable} at {level} hPa' if level else f'{upper_variable}'

                ax.scatter(comparison_data[f'reanalysis_lon'], comparison_data[f'reanalysis_lat'], c=comparison_data[reanalysis_upper], cmap='coolwarm', label=f'Reanalysis')
                ax.scatter(comparison_data[f'forecast_lon'], comparison_data[f'forecast_lat'], c=comparison_data[forecast_upper], cmap='coolwarm', marker='x', label=f'Pangu-Weather')

                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_title(f'Upper Variable: {upper_variable} comparison at {level} hPa')
                ax.legend()
                cb = plt.colorbar(ax.get_children()[0], ax=ax, orientation='vertical', label=upper_title)

    plt.tight_layout()
    plt.show()

def plot_comparison_at_location(comparison_data, lat, lon):
    """
    Plot the comparison data at a specific latitude and longitude.
    """
    filtered_data = comparison_data[(comparison_data['reanalysis_lat'] == lat) & (comparison_data['reanalysis_lon'] == lon)]

    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data.index, filtered_data['reanalysis_temperature_1000'], marker='o', label='Reanalysis')
    plt.plot(filtered_data.index, filtered_data['forecast_temperature_1000'], marker='', label='Pangu-Weather')
    plt.xlabel('Timestamp')
    plt.ylabel('Temperature (1000 hPa)')
    plt.title(f'Temperature (1000 hPa) at Lat={lat}, Lon={lon}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
